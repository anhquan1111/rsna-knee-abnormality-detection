"""Gop dac trung slice -> study va head 12 chieu (ngay 5).

Buoc pooling la cho nhan va du lieu gap nhau: du lieu o cap lat, nhan o cap study. Moi lua
chon pooling la mot gia thuyet khac nhau ve benh hoc:

  * `mean` - bat thuong the hien tren phan lon cac lat (vd: tran dich khop).
  * `max`  - mot lat duy nhat cung du ket luan (vd: rach day chang cheo truoc).
  * `attn` - model tu hoc lat nao dang chu y; linh hoat nhat nhung de khop thuoc nhat voi
             58 study.

Loss BO QUA nhan NaN. Day la dieu kien bat buoc de tron nhan nguoi va nhan may o ngay 8:
nhan may co rat nhieu o "khong ket luan duoc", va coi chung la 0 se day model ve phia am
tinh tren chinh nhung nhan ma bo rut nhan chua phu duoc.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import LABELS


class MaskedPool(nn.Module):
    """Gop (B, K, d) + mask (B, K) -> (B, d), bo qua cac vi tri pad."""

    def __init__(self, mode: str = "mean", dim: int | None = None):
        super().__init__()
        if mode not in {"mean", "max", "attn"}:
            raise ValueError(f"pooling khong hop le: {mode}")
        self.mode = mode
        if mode == "attn":
            if dim is None:
                raise ValueError("pooling 'attn' can biet so chieu dac trung")
            self.score = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(x.dtype)
        if self.mode == "mean":
            return (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.mode == "max":
            # -inf o vi tri pad, khong phai 0: dac trung co the am, lay max voi 0 se sai.
            return x.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        w = self.score(x).masked_fill(~mask.unsqueeze(-1), float("-inf")).softmax(1)
        return (x * w).sum(1)


class StudyHead(nn.Module):
    """LayerNorm -> pooling -> tuyen tinh 12 chieu. Tra ve LOGIT, khong sigmoid."""

    def __init__(self, dim: int, pooling: str = "mean", dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pool = MaskedPool(pooling, dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, len(LABELS))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.pool(self.norm(x), mask)))


def masked_bce(logits: torch.Tensor, targets: torch.Tensor,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCEWithLogits chi tinh tren cac nhan da biet. NaN = chua gan = khong dong gop gradient."""
    known = ~torch.isnan(targets)
    if not known.any():
        return logits.sum() * 0.0
    safe = torch.nan_to_num(targets, nan=0.0)
    per = nn.functional.binary_cross_entropy_with_logits(
        logits, safe, reduction="none", pos_weight=pos_weight)
    return (per * known).sum() / known.sum()


def collate_features(batch):
    """Giong `collate_pad` nhung cho dac trung (K, d) thay vi anh (K, H, W)."""
    xs, ys, ids = zip(*batch)
    k_max = max(x.shape[0] for x in xs)
    padded = torch.stack([
        torch.cat([x, x.new_zeros(k_max - x.shape[0], x.shape[1])]) for x in xs
    ])
    mask = torch.stack([torch.arange(k_max) < x.shape[0] for x in xs])
    return padded, mask, torch.stack(list(ys)), list(ids)


@torch.no_grad()
def predict(model: nn.Module, loader, device: str = "cpu"):
    """Tra ve (y_true, logits, study_ids) theo dung thu tu loader duyet."""
    model.eval()
    ys, ps, ids = [], [], []
    for x, mask, y, sid in loader:
        out = model(x.to(device), mask.to(device))
        ys.append(y.numpy())
        ps.append(out.cpu().numpy())
        ids.extend(sid)
    return np.concatenate(ys), np.concatenate(ps), ids


def train_head(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    log_every: int = 10,
) -> dict:
    """Vong train toi thieu nhung day du: co eval moi epoch va luu lai learning curve."""
    from .metrics import macro_auc

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": []}
    best = {"val_auc": float("-inf"), "val_loss": float("inf"), "epoch": -1, "state": None}

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, mask, y, _ in train_loader:
            opt.zero_grad()
            loss = masked_bce(model(x.to(device), mask.to(device)), y.to(device))
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(y)
            n += len(y)

        y_true, logits, _ = predict(model, val_loader, device)
        val_loss = float(masked_bce(torch.from_numpy(logits), torch.from_numpy(y_true)))
        try:
            auc = macro_auc(y_true, logits, LABELS).macro_auc
        except ValueError:
            auc = float("nan")

        history["epoch"].append(ep)
        history["train_loss"].append(tot / max(n, 1))
        history["val_loss"].append(val_loss)
        history["val_auc"].append(auc)

        # Chon checkpoint theo AUC khi tinh duoc. Nhung voi fold nho, CA MOI epoch deu co
        # the cho AUC = nan (moi nhan trong fold chi con mot lop) - luc do phai co duong lui
        # theo val_loss, neu khong `best["state"]` la None va load_state_dict nem TypeError
        # o mot cho khong lien quan gi toi nguyen nhan that.
        if auc == auc and auc > best["val_auc"]:
            best = {"val_auc": auc, "val_loss": val_loss, "epoch": ep, "state": snapshot()}
        elif best["state"] is None or (best["val_auc"] == float("-inf")
                                       and val_loss < best["val_loss"]):
            best = {**best, "val_loss": val_loss, "epoch": ep, "state": snapshot()}
        if log_every and ep % log_every == 0:
            print(f"    epoch {ep:>3} | train {tot/max(n,1):.4f} | val {val_loss:.4f} | AUC {auc:.4f}")

    return {"history": history, "best": best}
