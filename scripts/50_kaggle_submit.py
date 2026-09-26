"""CHAY TREN KAGGLE NOTEBOOK de NOP BAI - khong chay o may local.

Cuoc thi la kernel-only: khong nop file CSV, ma nop mot NOTEBOOK. Kaggle chay notebook
tren tap test an, va notebook phai ghi ra /kaggle/working/submission.csv.

RANG BUOC QUAN TRONG NHAT: notebook nop bai KHONG CO INTERNET.
`torch.hub.load("facebookresearch/dinov2", ...)` se that bai. Vi vay phai dong goi san
trong so + ma nguon DINOv2 + checkpoint head thanh mot Kaggle Dataset roi Add Input.

CHUAN BI (lam mot lan)
----------------------
1. O may, chay `python scripts/49_pack_submission_assets.py` -> tao thu muc
   `data/submit_assets/` gom:
       dinov2_repo/                     ma nguon DINOv2 (tu cache torch.hub)
       dinov2_vits14_pretrain.pth       trong so backbone (~84 MB)
       head_dinov2_vits14_mean.pt       head da train (~24 KB)
2. Kaggle -> Datasets -> New Dataset -> upload ca thu muc do. Dat ten vi du
   `rsna-knee-my-assets`.
3. Notebook: Add Input -> chon dataset do. Settings: Accelerator = GPU, Internet = OFF.
4. Dan file nay vao mot cell -> Run All -> kiem submission.csv -> Submit.

NOP LAI
-------
Toi da 5 lan/ngay. Han cuoi 22/10/2026. Cuoi cuoc thi tu chon bai tinh diem private
(xem tab Submissions). Nen cu nop thoai mai, moi lan nop la mot lan do that.
"""
from __future__ import annotations

# ============================== CONFIG ==============================
SIZE = 224                 # phai khop voi luc train
BATCH = 64                 # so lat moi lo qua backbone
POOLING = "mean"           # phai khop checkpoint; doc lai tu file nen day chi la du phong
# Ba mat phang, giong het luc train. Moi study lay MOT series moi mat phang roi NOI CAC
# LAT lai thanh mot chong duy nhat - dung phep gop da dung o scripts/07. Thu tu khong
# quan trong (masked pooling mean/max/attn deu bat bien voi hoan vi) nhung van giu dung
# thu tu luc train cho de doi chieu khi co su co.
PLANES = ["Sagittal", "Axial", "Coronal"]
DEBUG_N = 0                # >0 = chi chay N study dau, de thu nhanh truoc khi nop
# ====================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COMPETITION = "rsna-knee-abnormality-detection"
OUT = Path("/kaggle/working")


# ---------------------------------------------------------------- tim duong dan
def find_comp_root() -> Path:
    base = Path("/kaggle/input")
    for cand in (base / "competitions" / COMPETITION, base / COMPETITION):
        if (cand / "test_series.csv").exists():
            return cand
    for pattern in ("*/test_series.csv", "*/*/test_series.csv"):
        for hit in base.glob(pattern):
            return hit.parent
    raise SystemExit(f"Khong thay test_series.csv trong {base}. "
                     f"Co: {sorted(p.name for p in base.glob('*'))[:20]}")


def find_assets() -> tuple[Path, Path, Path]:
    """Tim ma nguon DINOv2, trong so backbone va checkpoint head trong /kaggle/input.

    Notebook nop bai khong co Internet, nen ba thu nay phai duoc upload san lam Dataset.
    Bao loi that ro neu thieu - day la cho hay hong nhat khi nop lan dau.
    """
    base = Path("/kaggle/input")

    # Kaggle dat dataset o do sau KHAC NHAU tuy cach them input:
    #   /kaggle/input/<slug>/...
    #   /kaggle/input/datasets/<user>/<slug>/...
    # Nen khong duoc dem do sau co dinh. Quet de quy, nhung PHAI bo qua thu muc
    # `competitions` - trong do co 819,640 file anh, rglob vao se treo vai phut.
    goc = [d for d in base.iterdir() if d.is_dir() and d.name != "competitions"]

    def tim(mau: str):
        for g in goc:
            for hit in g.rglob(mau):
                return hit
        return None

    hub = tim("hubconf.py")
    repo = hub.parent if hub else None
    w = tim("dinov2_vits14_pretrain.pth")
    head = tim("head_*.pt")
    # Neu dataset co nhieu head, rglob tra ve cai nao la ngau nhien -> co the nop bang
    # head cu cua vong truoc ma khong co dau hieu gi. Tha bao loi con hon nop nham.
    moi_head = [h for g in goc for h in g.rglob("head_*.pt")]
    if len(moi_head) > 1:
        raise SystemExit(f"Co {len(moi_head)} head trong asset: "
                         f"{[h.name for h in moi_head]}. Xoa bot, chi de lai mot cai.")

    thieu = [n for n, v in (("ma nguon DINOv2 (hubconf.py)", repo),
                            ("dinov2_vits14_pretrain.pth", w),
                            ("head_*.pt", head)) if v is None]
    if thieu:
        # Liet ke thu that su co (tru competitions) de chan doan ngay, thay vi chi in
        # ten thu muc cap 1 - lan truoc chi thay ['competitions', 'datasets'], vo ich.
        co: list[str] = []
        for g in goc:
            co += [str(x.relative_to(base)) for x in list(g.rglob("*"))[:40]]
        raise SystemExit(
            f"Thieu asset trong /kaggle/input: {thieu}\n"
            f"  Ngoai `competitions`, dang co: {co[:25]}\n"
            "  Chay scripts/49_pack_submission_assets.py o may, upload\n"
            "  data/submit_assets.zip len Kaggle Datasets, roi Add Input vao notebook nay."
        )
    return repo, w, head


# ---------------------------------------------------------------- tien xu ly
def chon_series(ts: "pd.DataFrame", planes: list[str]):
    """Moi study lay MOT series moi mat phang - phai khop y het luc train.

    Tra ve (bang series duoc chon, tap study khong co mat phang nao trong `planes`).

    Tach rieng ra khoi `main()` de test duoc o may: day la doan de hong mot cach am
    tham nhat. Chon sai mat phang thi model van chay, van ra file nop hop le, chi co
    diem la thap - va khong co gi bao la vi sao.
    """
    chon = []
    for pl in planes:
        g = ts[ts["Anatomical_Plane"] == pl]
        if len(g):
            chon.append(g.groupby("StudyInstanceUID", as_index=False).first())
    want = pd.concat(chon, ignore_index=True) if chon else ts.head(0).copy()

    # Study khong co mat phang nao trong `planes` van phai co du doan - lay bat ky series
    # nao con lai. Bo trang mot dong la mat diem dong do, khong phai mat diem mo hinh.
    thieu = set(ts["StudyInstanceUID"]) - set(want["StudyInstanceUID"])
    if thieu:
        want = pd.concat([want, ts[ts["StudyInstanceUID"].isin(thieu)]
                          .groupby("StudyInstanceUID", as_index=False).first()],
                         ignore_index=True)
    return want, thieu


def normalize_series(arr: np.ndarray) -> np.ndarray:
    """Percentile 1-99 tren ca series. PHAI giong het luc train, khong duoc lech mot chi tiet."""
    p_low, p_high = (np.float32(v) for v in np.percentile(arr, (1, 99)))
    out = np.clip(arr, p_low, p_high).astype(np.float32)
    return (out - p_low) / (p_high - p_low + np.float32(1e-8))


def resize_volume(arr: np.ndarray, size: int) -> np.ndarray:
    out = np.empty((arr.shape[0], size, size), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.asarray(Image.fromarray(arr[i]).resize((size, size), Image.BILINEAR),
                            dtype=np.float32)
    return out


def load_series(series_dir: Path) -> np.ndarray:
    """Doc series, sap theo InstanceNumber, bo qua file hong.

    Khac ban luc train mot diem quan trong: o day KHONG duoc phep nem loi. Mot ca loi la
    mat diem ca ca do, nhung mot exception khong bat duoc la mat TOAN BO bai nop.
    """
    items = []
    for p in sorted(series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(p))
            items.append((int(getattr(ds, "InstanceNumber", -1)), ds.pixel_array))
        except Exception:
            continue
    if not items:
        raise FileNotFoundError(f"khong doc duoc lat nao trong {series_dir}")

    items.sort(key=lambda t: t[0])
    shapes = {a.shape for _, a in items}
    if len(shapes) > 1:
        # Series nhieu kich thuoc: giu nhom lat co kich thuoc pho bien nhat thay vi bo ca ca
        pho_bien = max(shapes, key=lambda s: sum(1 for _, a in items if a.shape == s))
        items = [(i, a) for i, a in items if a.shape == pho_bien]
    return np.stack([a for _, a in items])


# ---------------------------------------------------------------- model
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MaskedPool(nn.Module):
    def __init__(self, mode: str, dim: int | None = None):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.score = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        if self.mode == "mean":
            return (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.mode == "max":
            return x.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        w = self.score(x).masked_fill(~mask.unsqueeze(-1), float("-inf")).softmax(1)
        return (x * w).sum(1)


class PerLabelAttnPool(nn.Module):
    """Moi benh mot bo trong so chu y rieng: (B,K,d) -> (B,L,d). Giong src/rsna_knee/head.py."""

    def __init__(self, dim: int, n_labels: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, n_labels))

    def forward(self, x, mask):
        w = self.score(x).masked_fill(~mask.unsqueeze(-1), float("-inf")).softmax(dim=1)
        return torch.einsum("bkl,bkd->bld", w, x)


class StudyHead(nn.Module):
    """Phai giong HET src/rsna_knee/head.py, neu khong `load_state_dict` se bao thieu khoa.

    Day la ban SAO CHEP co chu y: notebook nop bai khong import duoc package cua repo.
    Moi lan doi kien truc o src/rsna_knee/head.py deu phai chep sang day - va neu quen,
    `load_state_dict` se bao thieu khoa NGAY, chu khong am tham. Do la ly do KHONG dung
    `strict=False` o duoi.
    """

    def __init__(self, dim: int, pooling: str = "mean", dropout: float = 0.2):
        super().__init__()
        self.pooling = pooling
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        if pooling == "per_label_attn":
            self.pool = PerLabelAttnPool(dim, len(LABELS))
            self.w = nn.Parameter(torch.empty(len(LABELS), dim))
            self.b = nn.Parameter(torch.zeros(len(LABELS)))
        else:
            self.pool = MaskedPool(pooling, dim)
            self.fc = nn.Linear(dim, len(LABELS))

    def forward(self, x, mask):
        pooled = self.drop(self.pool(self.norm(x), mask))
        if self.pooling == "per_label_attn":
            return (pooled * self.w).sum(-1) + self.b
        return self.fc(pooled)


@torch.no_grad()
def encode(slices: np.ndarray, backbone, device: str) -> torch.Tensor:
    x = torch.from_numpy(slices).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    outs = []
    for i in range(0, len(x), BATCH):
        chunk = x[i:i + BATCH].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            outs.append(backbone(chunk).float().cpu())
    return torch.cat(outs)


# ---------------------------------------------------------------- chay
def main() -> None:
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = find_comp_root()
    repo, w_path, head_path = find_assets()
    print(f"device {device} | du lieu {root}")
    print(f"assets: repo={repo.name} weights={w_path.name} head={head_path.name}")

    # --- backbone, nap OFFLINE tu dataset ---
    backbone = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=False)
    backbone.load_state_dict(torch.load(str(w_path), map_location="cpu"))
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    # --- head da train ---
    blob = torch.load(str(head_path), map_location="cpu", weights_only=True)
    if blob.get("labels") and blob["labels"] != LABELS:
        raise SystemExit(f"Thu tu nhan trong checkpoint KHAC config:\n"
                         f"  checkpoint: {blob['labels']}\n  o day     : {LABELS}\n"
                         "  Nop tiep se ra submission sai toan bo ma khong bao loi.")
    head = StudyHead(blob["dim"], pooling=blob.get("pooling", POOLING))
    head.load_state_dict(blob["state_dict"])
    head.eval().to(device)
    print(f"head: {blob['dim']}d, pooling {blob.get('pooling')}, "
          f"OOF macro AUC luc train {blob.get('oof_macro_auc')}")

    # --- chon 1 series moi study test, giong luc train ---
    # dtype=str cho cot UID: pandas doc UID toan chu so thanh so nguyen, sau do
    # phep noi duong dan nem TypeError - hoac te hon, mat so 0 dau ma khong bao gi.
    ts = pd.read_csv(root / "test_series.csv",
                     dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    want, thieu = chon_series(ts, PLANES)
    uids = sorted(set(want["StudyInstanceUID"]))
    if DEBUG_N:
        uids = uids[:DEBUG_N]
        want = want[want["StudyInstanceUID"].isin(uids)]
    theo_study = {u: g for u, g in want.groupby("StudyInstanceUID")}
    print(f"test: {len(ts):,} series -> {len(uids):,} study, "
          f"{len(want):,} series dung ({len(want)/max(len(uids),1):.2f} series/study)"
          + (f" | {len(thieu)} ca khong co mat phang nao trong {PLANES}" if thieu else ""))

    # --- suy luan ---
    rows, loi, t_io, t_gpu, n_lat = [], [], 0.0, 0.0, 0
    for i, uid in enumerate(uids):
        try:
            phan = []
            for _, r in theo_study[uid].iterrows():
                sdir = root / "test_series" / uid / r["SeriesInstanceUID"]
                try:
                    t0 = time.time()
                    arr = resize_volume(normalize_series(load_series(sdir)), SIZE)
                    t_io += time.time() - t0
                    t0 = time.time()
                    phan.append(encode(arr, backbone, device))
                    t_gpu += time.time() - t0
                except Exception as exc:
                    # Hong MOT mat phang khong duoc lam hong ca study: hai mat phang kia
                    # van du de du doan. Chi khi ca ba deu hong moi rot xuong 0.5.
                    loi.append((uid, r["SeriesInstanceUID"], f"{type(exc).__name__}: {exc}"))
            if not phan:
                raise RuntimeError("khong doc duoc mat phang nao")

            feat = torch.cat(phan).unsqueeze(0)
            n_lat += feat.shape[1]
            mask = torch.ones(1, feat.shape[1], dtype=torch.bool)
            t0 = time.time()
            with torch.no_grad():
                prob = torch.sigmoid(head(feat.to(device), mask.to(device)))[0].cpu().numpy()
            t_gpu += time.time() - t0
            rows.append([uid, *prob.astype(float)])
        except Exception as exc:
            # Ca nao loi thi dien 0.5 - KHONG duoc de mot ca lam vo ca bai nop.
            loi.append((uid, "-", f"{type(exc).__name__}: {exc}"))
            rows.append([uid, *([0.5] * len(LABELS))])

        if (i + 1) % 100 == 0:
            el = time.time() - t_start
            print(f"  {i+1:>5}/{len(uids)} | {el/60:.1f} phut | "
                  f"con ~{el/(i+1)*(len(uids)-i-1)/60:.1f} phut | "
                  f"{n_lat/(i+1):.0f} lat/study | loi {len(loi)}", flush=True)

    sub = pd.DataFrame(rows, columns=["StudyInstanceUID", *LABELS])

    # --- kiem truoc khi ghi: nhung loi nay neu lot ra se mat diem ma khong biet vi sao ---
    assert list(sub.columns) == ["StudyInstanceUID", *LABELS], "sai thu tu cot"
    assert sub["StudyInstanceUID"].is_unique, "co study bi trung"
    assert set(sub["StudyInstanceUID"]) == set(uids), "thieu/thua study"
    v = sub[LABELS].to_numpy()
    assert np.isfinite(v).all(), "co gia tri nan/inf"
    assert (v >= 0).all() and (v <= 1).all(), "gia tri ngoai [0,1]"

    sub.to_csv(OUT / "submission.csv", index=False)

    el = time.time() - t_start
    print("\n=== XONG ===")
    print(sub.head(3).to_string())
    print(json.dumps({
        "n_study": len(sub), "n_loi": len(loi),
        "phut_tong": round(el / 60, 1),
        "giay_moi_study": round(el / max(len(sub), 1), 2),
        "phan_tram_doc_anh": round(100 * t_io / max(t_io + t_gpu, 1e-9)),
        "phan_tram_model": round(100 * t_gpu / max(t_io + t_gpu, 1e-9)),
        "do_lech_chuan_du_doan": round(float(v.std(axis=0).mean()), 4),
    }, indent=2))
    if float(v.std(axis=0).mean()) < 0.01:
        print("\nCANH BAO: du doan gan nhu giong nhau cho moi study -> AUC se ~0.5.")
        print("Kiem lai head da train chua.")
    # Doan in loi nay chay SAU khi submission.csv da ghi xong. Mot exception o day se
    # lam ca notebook bi danh FAILED va Kaggle tu choi bai nop - du file ket qua da dung.
    # Nen bat het, khong de doan bao cao lam hong thu no dang bao cao.
    try:
        if loi:
            print(f"\n{len(loi)} loi (ca hong da dien 0.5), vi du:")
            for uid, series, e in loi[:5]:
                print(f"  ...{str(uid)[-14:]} / ...{str(series)[-8:]}  {str(e)[:80]}")
    except Exception as exc:
        print(f"(khong in duoc danh sach loi: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
