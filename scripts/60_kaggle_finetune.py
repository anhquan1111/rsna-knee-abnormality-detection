"""NGAY 9 - Fine-tune backbone DINOv2. CHAY TREN KAGGLE NOTEBOOK (Internet = On, GPU).

Vi sao khong chay o may: fine-tune can ANH, khong phai dac trung. May chi co 64 thu muc
anh (665 MB, 16/58 ca gold); ca bo la 569 GB va nam tren Kaggle. Cai CUDA o may cung
khong giai quyet duoc chuyen thieu du lieu.

BA QUYET DINH THIET KE, va vi sao
=================================

1. **Fine-tune CHI tren ca NGOAI gold, bang nhan may.**
   58 ca co nhan bac si khong bao gio duoc dung de cap nhat trong so backbone. Nho vay
   giao thuc danh gia cua ngay 5/8 (5-fold tren gold) van con hieu luc nguyen ven: backbone
   chua he nhin thay dap an cua bat ky ca danh gia nao. Neu fine-tune tren ca gold thi moi
   diem do duoc sau do deu vo nghia - va voi 58 ca thi no se khop thuoc gan nhu tuc thi.

2. **Fine-tune MOT LAN, khong phai mot lan moi fold.**
   He qua truc tiep cua (1): backbone khong dung gold nen no giong nhau o moi fold. Lam
   5 lan chi ton 5x thoi gian ma ra dung mot ket qua.

3. **Cache truoc anh da tien xu ly, roi moi train nhieu epoch.**
   Doc DICOM chiem 86% thoi gian (do duoc o buoc nop bai). Doc lai moi epoch thi mot epoch
   tren 4.349 ca mat ~150 phut - 5 epoch la 12 tieng, khong kha thi. Nen doc MOT LAN ra
   memmap float16 roi cac epoch sau chi doc dia.
   Doi lai: phai gioi han so ca train cho vua /kaggle/working (20 GB).
   1.000 ca x ~95 lat x 224 x 224 x 2 byte = ~9,5 GB.

GIAI DOAN
=========
STAGE = "gold"  (~1-1.5h)  Fine-tune roi CHI trich lai dac trung cho 58 ca gold.
                           Du de tra loi "fine-tune co giup khong" bang cach so
                           voi cau hinh A cua ngay 5 (0.6266). Re gap ~76 lan.
STAGE = "all"   (~3-4h)    Trich lai ca 4.407 ca, de chay duoc cau hinh B cua ngay 8
                           (0.6923). Chi lam neu giai doan 1 co ket qua.

Lam thi nghiem re truoc. Neu fine-tune khong giup o giai doan 1 thi khong co ly do gi
tieu them 3 tieng GPU cho giai doan 2.

CHUAN BI
========
1. Chay `python scripts/49_pack_submission_assets.py` o may - no gop san nguon nhan
   vao data/submit_assets.zip - roi upload zip do len Dataset `rsna-knee-my-assets`.
   Script nay UU TIEN nhan LLM (llm_labels_v4_blend.csv) vi ngay 10 do duoc no hon
   han nhan tu dien: 0.7467 so voi 0.6923 khi dung train head.
2. Notebook: Add Input = competition + dataset do. Settings: GPU On, Internet On.
3. Dan file nay vao mot cell -> Run All.
4. Tai ve 3 file tu /kaggle/working: features_finetuned_*.npz, backbone_finetuned.pt,
   finetune_summary.json - roi o may chay:
       python scripts/61_eval_finetuned.py <thu muc vua tai ve>
"""
from __future__ import annotations

# ============================== CONFIG ==============================
STAGE = "all"              # "gold" = re, "all" = day du (xem tren)
PLANES = ["Sagittal", "Axial", "Coronal"]
SIZE = 224
N_TRAIN_STUDIES = 1000     # so ca ngoai gold dung de fine-tune (gioi han boi dia 20 GB)
N_VAL_STUDIES = 150        # ca ngoai gold de lam val - KHONG duoc dung gold cho viec nay
UNFREEZE_BLOCKS = 4        # so block cuoi cua DINOv2 duoc mo khoa (het la 12)
SLICES_PER_STUDY = 16      # so lat lay ngau nhien moi ca moi epoch
EPOCHS = 8
LR_BACKBONE = 1e-5         # backbone da hoc tot roi, day chi la chinh nhe
LR_HEAD = 1e-3             # head khoi tao ngau nhien nen can lr lon hon nhieu
BATCH_STUDIES = 4
SEED = 42
BATCH_INFER = 64
# ====================================================================

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

COMPETITION = "rsna-knee-abnormality-detection"
OUT = Path("/kaggle/working")

# Cache ~11 GB PHAI nam ngoai /kaggle/working. Moi thu trong /kaggle/working deu duoc
# Kaggle luu lai thanh output cua version - 11 GB cache tam se lam buoc luu version cham
# kinh khung hoac that bai, trong khi thu ta can giu chi la vai chuc MB npz.
# /kaggle/temp la o nhap: mat khi het phien nhung KHONG bi tinh vao output.
_TEMP = Path("/kaggle/temp")
CACHE = (_TEMP if _TEMP.exists() else OUT) / "ft_cache"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- duong dan
def find_root() -> Path:
    """Giong scripts/06: Kaggle dat du lieu o do sau khac nhau tuy cach them input."""
    base = Path("/kaggle/input")
    for cand in (base / "competitions" / COMPETITION, base / COMPETITION):
        if (cand / "train_series.csv").exists():
            return cand
    for pattern in ("*/train_series.csv", "*/*/train_series.csv"):
        for hit in base.glob(pattern):
            return hit.parent
    raise SystemExit(f"Khong thay train_series.csv trong {base}. "
                     f"Co: {sorted(p.name for p in base.glob('*'))[:20]}")


def find_weak_labels() -> tuple[Path, str]:
    """Tim nguon nhan may trong cac dataset da Add Input (bo qua `competitions`).

    UU TIEN nhan LLM. Ngay 10 do duoc no hon han nhan tu dien khi dung de train head:
    0.7467 so voi 0.6923 (KTC cua chenh lech khong chua 0, va tai lap duoc o ca hai kieu
    pooling). Fine-tune backbone bang nguon nhan tot hon thi hop ly hon han - vong truoc
    dung nhan tu dien chi vi luc do chua do.

    Nhan LLM la nhan MEM (211 muc tu 0.005 den 1.0) chu khong phai 0/1. `masked_bce`
    dung duoc thang: BCE nhan target trong [0,1]. Nhan 0.25 cho gradient nho hon nhan
    0.0 - dung la thu ta muon khi nguon nhan khong chac chan.
    """
    base = Path("/kaggle/input")
    goc = [g for g in base.iterdir() if g.is_dir() and g.name != "competitions"]
    for mau, ten in (("llm_labels_v4_blend.csv", "llm"), ("weak_labels.csv", "tu_dien")):
        for g in goc:
            for hit in g.rglob(mau):
                return hit, ten
    raise SystemExit(
        "Khong thay nguon nhan nao trong /kaggle/input "
        "(can llm_labels_v4_blend.csv hoac weak_labels.csv).\n"
        "  O may: python scripts/49_pack_submission_assets.py (da gop san vao zip)\n"
        "  roi upload lai data/submit_assets.zip len dataset rsna-knee-my-assets."
    )


# ---------------------------------------------------------------- tien xu ly
def normalize_series(arr: np.ndarray) -> np.ndarray:
    """Percentile 1-99 tren CA series. Phai giong het scripts/06 va scripts/50.

    `np.percentile` tra ve float64; tru float64 vao mang float32 se NANG ca mang len
    float64 va lam cache phinh gap doi. Ep tung he so ve float32 truoc.
    """
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
        pho_bien = max(shapes, key=lambda s: sum(1 for _, a in items if a.shape == s))
        items = [(i, a) for i, a in items if a.shape == pho_bien]
    return np.stack([a for _, a in items])


def doc_mot_ca(root: Path, rows: pd.DataFrame) -> np.ndarray:
    """Doc het cac mat phang cua mot ca roi NOI LAT lai - dung phep gop cua scripts/07."""
    phan = []
    for _, r in rows.iterrows():
        sdir = root / "train_series" / r["StudyInstanceUID"] / r["SeriesInstanceUID"]
        try:
            phan.append(resize_volume(normalize_series(load_series(sdir)), SIZE))
        except Exception:
            continue          # hong mot mat phang thi hai mat phang kia van dung duoc
    if not phan:
        raise FileNotFoundError("khong doc duoc mat phang nao")
    return np.concatenate(phan)


# ---------------------------------------------------------------- cache
def xay_cache(root: Path, chon: dict[str, pd.DataFrame], ten: str) -> tuple[Path, dict]:
    """Doc anh MOT LAN ra memmap float16 tren dia, tra ve (duong dan, chi muc).

    Vi sao phai co buoc nay: doc DICOM chiem 86% thoi gian chay. Doc lai moi epoch thi
    8 epoch la 8 lan tra cai gia do. Doc mot lan roi cac epoch sau doc dia thi re hon
    han. Doi lai la ton dia, nen so ca train bi gioi han boi 20 GB cua /kaggle/working.

    CHECKPOINT THAT: chi muc duoc ghi lai moi 100 ca, va lan chay sau se GHI TIEP vao
    cuoi file thay vi lam lai tu dau. Buoc nay mat ~35 phut cho 1.000 ca - dung bai hoc
    cua buoc trich dac trung: checkpoint chi co gia tri neu no duoc DOC LAI, va phai doc
    lai duoc ca khi phien chay dut giua chung chu khong chi khi da xong.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    dat = CACHE / f"{ten}.dat"
    idx_path = CACHE / f"{ten}_index.json"

    offsets, hong, n_slice = {}, [], 0
    if dat.exists() and idx_path.exists():
        try:
            cu = json.loads(idx_path.read_text())
            if cu.get("size") == SIZE:
                offsets = {k: list(v) for k, v in cu["offsets"].items()}
                hong = [tuple(h) for h in cu.get("hong", [])]
                n_slice = cu["n_slice"]
                if cu.get("xong"):
                    print(f"  cache {ten} da xong: {cu['n_study']:,} ca, "
                          f"{n_slice:,} lat, {dat.stat().st_size/1024**3:.1f} GB")
                    return dat, cu
                print(f"  cache {ten} chay do: da co {len(offsets):,} ca "
                      f"({n_slice:,} lat) - ghi tiep")
        except Exception as exc:
            print(f"  chi muc cu doc khong duoc ({type(exc).__name__}) - lam lai tu dau")
            offsets, hong, n_slice = {}, [], 0

    can = [u for u in chon if u not in offsets]
    if not can:
        idx = {"xong": True, "n_study": len(offsets), "n_slice": n_slice, "size": SIZE,
               "offsets": offsets, "hong": hong[:50], "n_hong": len(hong), "phut": 0.0}
        idx_path.write_text(json.dumps(idx))
        return dat, idx

    print(f"  dang doc {len(can):,} ca vao cache {ten} ...", flush=True)
    t0 = time.time()
    lat_size = SIZE * SIZE

    def ghi_chi_muc(xong: bool) -> dict:
        d = {"xong": xong, "n_study": len(offsets), "n_slice": n_slice, "size": SIZE,
             "offsets": offsets, "hong": hong[:50], "n_hong": len(hong),
             "phut": round((time.time() - t0) / 60, 1)}
        idx_path.write_text(json.dumps(d))
        return d

    # "r+b" khi ghi tiep de khong xoa phan da doc duoc o lan chay truoc.
    che_do = "r+b" if offsets and dat.exists() else "wb"
    with open(dat, che_do) as f:
        f.seek(n_slice * lat_size * 2)
        f.truncate()          # bo phan ghi do cua lan dut truoc, neu co
        for i, uid in enumerate(can):
            try:
                vol = doc_mot_ca(root, chon[uid]).astype(np.float16)
            except Exception as exc:
                hong.append((uid, f"{type(exc).__name__}: {exc}"))
                continue
            f.write(vol.tobytes())
            offsets[uid] = [n_slice, vol.shape[0]]
            n_slice += vol.shape[0]

            if (i + 1) % 100 == 0:
                f.flush()
                ghi_chi_muc(False)
                el = time.time() - t0
                print(f"    {i+1:>5}/{len(can)} | {el/60:.1f} phut | "
                      f"con ~{el/(i+1)*(len(can)-i-1)/60:.1f} phut | "
                      f"{n_slice*lat_size*2/1024**3:.1f} GB | hong {len(hong)}", flush=True)

    idx = ghi_chi_muc(True)
    print(f"  cache {ten}: {len(offsets):,} ca | {n_slice:,} lat | "
          f"{dat.stat().st_size/1024**3:.1f} GB | {idx['phut']} phut | hong {len(hong)}")
    return dat, idx


class CachedStudies(Dataset):
    """Mot ca = mot chong lat doc tu memmap, lay ngau nhien `n_lat` lat moi lan.

    Lay ngau nhien moi epoch dong hai vai: giam chi phi tinh, va dong thoi la mot dang
    tang cuong du lieu - moi epoch model thay mot to hop lat khac nhau cua cung mot ca.
    """

    def __init__(self, dat: Path, idx: dict, y: dict[str, np.ndarray], n_lat: int,
                 ngau_nhien: bool = True):
        self.mm = np.memmap(dat, dtype=np.float16, mode="r").reshape(-1, idx["size"], idx["size"])
        self.offsets = idx["offsets"]
        self.uids = [u for u in self.offsets if u in y]
        self.y = y
        self.n_lat = n_lat
        self.ngau_nhien = ngau_nhien

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        start, n = self.offsets[uid]
        if n <= self.n_lat:
            lay = np.arange(n)
        elif self.ngau_nhien:
            lay = np.sort(np.random.choice(n, self.n_lat, replace=False))
        else:
            # Val phai TAT DINH, neu khong thi diem val doi moi lan goi va khong the
            # dung de chon epoch tot nhat.
            lay = np.linspace(0, n - 1, self.n_lat).astype(int)
        x = np.asarray(self.mm[start + lay], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(self.y[uid]), uid


def gop_lo(batch):
    xs, ys, ids = zip(*batch)
    k = max(x.shape[0] for x in xs)
    padded = torch.stack([torch.cat([x, x.new_zeros(k - x.shape[0], *x.shape[1:])])
                          for x in xs])
    mask = torch.stack([torch.arange(k) < x.shape[0] for x in xs])
    return padded, mask, torch.stack(ys), list(ids)


# ---------------------------------------------------------------- model
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


class StudyHead(nn.Module):
    """Phai giong HET src/rsna_knee/head.py de checkpoint dung chung duoc."""

    def __init__(self, dim: int, pooling: str = "mean", dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pool = MaskedPool(pooling, dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, len(LABELS))

    def forward(self, x, mask):
        return self.fc(self.drop(self.pool(self.norm(x), mask)))


def masked_bce(logits, targets):
    """Bo qua vi tri NaN. Nhan may chi phu 30,7% cap (ca, nhan) - ep phan con lai ve 0
    se nhoi hang loat nhan sai, ngay 8 da do duoc cai gia do."""
    biet = ~torch.isnan(targets)
    if not biet.any():
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[biet], targets[biet])


def mo_khoa(backbone, n_block: int) -> int:
    """Mo khoa `n_block` block cuoi + norm cuoi. Tra ve so tham so duoc train.

    Vi sao khong mo het 12 block: 1.000 ca voi nhan phu 30,7% la rat it so voi 21 trieu
    tham so cua ViT-S. Mo het thi khop thuoc truoc khi hoc duoc gi. Cac block dau giu
    dac trung thi giac chung (canh, ket cau) - thu do khong can hoc lai tu anh MRI goi.
    """
    for p in backbone.parameters():
        p.requires_grad_(False)
    mo = list(backbone.blocks[-n_block:]) + [backbone.norm]
    for m in mo:
        for p in m.parameters():
            p.requires_grad_(True)
    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)


def chuan_hoa(x: torch.Tensor) -> torch.Tensor:
    """(B,K,H,W) -> (B*K,3,H,W) da chuan hoa theo ImageNet."""
    b, k, h, w = x.shape
    x = x.reshape(b * k, 1, h, w).repeat(1, 3, 1, 1)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


# ---------------------------------------------------------------- chay
def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trung binh AUC cac nhan cham duoc. Nhan chi co mot lop bi BO, khong tinh la 0.5."""
    from sklearn.metrics import roc_auc_score

    diem = []
    for j in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, j])
        yt = y_true[m, j]
        if m.sum() < 2 or len(np.unique(yt)) < 2:
            continue
        diem.append(roc_auc_score(yt, y_score[m, j]))
    return float(np.mean(diem)) if diem else float("nan")


@torch.no_grad()
def cham_val(backbone, head, loader, device: str) -> float:
    backbone.eval(); head.eval()
    ys, ps = [], []
    for x, mask, y, _ in loader:
        x = chuan_hoa(x.to(device))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = backbone(x).float().reshape(mask.shape[0], mask.shape[1], -1)
            logit = head(f, mask.to(device))
        ys.append(y.numpy()); ps.append(logit.float().cpu().numpy())
    return macro_auc(np.concatenate(ys), np.concatenate(ps))


@torch.no_grad()
def trich(vol: np.ndarray, backbone, device: str) -> np.ndarray:
    """(N,H,W) trong [0,1] -> (N,384). Dung y het scripts/06 de so sanh duoc."""
    x = torch.from_numpy(vol).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    outs = []
    for i in range(0, len(x), BATCH_INFER):
        chunk = x[i:i + BATCH_INFER].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            outs.append(backbone(chunk).float().cpu())
    return torch.cat(outs).numpy().astype(np.float32)


def main() -> None:
    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CANH BAO: khong thay GPU. Panel ben phai -> Settings -> Accelerator = GPU.")
    root = find_root()
    print(f"device {device} | du lieu {root}")

    # --- du lieu ---
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(root / "train_series.csv",
                         dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    weak_path, nguon_nhan = find_weak_labels()
    weak = pd.read_csv(weak_path, dtype={"StudyInstanceUID": str})
    wv = weak[LABELS].to_numpy(dtype=float)
    print(f"nguon nhan may: {nguon_nhan} ({weak_path.name}) | "
          f"do phu {np.isfinite(wv).mean():.1%} | "
          f"{len(np.unique(wv[np.isfinite(wv)]))} muc gia tri")

    co_nhan = train[LABELS].notna().any(axis=1)
    gold_ids = set(train.loc[co_nhan, "StudyInstanceUID"])
    print(f"ca co nhan bac si: {len(gold_ids)} | tong: {len(train):,}")

    # Chan duong ro ri da tim ra o ngay 8: nhan may suy tu bao cao, nen mot ca ngoai gold
    # co bao cao TRUNG TUNG KY TU voi ca gold se mang dap an cua ca gold do vao train.
    # O day backbone dung chung cho moi fold nen phai loai theo TOAN BO gold, khong chi val.
    bam = (train["Report"].astype(str).str.strip().str.lower()
           .map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()))
    train = train.assign(rk=bam)
    rk_gold = set(train.loc[train["StudyInstanceUID"].isin(gold_ids), "rk"])
    ngoai = train[~train["StudyInstanceUID"].isin(gold_ids)]
    truoc = len(ngoai)
    ngoai = ngoai[~ngoai["rk"].isin(rk_gold)]
    print(f"ca ngoai gold: {truoc:,} -> {len(ngoai):,} "
          f"(bo {truoc - len(ngoai)} ca trung bao cao voi gold)")

    # --- nhan may, chi giu ca co it nhat mot nhan ket luan duoc ---
    w = weak.set_index("StudyInstanceUID")
    y_weak = {}
    for uid in ngoai["StudyInstanceUID"]:
        if uid in w.index:
            v = w.loc[uid, LABELS].to_numpy(dtype=np.float32)
            if np.isfinite(v).any():
                y_weak[uid] = v
    print(f"ca ngoai gold co it nhat 1 nhan may: {len(y_weak):,}")

    rng = np.random.default_rng(SEED)
    chon_uid = list(y_weak)
    rng.shuffle(chon_uid)
    tr_uid = chon_uid[:N_TRAIN_STUDIES]
    va_uid = chon_uid[N_TRAIN_STUDIES:N_TRAIN_STUDIES + N_VAL_STUDIES]
    print(f"fine-tune tren {len(tr_uid):,} ca | val {len(va_uid):,} ca "
          f"(val cung la nhan may - gold KHONG duoc dung o day)")

    # --- chon series: 1 series moi mat phang, giong luc trich dac trung ---
    def theo_ca(uids: list[str]) -> dict[str, pd.DataFrame]:
        s = series[series["StudyInstanceUID"].isin(set(uids))]
        phan = [g.groupby("StudyInstanceUID", as_index=False).first()
                for pl in PLANES
                for g in [s[s["Anatomical_Plane"] == pl]] if len(g)]
        got = pd.concat(phan, ignore_index=True) if phan else s.head(0)
        return {u: g for u, g in got.groupby("StudyInstanceUID")}

    print("\n=== 1. Cache anh (doc DICOM mot lan) ===")
    dat_tr, idx_tr = xay_cache(root, theo_ca(tr_uid), "train")
    dat_va, idx_va = xay_cache(root, theo_ca(va_uid), "val")

    # --- model ---
    print("\n=== 2. Fine-tune ===")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    n_train_param = mo_khoa(backbone, UNFREEZE_BLOCKS)
    backbone.to(device)
    head = StudyHead(384, pooling="mean").to(device)
    tong = sum(p.numel() for p in backbone.parameters())
    print(f"  mo khoa {UNFREEZE_BLOCKS}/12 block cuoi: "
          f"{n_train_param/1e6:.1f}M / {tong/1e6:.1f}M tham so backbone")

    tl = DataLoader(CachedStudies(dat_tr, idx_tr, y_weak, SLICES_PER_STUDY, True),
                    batch_size=BATCH_STUDIES, shuffle=True, collate_fn=gop_lo)
    vl = DataLoader(CachedStudies(dat_va, idx_va, y_weak, SLICES_PER_STUDY, False),
                    batch_size=BATCH_STUDIES, shuffle=False, collate_fn=gop_lo)

    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
        {"params": head.parameters(), "lr": LR_HEAD},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * max(len(tl), 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    tot_nhat, best_state, lich_su = -1.0, None, []
    for ep in range(1, EPOCHS + 1):
        backbone.train(); head.train()
        t0, tong_loss, n = time.time(), 0.0, 0
        for x, mask, y, _ in tl:
            x = chuan_hoa(x.to(device, non_blocking=True))
            mask, y = mask.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                f = backbone(x).reshape(mask.shape[0], mask.shape[1], -1)
                loss = masked_bce(head(f.float(), mask), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tong_loss += loss.detach().item() * len(y); n += len(y)

        auc = cham_val(backbone, head, vl, device)
        lich_su.append({"epoch": ep, "loss": tong_loss / max(n, 1), "val_auc": auc,
                        "phut": round((time.time() - t0) / 60, 1)})
        moc = ""
        if auc > tot_nhat:
            tot_nhat = auc
            best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            moc = "  <- tot nhat"
        print(f"  epoch {ep}/{EPOCHS} | loss {tong_loss/max(n,1):.4f} | "
              f"val AUC (nhan may) {auc:.4f} | {(time.time()-t0)/60:.1f} phut{moc}", flush=True)

    if best_state is not None:
        backbone.load_state_dict(best_state)
    else:
        # Xay ra khi MOI epoch deu cho val AUC = nan (tap val khong co nhan nao du hai
        # lop de cham). Van chay tiep duoc, nhung phai noi ro la dang giu epoch CUOI chu
        # khong phai epoch tot nhat - im lang o day la dung kieu loi da dinh o ngay 5.
        print("  CANH BAO: moi epoch deu cho val AUC = nan (tap val khong cham duoc).")
        print("  Dang giu trong so cua epoch CUOI, khong phai epoch tot nhat.")
        print(f"  Tang N_VAL_STUDIES (dang {N_VAL_STUDIES}) roi chay lai neu can chon epoch.")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    torch.save({"state_dict": backbone.state_dict(), "arch": "dinov2_vits14",
                "unfreeze_blocks": UNFREEZE_BLOCKS, "val_auc_weak": tot_nhat,
                "nguon_nhan": nguon_nhan,
                "n_train_studies": len(tr_uid), "epochs": EPOCHS, "lich_su": lich_su},
               OUT / "backbone_finetuned.pt")
    print(f"  ghi backbone_finetuned.pt | val AUC tot nhat (nhan may) {tot_nhat:.4f}")

    # --- trich lai dac trung bang backbone da fine-tune ---
    print(f"\n=== 3. Trich lai dac trung (STAGE = {STAGE}) ===")
    if STAGE == "gold":
        can = sorted(gold_ids)
    else:
        can = sorted(set(train["StudyInstanceUID"]))
    print(f"  {len(can):,} ca can trich")

    chon = theo_ca(can)
    feats, hong = {}, []
    t0 = time.time()
    for i, uid in enumerate(can):
        if uid not in chon:
            hong.append((uid, "khong co series nao"))
            continue
        try:
            feats[uid] = trich(doc_mot_ca(root, chon[uid]), backbone, device)
        except Exception as exc:
            hong.append((uid, f"{type(exc).__name__}: {exc}"))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"    {i+1:>5}/{len(can)} | {el/60:.1f} phut | "
                  f"con ~{el/(i+1)*(len(can)-i-1)/60:.1f} phut | hong {len(hong)}", flush=True)

    npz = OUT / f"features_finetuned_{STAGE}.npz"
    np.savez_compressed(npz, **feats)
    tom_tat = {"stage": STAGE, "nguon_nhan": nguon_nhan, "n_study": len(feats),
               "n_gold": len(set(feats) & gold_ids),
               "n_slice": int(sum(v.shape[0] for v in feats.values())),
               "n_hong": len(hong), "hong": hong[:20],
               "val_auc_weak": tot_nhat, "unfreeze_blocks": UNFREEZE_BLOCKS,
               "n_train_studies": len(tr_uid), "epochs": EPOCHS,
               "phut_tong": round((time.time() - t_start) / 60, 1),
               "lich_su": lich_su}
    (OUT / "finetune_summary.json").write_text(json.dumps(tom_tat, indent=2))

    print(f"\n=== XONG ({tom_tat['phut_tong']} phut) ===")
    print(f"  {npz.name}  ({npz.stat().st_size/1024**2:.0f} MB)")
    print(f"  {len(feats):,} ca | {tom_tat['n_gold']}/58 gold | "
          f"{tom_tat['n_slice']:,} lat | hong {len(hong)}")
    print("\nTai ve 3 file: features_finetuned_*.npz, backbone_finetuned.pt, "
          "finetune_summary.json")
    print("Roi o may chay:")
    print("  python scripts/61_eval_finetuned.py <thu muc vua tai ve>")


if __name__ == "__main__":
    main()
