"""DO TOC DO cac backbone tren GPU Kaggle. CHAY TREN KAGGLE (GPU On, Internet On).

Chay ~10 phut. Dan vao mot cell -> Run All -> doc bang o cuoi.

VI SAO CAN BUOC NAY
===================
Chan doan o may cho thay nut that nam o DAC TRUNG chu khong o head: them nang luc phan
loai (HistGradientBoosting) con lam te di (0.7089 so voi 0.7152 cua hoi quy tuyen tinh),
trong khi head tu hoc cach gop lat dat 0.7698. Nen buoc tiep la doi backbone to hon hoac
do phan giai cao hon.

Nhung DOI CAI NAO thi phu thuoc vao gia, va gia do KHONG suy ra duoc tu may CPU. Do o may:

    S/14 @ 224   1.0x        B/14 @ 224   5.0x
    S/14 @ 392   6.6x        B/14 @ 336  11.0x

Tren CPU, ViT-S nho nen bi chan boi bang thong bo nho va khong dat duoc thong luong toi da;
tren GPU voi batch 64 thi ViT-B lap day tot hon han, nen ti le THAT co the thap hon nhieu.
Lay ti le CPU ma lap ke hoach la cach chac chan nhat de dot 12 tieng GPU vo ich.

Script nay do THANG tren GPU se dung, bang chinh anh that, va quy ra tong thoi gian cho
437.000 lat.
"""
from __future__ import annotations

# ============================== CONFIG ==============================
N_STUDY_THU = 12           # so ca that dung de do (du de trung binh hoa, du nhanh)
BATCH = 64
TONG_LAT = 436_950         # tong so lat cua 3 mat phang, de quy ra tong thoi gian
GIOI_HAN_PHIEN = 12        # gio, gioi han mot phien Kaggle
CAU_HINH = [
    ("dinov2_vits14", 224),   # dang dung, lam moc
    ("dinov2_vits14", 392),   # chi tang do phan giai
    ("dinov2_vitb14", 224),   # chi tang nang luc
    ("dinov2_vitb14", 336),   # tang ca hai
]
# ====================================================================

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

COMPETITION = "rsna-knee-abnormality-detection"
PLANES = ["Sagittal", "Axial", "Coronal"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find_root() -> Path:
    base = Path("/kaggle/input")
    for cand in (base / "competitions" / COMPETITION, base / COMPETITION):
        if (cand / "train_series.csv").exists():
            return cand
    for pattern in ("*/train_series.csv", "*/*/train_series.csv"):
        for hit in base.glob(pattern):
            return hit.parent
    raise SystemExit(f"Khong thay train_series.csv trong {base}")


def normalize_series(arr: np.ndarray) -> np.ndarray:
    p_low, p_high = (np.float32(v) for v in np.percentile(arr, (1, 99)))
    out = np.clip(arr, p_low, p_high).astype(np.float32)
    return (out - p_low) / (p_high - p_low + np.float32(1e-8))


def load_series(series_dir: Path) -> np.ndarray:
    items = []
    for p in sorted(series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(p))
            items.append((int(getattr(ds, "InstanceNumber", -1)), ds.pixel_array))
        except Exception:
            continue
    if not items:
        raise FileNotFoundError(series_dir)
    items.sort(key=lambda t: t[0])
    shapes = {a.shape for _, a in items}
    if len(shapes) > 1:
        pho_bien = max(shapes, key=lambda s: sum(1 for _, a in items if a.shape == s))
        items = [(i, a) for i, a in items if a.shape == pho_bien]
    return np.stack([a for _, a in items])


def resize_volume(arr: np.ndarray, size: int) -> np.ndarray:
    out = np.empty((arr.shape[0], size, size), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.asarray(Image.fromarray(arr[i]).resize((size, size), Image.BILINEAR),
                            dtype=np.float32)
    return out


@torch.no_grad()
def do_mot_cau_hinh(ten: str, size: int, vols: list, device: str) -> dict:
    model = torch.hub.load("facebookresearch/dinov2", ten, verbose=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n_param = sum(p.numel() for p in model.parameters())

    # Lam nong: lan chay dau luon cham hon (cap phat bo nho, chon thuat toan cuDNN).
    # Tinh ca no vao se lam cau hinh nao chay truoc bi thiet.
    hot = torch.randn(BATCH, 3, size, size, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        model(hot)
    if device == "cuda":
        torch.cuda.synchronize()

    n_lat, dim = 0, None
    t0 = time.time()
    for vol in vols:
        v = resize_volume(vol, size)
        x = torch.from_numpy(v).float().unsqueeze(1).repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        for i in range(0, len(x), BATCH):
            chunk = x[i:i + BATCH].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                out = model(chunk)
            dim = out.shape[1]
        n_lat += len(x)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"backbone": ten, "size": size, "patch": (size // 14) ** 2,
            "tham_so_M": round(n_param / 1e6), "dim": dim,
            "n_lat": n_lat, "giay": round(dt, 1),
            "lat_moi_giay": round(n_lat / dt, 1),
            "gio_cho_437k": round(TONG_LAT / (n_lat / dt) / 3600, 1)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CANH BAO: khong thay GPU -> so do duoc VO NGHIA cho viec lap ke hoach.")
        print("  Settings -> Accelerator = GPU roi chay lai.")
    print(f"device {device}"
          + (f" | {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))

    root = find_root()
    series = pd.read_csv(root / "train_series.csv",
                         dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    phan = [g.groupby("StudyInstanceUID", as_index=False).first()
            for pl in PLANES for g in [series[series["Anatomical_Plane"] == pl]] if len(g)]
    got = pd.concat(phan, ignore_index=True)
    theo_ca = {u: g for u, g in got.groupby("StudyInstanceUID")}

    # DOC ANH MOT LAN roi dung lai cho moi cau hinh. Neu doc lai moi lan thi phep do se
    # bao gom ca thoi gian doc DICOM - von chiem 86% va giong nhau o moi cau hinh - lam
    # moi ti le bi keo ve gan 1.0 va bang so tro nen vo dung.
    print(f"\ndang doc {N_STUDY_THU} ca (doc MOT lan, dung lai cho moi cau hinh) ...",
          flush=True)
    # Giu TUNG SERIES rieng, khong noi lai. Anh goc cua cac mat phang co kich thuoc khac
    # nhau (512 vs 560...), nen noi truoc khi resize se nem ValueError. Duong trich that
    # resize ve SIZE roi moi noi - nhung o day SIZE khac nhau theo tung cau hinh dang do,
    # nen phai giu anh goc va resize ben trong vong do.
    vols, t0, n_ca = [], time.time(), 0
    for uid in list(theo_ca)[:N_STUDY_THU]:
        co = False
        for _, r in theo_ca[uid].iterrows():
            sdir = root / "train_series" / uid / r["SeriesInstanceUID"]
            try:
                vols.append(normalize_series(load_series(sdir)))
                co = True
            except Exception:
                continue
        n_ca += int(co)
    n_lat = sum(len(v) for v in vols)
    print(f"  {n_ca} ca | {len(vols)} series | {n_lat:,} lat | "
          f"{time.time()-t0:.0f}s doc anh")
    if not vols:
        raise SystemExit("Khong doc duoc ca nao")

    rows = []
    for ten, size in CAU_HINH:
        print(f"\ndang do {ten} @ {size} ...", flush=True)
        try:
            r = do_mot_cau_hinh(ten, size, vols, device)
        except RuntimeError as exc:
            # Het VRAM la ket qua HUU ICH, khong phai that bai: no loai cau hinh do khoi
            # danh sach chon, hoac bao rang phai giam BATCH.
            print(f"  LOI: {type(exc).__name__}: {str(exc)[:120]}")
            rows.append({"backbone": ten, "size": size, "patch": (size // 14) ** 2,
                         "loi": type(exc).__name__})
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(f"  {r['lat_moi_giay']} lat/giay -> {r['gio_cho_437k']}h cho 437k lat")

    print("\n=== KET QUA ===")
    print(f"{'backbone':<16} {'size':>5} {'patch':>6} {'tham so':>8} {'dim':>5} "
          f"{'lat/giay':>9} {'gio/437k':>9} {'vua 1 phien?':>13}")
    goc = None
    for r in rows:
        if "loi" in r:
            print(f"{r['backbone']:<16} {r['size']:>5} {r['patch']:>6}   {r['loi']}")
            continue
        if goc is None:
            goc = r["gio_cho_437k"]
        vua = "OK" if r["gio_cho_437k"] <= GIOI_HAN_PHIEN else "QUA DAI"
        print(f"{r['backbone']:<16} {r['size']:>5} {r['patch']:>6} "
              f"{r['tham_so_M']:>7}M {r['dim']:>5} {r['lat_moi_giay']:>9.1f} "
              f"{r['gio_cho_437k']:>8.1f}h {vua:>13}")

    print("\nGhi chu de doc cho dung:")
    print("  - Con so tren CHI la thoi gian model chay. Doc DICOM chiem them ~86% nua,")
    print("    nhung phan do GIONG NHAU o moi cau hinh nen khong anh huong viec CHON.")
    print(f"  - Neu cau hinh muon dung vuot {GIOI_HAN_PHIEN}h: lay thua lat (moi lat thu 2)")
    print("    la giam mot nua, hoac tach lam hai phien theo mat phang.")

    out = Path("/kaggle/working") / "toc_do_backbone.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nDa ghi {out}")


if __name__ == "__main__":
    main()
