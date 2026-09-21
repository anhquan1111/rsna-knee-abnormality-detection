"""CHAY TREN KAGGLE NOTEBOOK - khong chay o may local.

Vi sao can file nay: tai anh qua Kaggle API bi gioi han toc do (HTTP 429). Tren Kaggle
notebook, dataset da duoc mount san o /kaggle/input nen KHONG co request API nao - doc
file truc tiep tu dia. Them nua, GPU T4 mien phi nhanh hon CPU may nha nhieu lan.

Dau ra chi vai tram MB thay vi 569 GB anh goc:
    /kaggle/working/features_dinov2_<plane>.npz   - dac trung tung lat, moi study mot mang
    /kaggle/working/dicom_headers.csv             - metadata de chan ro ri theo may chup
    /kaggle/working/extract_summary.json          - so lieu de doi chieu

CACH DUNG
---------
1. Mo https://www.kaggle.com/competitions/rsna-knee-abnormality-detection -> New Notebook.
2. Settings: Accelerator = GPU T4 x2, Internet = ON (de tai trong so DINOv2).
3. Dan toan bo file nay vao MOT cell, sua CONFIG o duoi neu can, roi Run All.
4. Xong thi tai ba file o /kaggle/working ve, dat vao:
       data/interim/features_dinov2_vits14_224.npz
       data/manifest/dicom_headers.csv
5. Chay tiep o may: scripts/31_train_frozen_head.py va scripts/40_experiment_weak_labels.py

GHI CHU: Kaggle gioi han 12 tieng moi phien va 20 GB o /kaggle/working. Trich dac trung cho
ca 4,407 study uoc tinh ~700 MB, nam thoai mai trong gioi han.
"""
from __future__ import annotations

# ============================== CONFIG ==============================
PLANE = "Sagittal"        # Sagittal | Coronal | Axial
SIZE = 224                # boi cua 14 cho DINOv2 patch-14
BATCH = 64                # so lat moi lo dua vao backbone
ONLY_LABELLED = False     # True = chi 58 study co nhan nguoi gan (chay ~2 phut de thu)
MAX_STUDIES = 0           # 0 = khong gioi han. CHAY THU LAN DAU: dat 60 (~3 phut)
                          # de chac chan moi thu chay duoc, roi doi ve 0 va Run All lai.
# ====================================================================

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

COMPETITION = "rsna-knee-abnormality-detection"


def find_root() -> Path:
    """Tim thu muc chua train_series.csv trong /kaggle/input.

    Kaggle dat du lieu o CHO KHAC NHAU tuy cach them input:
        /kaggle/input/competitions/<slug>/   <- khi mo notebook tu trang competition
        /kaggle/input/<slug>/                <- khi Add Input thu cong
    Hard-code mot duong dan la hong ngay dong doc CSV dau tien, voi FileNotFoundError
    khong he nhac gi toi chuyen duong dan sai cho.

    Chi thu mot so duong dan ung vien roi glob nong 2 cap - KHONG rglob ca cay, vi
    /kaggle/input co 819,640 file va duyet het se treo vai phut.
    """
    base = Path("/kaggle/input")
    for cand in (base / "competitions" / COMPETITION, base / COMPETITION):
        if (cand / "train_series.csv").exists():
            return cand
    for pattern in ("*/train_series.csv", "*/*/train_series.csv"):
        for hit in base.glob(pattern):
            return hit.parent
    found = sorted(p.name for p in base.glob("*"))[:20]
    raise SystemExit(
        f"Khong thay train_series.csv trong {base}.\n"
        f"Co trong {base}: {found}\n"
        "Kiem lai panel Input ben phai da co 'RSNA Knee Abnormality Detection' chua."
    )


ROOT = find_root()
OUT = Path("/kaggle/working")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
META_TAGS = ("StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber", "Manufacturer",
             "ManufacturerModelName", "MagneticFieldStrength", "Rows", "Columns",
             "PixelRepresentation", "SliceThickness")


# ---------------------------------------------------------------- tien xu ly
def normalize_series(arr: np.ndarray) -> np.ndarray:
    """Percentile 1-99 tren CA series. Giong het ban o src/rsna_knee/dicom_io.py.

    Chuan hoa theo series chu khong theo tung lat: cuong do MRI khong tuyet doi, va ep
    tung lat ve cung mot dai se xoa mat do sang tuong doi giua cac lat.
    `+ 1e-8` chan chia cho 0 khi series toan mot gia tri.
    """
    # `np.percentile` tra ve float64; tru vao mang float32 se nang ca ket qua len float64
    # (cache to gap doi ma ham van chay dung). Ep he so ve float32 truoc.
    p_low, p_high = (np.float32(v) for v in np.percentile(arr, (1, 99)))
    out = np.clip(arr, p_low, p_high).astype(np.float32)
    return (out - p_low) / (p_high - p_low + np.float32(1e-8))


def load_series(series_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Doc ca series, sap theo InstanceNumber. Tra ve (N,H,W) va metadata tung lat.

    Sap xep la bat buoc: ten file la UID ngau nhien nen sorted(glob) cho ra mot chong lat
    bi xao tron, khong phai thu tu giai phau.
    """
    items = []
    for p in sorted(series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(p))
        except Exception as exc:
            print(f"    bo file hong {p.name[-16:]}: {type(exc).__name__}")
            continue
        meta = {t: getattr(ds, t, None) for t in META_TAGS}
        meta["transfer_syntax"] = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
        items.append((int(getattr(ds, "InstanceNumber", -1)), ds, meta))
    if not items:
        raise FileNotFoundError(f"khong doc duoc lat nao trong {series_dir}")

    items.sort(key=lambda t: t[0])
    shapes = {(int(d.Rows), int(d.Columns)) for _, d, _ in items}
    if len(shapes) > 1:                      # hiem, nhung co that trong dataset nay
        raise ValueError(f"series co nhieu kich thuoc: {shapes}")
    return np.stack([d.pixel_array for _, d, _ in items]), [m for _, _, m in items]


def resize_volume(arr: np.ndarray, size: int) -> np.ndarray:
    """Chuan hoa TRUOC roi resize. Lam nguoc lai thi noi suy tao gia tri moi o bien,
    keo percentile lech di."""
    out = np.empty((arr.shape[0], size, size), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.asarray(Image.fromarray(arr[i]).resize((size, size), Image.BILINEAR),
                            dtype=np.float32)
    return out


# ---------------------------------------------------------------- backbone
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_backbone(device: str):
    """Tai DINOv2-S/14 va dong bang.

    `torch.hub.load` can Internet. Kaggle mac dinh TAT Internet va phai xac minh so dien
    thoai moi bat duoc - day la cho hay ket nhat khi chay lan dau, nen bao loi that ro.
    """
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    except Exception as exc:
        raise SystemExit(
            f"Khong tai duoc trong so DINOv2 ({type(exc).__name__}).\n"
            "Nguyen nhan hay gap nhat: Kaggle notebook dang TAT Internet.\n"
            "  Sua: panel ben phai -> Settings -> Internet = On "
            "(can xac minh so dien thoai).\n"
            "  Khong bat duoc Internet? Add Data -> tim dataset chua trong so dinov2,\n"
            "  roi doi build_backbone() sang torch.load tu duong dan /kaggle/input do."
        ) from exc

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)   # dong bang that su, khong chi dua vao no_grad
    return model


@torch.no_grad()
def extract(slices: np.ndarray, model, device: str, batch: int) -> np.ndarray:
    """(N,H,W) trong [0,1] -> (N,384). Lap 1 kenh thanh 3 de giu trong so pretrained."""
    x = torch.from_numpy(slices).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    outs = []
    for i in range(0, len(x), batch):
        chunk = x[i:i + batch].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            outs.append(model(chunk).float().cpu())
    return torch.cat(outs).numpy().astype(np.float32)


# ---------------------------------------------------------------- chay
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train = pd.read_csv(ROOT / "train.csv")
    series = pd.read_csv(ROOT / "train_series.csv")
    labelled = set(train[train[LABELS].notna().all(axis=1)]["StudyInstanceUID"])
    print(f"train.csv {train.shape} | train_series.csv {series.shape} "
          f"| study co nhan nguoi gan: {len(labelled)}")

    want = series[series["Anatomical_Plane"] == PLANE].copy()
    if ONLY_LABELLED:
        want = want[want["StudyInstanceUID"].isin(labelled)]
    # Moi study lay dung MOT series cua mat phang do
    want = want.groupby("StudyInstanceUID", as_index=False).first()
    # Study co nhan nguoi gan len dau: neu het gio thi phan quan trong nhat da xong
    want["is_gold"] = want["StudyInstanceUID"].isin(labelled)
    want = want.sort_values("is_gold", ascending=False).reset_index(drop=True)
    if MAX_STUDIES:
        want = want.head(MAX_STUDIES)
    print(f"se xu ly {len(want):,} study ({int(want.is_gold.sum())} co nhan nguoi gan), "
          f"mat phang {PLANE}")

    model = build_backbone(device)
    feats, headers, failed = {}, [], []
    t0 = time.time()

    for i, row in want.iterrows():
        sdir = ROOT / "train_series" / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
        try:
            vol, metas = load_series(sdir)
            arr = resize_volume(normalize_series(vol), SIZE)
            feats[row["StudyInstanceUID"]] = extract(arr, model, device, BATCH)
            headers.extend(metas)
        except Exception as exc:
            failed.append((row["StudyInstanceUID"], f"{type(exc).__name__}: {exc}"))
            continue

        if (i + 1) % 100 == 0 or i + 1 == len(want):
            done, el = i + 1, time.time() - t0
            print(f"  {done:>5}/{len(want)} study | {el/60:>5.1f} phut | "
                  f"con ~{el/done*(len(want)-done)/60:>5.1f} phut | hong {len(failed)}",
                  flush=True)
            # Ghi tam moi 500 study: Kaggle cat phien o 12 tieng, va study co nhan nguoi
            # gan da duoc xep len dau nen ban ghi tam dau tien da du dung cho ngay 5 va 8.
            if (i + 1) % 500 == 0:
                np.savez_compressed(OUT / f"features_dinov2_{PLANE.lower()}.npz", **feats)
                print(f"    (da ghi tam {len(feats)} study)", flush=True)

    np.savez_compressed(OUT / f"features_dinov2_{PLANE.lower()}.npz", **feats)
    pd.DataFrame(headers).to_csv(OUT / "dicom_headers.csv", index=False)

    n_slices = sum(f.shape[0] for f in feats.values())
    size_mb = (OUT / f"features_dinov2_{PLANE.lower()}.npz").stat().st_size / 1024**2
    summary = {
        "plane": PLANE, "size": SIZE, "n_studies": len(feats),
        "n_gold": len(set(feats) & labelled), "n_slices": n_slices,
        "dim": int(next(iter(feats.values())).shape[1]) if feats else 0,
        "npz_mb": round(size_mb, 1), "minutes": round((time.time() - t0) / 60, 1),
        "failed": failed[:50], "n_failed": len(failed),
    }
    (OUT / "extract_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== XONG ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed"}, indent=2))
    print(f"\nTai ve: features_dinov2_{PLANE.lower()}.npz ({size_mb:,.0f} MB), "
          f"dicom_headers.csv, extract_summary.json")
    if failed:
        print(f"\n{len(failed)} study loi, vi du dau tien:")
        for uid, err in failed[:5]:
            print(f"  {uid[-16:]}  {err}")


if __name__ == "__main__":
    main()
