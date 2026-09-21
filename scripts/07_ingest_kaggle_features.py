"""Nhan ket qua tu Kaggle notebook vao repo va kiem tra truoc khi dung.

Chay sau khi tai ba file tu /kaggle/working ve mot thu muc bat ky:

    python scripts/07_ingest_kaggle_features.py ~/Downloads
    python scripts/07_ingest_kaggle_features.py "C:/Users/Admin/Downloads" --plane sagittal

Viec no lam:
  1. Tim `features_dinov2_<plane>.npz`, `dicom_headers.csv`, `extract_summary.json`.
  2. Kiem dac trung: dung so chieu, khong co nan/inf, khop UID trong train.csv.
  3. Doi chieu voi dac trung da trich o may (neu co) - phai trung khop tung bit.
  4. Chuan hoa ten hang may chup, ghi cot `manufacturer_norm` vao manifest header.
  5. Chep vao dung cho, in ra viec can chay tiep.

Buoc 3 la buoc dang gia nhat: no chung minh dac trung tren Kaggle va o may sinh ra tu
cung mot phep tinh. Neu lech, moi so do sau deu khong so sanh duoc voi so do truoc.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rsna_knee.config import DATA_INTERIM, DATA_MANIFEST, LABELS, STUDY_COL
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw

# Bay da gap that o ngay 3: bay chuoi `Manufacturer` nhung chi la bon hang.
# Nhom theo chuoi tho se tao nhom gia, va split "theo may chup" van de hai ca cung hang
# nam hai ben. Phai chuan hoa truoc khi dung lam khoa nhom.
VENDOR_RULES = (
    ("siemens", "SIEMENS"),
    ("philips", "PHILIPS"),
    ("ge medical", "GE"),
    ("ge health", "GE"),
    ("gehc", "GE"),                 # viet tat cua GE HealthCare
    # Toshiba Medical -> Canon Medical (2016), Hitachi Medical -> Fujifilm Healthcare
    # (2021). Cung mot dong may duoc doi ten, nen dau van tay thiet bi la mot. Muc dich
    # o day la nhom theo PHAN CUNG chu khong theo phap nhan, nen gop lai.
    ("toshiba", "CANON"),
    ("canon", "CANON"),
    ("hitachi", "FUJIFILM"),
    ("fujifilm", "FUJIFILM"),
    ("united imaging", "UIH"),
)


def normalize_vendor(raw: object) -> str:
    text = str(raw).strip().lower()
    if text in ("", "nan", "none"):
        return "UNKNOWN"
    for needle, name in VENDOR_RULES:
        if needle in text:
            return name
    return text.upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", help="thu muc chua ba file tai tu Kaggle ve")
    ap.add_argument("--plane", default="sagittal")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser()
    npz = src / f"features_dinov2_{args.plane}.npz"
    heads = src / "dicom_headers.csv"
    summary = src / "extract_summary.json"

    missing = [p.name for p in (npz, heads) if not p.exists()]
    if missing:
        raise SystemExit(f"Khong thay {missing} trong {src}\n"
                         f"Co trong thu muc do: {sorted(p.name for p in src.glob('*'))[:20]}")

    if summary.exists():
        print("=== Tom tat tu Kaggle ===")
        for k, v in json.loads(summary.read_text()).items():
            if k != "failed":
                print(f"  {k:<12} {v}")

    print("\n=== 1. Kiem dac trung ===")
    with np.load(npz) as z:
        feats = {k: z[k] for k in z.files}
    dims = {f.shape[1] for f in feats.values()}
    n_slices = sum(f.shape[0] for f in feats.values())
    print(f"  {len(feats):,} study | {n_slices:,} lat | so chieu {dims}")
    if len(dims) != 1:
        raise SystemExit(f"Dac trung co nhieu so chieu khac nhau: {dims}")

    bad = [k for k, v in feats.items() if not np.isfinite(v).all()]
    print(f"  study co nan/inf: {len(bad)}" + (f" -> {bad[:5]}" if bad else ""))
    if bad:
        raise SystemExit("Co dac trung khong huu han - khong dung duoc, chay lai tren Kaggle")

    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold = gold_subset(man)
    known = set(man[STUDY_COL])
    lac = set(feats) - known
    print(f"  UID khop train.csv: {len(set(feats) & known):,}/{len(feats):,}"
          + (f"  | LAC: {len(lac)}" if lac else ""))
    have_gold = set(feats) & set(gold[STUDY_COL])
    print(f"  study co nhan nguoi gan: {len(have_gold)}/58"
          + ("  <- DU" if len(have_gold) == 58 else "  <- CON THIEU"))

    print("\n=== 2. Doi chieu voi dac trung da trich o may ===")
    old_path = DATA_INTERIM / f"features_dinov2_vits14_{args.size}.npz"
    if old_path.exists():
        with np.load(old_path) as z:
            old = {k: z[k] for k in z.files}
        shared = sorted(set(old) & set(feats))
        if shared:
            # Lech tuyet doi mot minh KHONG doc duoc: dac trung DINOv2 co do lon rat khac
            # nhau giua cac chieu. Thuoc do dung la do tuong dong cosin - no tra loi dung
            # cau hoi can hoi: "hai ben co phai CUNG MOT bieu dien khong?"
            cos, rel, lech_shape = [], [], 0
            for uid in shared[:50]:
                a, b = old[uid], feats[uid]
                if a.shape != b.shape:
                    lech_shape += 1
                    continue
                af, bf = a.ravel(), b.ravel()
                cos.append(float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)))
                rel.append(float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12)))

            if lech_shape:
                print(f"  {lech_shape} study lech so lat - KHAC tien xu ly, phai xem lai")
            if cos:
                print(f"  doi chieu {len(cos)} study chung:")
                print(f"    tuong dong cosin : {min(cos):.6f} - {max(cos):.6f}")
                print(f"    lech tuong doi   : {max(rel):.2%} so voi gia tri lon nhat")
                if min(cos) > 0.999:
                    print("    -> CUNG mot bieu dien. Chenh lech den tu GPU fp16 (Kaggle)")
                    print("       vs CPU fp32 (may) - khong anh huong ket qua.")
                else:
                    print("    -> KHAC bieu dien. Kiem lai tien xu ly hai ben co giong nhau")
                    print("       khong (chuan hoa, resize, thu tu lat).")
        else:
            print("  khong co study chung de doi chieu")
    else:
        print(f"  chua co {old_path.name} o may - bo qua buoc doi chieu")

    print("\n=== 3. Chuan hoa ten hang may chup ===")
    hdf = pd.read_csv(heads)
    hdf["manufacturer_norm"] = hdf["Manufacturer"].map(normalize_vendor)
    raw_n = hdf["Manufacturer"].nunique(dropna=False)
    norm_n = hdf["manufacturer_norm"].nunique()
    print(f"  chuoi Manufacturer tho : {raw_n}")
    print(f"  sau khi chuan hoa      : {norm_n}")
    print(hdf["manufacturer_norm"].value_counts().to_string())
    per_study = (hdf.groupby("StudyInstanceUID")["manufacturer_norm"]
                 .agg(lambda s: s.mode().iat[0]).reset_index())
    print(f"  study co hang may chup : {len(per_study):,}")

    print("\n=== 4. Ghi vao repo ===")
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST.mkdir(parents=True, exist_ok=True)
    dst_npz = DATA_INTERIM / f"features_dinov2_vits14_{args.size}.npz"
    if dst_npz.exists():
        backup = dst_npz.with_suffix(".npz.bak")
        shutil.move(str(dst_npz), str(backup))
        print(f"  ban cu doi ten thanh {backup.name}")
    shutil.copy2(npz, dst_npz)
    hdf.to_csv(DATA_MANIFEST / "dicom_headers.csv", index=False)
    per_study.to_csv(DATA_MANIFEST / "study_manufacturer.csv", index=False)
    print(f"  {dst_npz}  ({dst_npz.stat().st_size/1024**2:,.0f} MB)")
    print(f"  {DATA_MANIFEST / 'dicom_headers.csv'}")
    print(f"  {DATA_MANIFEST / 'study_manufacturer.csv'}")

    print("\n=== Chay tiep ===")
    print("  python scripts/31_train_frozen_head.py --backbone dinov2_vits14 --epochs 80")
    print("  python scripts/40_experiment_weak_labels.py --backbone dinov2_vits14 --epochs 80")


if __name__ == "__main__":
    main()
