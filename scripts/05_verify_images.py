"""Kiem tra toan ven anh da tai truoc khi train.

Vi sao can mot buoc rieng: mot file .dcm tai do dang (0 byte, hoac cut giua chung) khong
lam hong buoc tai - no chi no ra o giua epoch dau tien, voi mot thong bao khong he nhac
den chuyen tai thieu. Kiem trong 10 giay o day re hon nhieu.

Chay:
    python scripts/05_verify_images.py
    python scripts/05_verify_images.py --delete-bad   # xoa file hong de tai lai
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rsna_knee.config import DATA_MANIFEST, DATA_RAW

IMAGES = DATA_RAW / "images"
LOCAL_CSV = DATA_MANIFEST / "local_images.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-bad", action="store_true")
    args = ap.parse_args()

    import pydicom

    files = sorted(IMAGES.rglob("*.dcm"))
    print(f"Kiem {len(files):,} file trong {IMAGES}")

    bad_size, bad_parse = [], []
    if LOCAL_CSV.exists():
        expected = pd.read_csv(LOCAL_CSV).set_index("local_path")["size_bytes"].to_dict()
        for p in files:
            want = expected.get(str(p))
            if want is not None and p.stat().st_size != want:
                bad_size.append((p, p.stat().st_size, want))

    for p in files:
        try:
            pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception as exc:
            bad_parse.append((p, type(exc).__name__))

    print(f"  sai kich thuoc : {len(bad_size)}")
    print(f"  khong doc duoc : {len(bad_parse)}")
    for p, kind in bad_parse[:10]:
        print(f"    {kind:<20} {p.stat().st_size:>9,} byte  ...{p.name[-24:]}")

    missing = 0
    if LOCAL_CSV.exists():
        wanted = pd.read_csv(LOCAL_CSV)["local_path"]
        missing = int((~wanted.map(lambda s: Path(s).exists())).sum())
        print(f"  con thieu      : {missing} / {len(wanted)} file trong manifest")

    if args.delete_bad:
        for p, _ in bad_parse:
            p.unlink()
        for p, _, _ in bad_size:
            p.unlink(missing_ok=True)
        print(f"  da xoa {len(bad_parse) + len(bad_size)} file - chay lai "
              f"scripts/04_download_from_listing.py de tai lai")

    ok = not bad_parse and not bad_size and missing == 0
    print(f"\n{'DAT' if ok else 'CHUA DAT'} - {'san sang train' if ok else 'xu ly cac muc tren truoc'}")


if __name__ == "__main__":
    main()
