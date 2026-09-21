"""Tai .dcm dua tren danh sach file lay tu Kaggle API (scripts/02_list_files_api.py).

Khac `03_download_subset.py`: script cu can `files_manifest.csv` sinh tu mot Kaggle notebook.
Script nay dung thang `data/manifest/competition_files.csv` do API tra ve, nen khong con
buoc thu cong nao.

Chay:
    python scripts/04_download_from_listing.py --gold --plane Sagittal --dry-run
    python scripts/04_download_from_listing.py --gold --plane Sagittal --max-studies 10

Tai lai duoc: file da co tren dia va dung kich thuoc thi bo qua.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rsna_knee.config import DATA_MANIFEST, DATA_RAW, SERIES_COL, STUDY_COL, short_uid
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw

COMPETITION = "rsna-knee-abnormality-detection"
FILES_CSV = DATA_MANIFEST / "competition_files.csv"
DEST = DATA_RAW / "images"


def select_files(args) -> pd.DataFrame:
    if not FILES_CSV.exists():
        raise SystemExit(f"Chua co {FILES_CSV}. Chay scripts/02_list_files_api.py truoc.")
    files = pd.read_csv(FILES_CSV)
    files = files[files["name"].str.startswith("train_series/")].copy()
    parts = files["name"].str.split("/")
    files[STUDY_COL] = parts.str[1]
    files[SERIES_COL] = parts.str[2]

    # Study cuoi cung trong danh sach co the moi duoc liet ke mot phan -> bo cho chac.
    files = files[files[STUDY_COL] != files[STUDY_COL].iloc[-1]]

    train, series = load_raw()
    if args.gold:
        wanted = set(gold_subset(build_study_manifest(train, series))[STUDY_COL])
        files = files[files[STUDY_COL].isin(wanted)]
    if args.plane:
        keep = set(series[series["Anatomical_Plane"] == args.plane][SERIES_COL])
        files = files[files[SERIES_COL].isin(keep)]

    # Moi study lay dung MOT series - tranh tai ca chuc series cua cung mot ca.
    first = files.groupby(STUDY_COL)[SERIES_COL].transform("first")
    files = files[files[SERIES_COL] == first]

    if args.max_studies:
        keep_studies = sorted(files[STUDY_COL].unique())[: args.max_studies]
        files = files[files[STUDY_COL].isin(keep_studies)]
    return files.sort_values("name").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true", help="chi tai study co nhan nguoi gan")
    ap.add_argument("--plane", default=None, choices=["Sagittal", "Coronal", "Axial"])
    ap.add_argument("--max-studies", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = select_files(args)
    total_mb = files["size_bytes"].sum() / 1024**2
    print(f"Chon {len(files):,} file | {files[STUDY_COL].nunique()} study "
          f"| {files[SERIES_COL].nunique()} series | {total_mb:,.1f} MB")
    if args.dry_run:
        print(files.groupby(STUDY_COL).agg(
            n=("name", "size"), mb=("size_bytes", lambda s: s.sum() / 1024**2)).to_string())
        return

    import kaggle

    DEST.mkdir(parents=True, exist_ok=True)
    # Duong dan cuc bo dung UID rut gon - xem `short_uid` ve ly do (MAX_PATH tren Windows).
    files["local_path"] = [
        str(DEST / short_uid(s) / short_uid(se) / Path(n).name)
        for s, se, n in zip(files[STUDY_COL], files[SERIES_COL], files["name"])
    ]
    # GOP vao manifest cu chu khong ghi de: du lieu duoc tai thanh nhieu dot (gold truoc,
    # study khac sau). Ghi de se lam scripts/05_verify_images.py bao thieu toan bo dot truoc.
    local_csv = DATA_MANIFEST / "local_images.csv"
    merged = files
    if local_csv.exists():
        merged = (pd.concat([pd.read_csv(local_csv), files], ignore_index=True)
                  .drop_duplicates(subset="name", keep="last"))
    merged.to_csv(local_csv, index=False)
    print(f"  manifest cuc bo: {len(merged):,} file (dot nay {len(files):,})")

    done = skipped = 0
    failed: list[str] = []
    t0 = time.time()
    for _, row in files.iterrows():
        out = Path(row["local_path"])
        if out.exists() and out.stat().st_size == row["size_bytes"]:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        # Bo qua mot file la tao ra mot series thieu lat - loi nay KHONG nem exception luc
        # train, chi lam ket qua te di mot cach kho hieu. Vi vay thu ky truoc khi bo.
        for attempt in range(8):
            try:
                kaggle.api.competition_download_file(
                    COMPETITION, row["name"], path=str(out.parent), force=True, quiet=True
                )
                break
            except Exception as exc:
                wait = min(2 ** attempt, 30)
                print(f"  loi {type(exc).__name__} o ...{row['name'][-32:]} "
                      f"(lan {attempt+1}/8) -> cho {wait}s", flush=True)
                time.sleep(wait)
        else:
            failed.append(row["name"])
            print(f"  BO QUA {row['name']}", flush=True)
            continue
        done += 1
        if done % 25 == 0:
            mb = sum(f.stat().st_size for f in DEST.rglob("*.dcm")) / 1024**2
            print(f"  {done:>5}/{len(files)} file | {mb:,.0f} MB | {time.time()-t0:,.0f}s")

    print(f"\nXong: tai moi {done}, bo qua {skipped} (da co), {time.time()-t0:,.0f}s")
    if failed:
        print(f"CHUA TAI DUOC {len(failed)} file - chay lai dung lenh nay de tai not")
    print(f"Thu muc: {DEST}")


if __name__ == "__main__":
    main()
