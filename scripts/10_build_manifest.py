"""Ngay 3-4: dung manifest cap study tu CSV goc va in cac kiem tra toan ven.

Chay:  python scripts/10_build_manifest.py
Ra:    data/interim/study_manifest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rsna_knee.config import DATA_INTERIM, LABELS, SERIES_COL, STUDY_COL
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw


def main() -> None:
    train, series = load_raw()
    print(f"train.csv        {train.shape[0]:>7,} dong x {train.shape[1]} cot")
    print(f"train_series.csv {series.shape[0]:>7,} dong x {series.shape[1]} cot")

    man = build_study_manifest(train, series)
    gold = gold_subset(man)

    print("\n--- Toan ven khoa ---")
    print(f"study duy nhat trong train.csv   : {train[STUDY_COL].is_unique}")
    print(f"series duy nhat trong series.csv : {series[SERIES_COL].is_unique}")
    print(f"study cua series co trong train  : {series[STUDY_COL].isin(train[STUDY_COL]).all()}")
    print(f"study khong co series nao        : {int(man['n_series'].isna().sum())}")

    print("\n--- Nhan ---")
    print(f"study co nhan nguoi gan : {len(gold)} / {len(man)} ({len(gold)/len(man):.2%})")
    part = train[list(LABELS)].notna().sum(axis=1)
    print(f"study gan nhan do dang  : {int(((part > 0) & (part < 12)).sum())}")
    print(f"tong nhan duong tinh    : {int(gold[list(LABELS)].sum().sum())} "
          f"({gold[list(LABELS)].sum().sum()/len(gold):.2f} nhan/study)")
    print(f"study khong co nhan duong tinh nao: {int((gold['n_positive'] == 0).sum())}")

    print("\n--- Duong tinh tung nhan (tren 58 study) ---")
    prev = gold[list(LABELS)].sum().astype(int).sort_values(ascending=False)
    for name, n in prev.items():
        print(f"  {name:<18} {n:>3} / 58  ({n/58:.1%})")

    print("\n--- Series moi study ---")
    print(man["n_series"].describe()[["min", "25%", "50%", "75%", "max"]].to_string())
    print(f"study co du ca 3 mat phang: "
          f"{int(((man.n_sagittal > 0) & (man.n_coronal > 0) & (man.n_axial > 0)).sum())} / {len(man)}")

    print("\n--- Hai co ngo doc lap? ---")
    same = (series["Fluid_Sensitive"] == series["Fat_Suppression"]).all()
    print(f"Fluid_Sensitive == Fat_Suppression tren ca {len(series):,} series: {same}")
    if same:
        print("  -> hai cot trung nhau hoan toan, khong duoc coi la hai feature doc lap")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out = DATA_INTERIM / "study_manifest.csv"
    man.to_csv(out, index=False)
    print(f"\nDa ghi {out}  ({len(man):,} dong)")


if __name__ == "__main__":
    main()
