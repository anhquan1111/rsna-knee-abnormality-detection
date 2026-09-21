"""Chia fold chong ro ri, don vi la study.

Ba tang ro ri cua bai nay:
  1. slice -> study: hai lat cua cung mot ca nam o hai tap. Chan bang cach chia o cap study.
  2. series -> study: mot ca co ~5.5 series; chia ngau nhien cap series lam ro ri gan nhu
     toan bo tap val (do thuc te: 49/49 study).
  3. report trung lap: 177 dong co `Report` trung y nguyen, 204 dong neu bo qua hoa/thuong.
     Hai study khac UID nhung cung mot bao cao ma nam hai ben thi bo rut nhan yeu duoc cham
     tren chinh van ban no da thay.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import N_FOLDS, SEED, STRATIFY_LABEL, STUDY_COL


def normalize_report(text: str) -> str:
    """Chuan hoa de phat hien bao cao trung: bo hoa/thuong va gop khoang trang."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def report_group_key(text: str) -> str:
    return hashlib.md5(normalize_report(text).encode("utf-8")).hexdigest()[:12]


def build_groups(df: pd.DataFrame, use_report_dedup: bool = True) -> pd.Series:
    """Khoa nhom cho split.

    Mac dinh la StudyInstanceUID. Neu bat `use_report_dedup`, cac study co bao cao trung
    nhau bi gop lam MOT nhom - dieu kien bat buoc khi danh gia bo rut nhan tu van ban.
    """
    groups = df[STUDY_COL].astype(str).copy()
    if use_report_dedup and "Report" in df.columns:
        keys = df["Report"].map(report_group_key)
        dup_keys = set(keys[keys.duplicated(keep=False)])
        merged = keys.where(keys.isin(dup_keys), groups)
        groups = merged
    return groups


def assign_folds(
    df: pd.DataFrame,
    stratify_label: str = STRATIFY_LABEL,
    n_splits: int = N_FOLDS,
    seed: int = SEED,
    use_report_dedup: bool = True,
) -> pd.DataFrame:
    """Gan cot `fold` cho tung dong. Tra ve ban sao, khong sua df goc.

    `StratifiedGroupKFold` chi nhan y mot chieu -> phai chon dung MOT nhan de can bang.
    Chon nhan hiem nhat (MCL) vi no la nhan de vo nhat: neu fold nao khong co duong tinh
    MCL thi macro AUC cua fold do mat han mot nhan.
    """
    out = df.copy().reset_index(drop=True)
    groups = build_groups(out, use_report_dedup=use_report_dedup)
    y = out[stratify_label].fillna(-1).astype(int)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out["fold"] = -1
    for fold, (_, va) in enumerate(sgkf.split(out, y=y, groups=groups)):
        out.loc[va, "fold"] = fold
    if (out["fold"] < 0).any():
        raise RuntimeError("Con dong chua duoc gan fold")
    return out


def assert_no_leakage(df: pd.DataFrame, use_report_dedup: bool = True) -> dict:
    """Kiem lai sau khi chia - lam moi lan, khong tin suong vao ten ham sklearn."""
    groups = build_groups(df, use_report_dedup=use_report_dedup)
    stats = {}
    for fold in sorted(df["fold"].unique()):
        va = set(groups[df["fold"] == fold])
        tr = set(groups[df["fold"] != fold])
        overlap = va & tr
        stats[int(fold)] = len(overlap)
        if overlap:
            raise AssertionError(f"fold {fold}: {len(overlap)} nhom nam ca hai ben")
    return stats


def measure_series_split_leakage(series_df: pd.DataFrame, seed: int = SEED, test_size: float = 0.2) -> dict:
    """Do truc tiep hau qua cua viec chia ngau nhien o CAP SERIES thay vi cap study.

    Day la con so dung de thuyet phuc: khong phai canh bao ly thuyet ma la ti le study bi
    ro ri that tren chinh du lieu nay.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(series_df))
    n_val = int(len(series_df) * test_size)
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    va_studies = set(series_df.iloc[va_idx][STUDY_COL])
    tr_studies = set(series_df.iloc[tr_idx][STUDY_COL])
    leaked = va_studies & tr_studies
    return {
        "n_series_val": int(n_val),
        "n_study_val": len(va_studies),
        "n_study_leaked": len(leaked),
        "leak_rate": len(leaked) / max(len(va_studies), 1),
    }
