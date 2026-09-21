"""Test chong ro ri khi chia tap.

Ro ri chi duoc coi la da chan khi co MOT CON SO DEM DUOC bang 0, khong phai khi da dung
dung ten ham. Cac test duoi la phien ban tu dong cua chinh phep dem do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import LABELS, STUDY_COL
from rsna_knee.splits import (
    assert_no_leakage,
    assign_folds,
    build_groups,
    measure_series_split_leakage,
    normalize_report,
    report_group_key,
)


def _gold_gia_lap(n=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({STUDY_COL: [f"study_{i:03d}" for i in range(n)]})
    for lab in LABELS:
        df[lab] = rng.integers(0, 2, n).astype(float)
    df["MCL"] = ([1] * 6) + ([0] * (n - 6))    # nhan hiem, du de stratify n_splits=5
    df["Report"] = [f"bao cao so {i}" for i in range(n)]
    return df


def test_chia_dung_thi_khong_ca_nao_nam_hai_ben():
    folded = assign_folds(_gold_gia_lap(), n_splits=5, seed=42)
    assert set(folded["fold"]) == {0, 1, 2, 3, 4}
    assert assert_no_leakage(folded) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}


def test_phat_hien_duoc_ro_ri_khi_co_that():
    """Neu assert_no_leakage khong bat duoc ro ri co that thi no vo dung."""
    df = _gold_gia_lap(20)
    df["fold"] = [0] * 10 + [1] * 10
    df.loc[0, STUDY_COL] = df.loc[15, STUDY_COL]     # co tinh cho mot ca nam ca hai fold
    with pytest.raises(AssertionError):
        assert_no_leakage(df, use_report_dedup=False)


def test_chia_cap_series_lam_ro_ri_gan_toan_bo():
    """Bang chung cot loi cua ngay 4, o dang test tu dong.

    Moi ca co nhieu chuoi anh -> chia ngau nhien cap chuoi thi hau het ca deu lot ca hai ben.
    """
    series = pd.DataFrame({
        STUDY_COL: np.repeat([f"study_{i}" for i in range(30)], 6),
        "SeriesInstanceUID": [f"series_{i}" for i in range(180)],
    })
    r = measure_series_split_leakage(series, seed=42, test_size=0.2)
    assert r["leak_rate"] > 0.9, f"phai ro ri gan het, do duoc {r['leak_rate']:.1%}"


def test_bao_cao_trung_bi_gop_lam_mot_nhom():
    """Hai ca khac UID nhung cung mot bao cao khong duoc nam hai ben khi cham bo rut nhan."""
    df = _gold_gia_lap(20)
    df.loc[1, "Report"] = df.loc[0, "Report"]        # trung y nguyen
    df.loc[2, "Report"] = "  BAO CAO SO 0   "        # trung sau khi chuan hoa

    groups = build_groups(df, use_report_dedup=True)
    assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2]
    assert groups.nunique() == 18, "20 ca, 3 ca chung bao cao -> 18 nhom"

    folded = assign_folds(df, n_splits=5, seed=42)
    assert_no_leakage(folded)
    assert len({folded.loc[i, "fold"] for i in (0, 1, 2)}) == 1


def test_tat_gop_bao_cao_thi_moi_ca_la_mot_nhom():
    df = _gold_gia_lap(20)
    df.loc[1, "Report"] = df.loc[0, "Report"]
    assert build_groups(df, use_report_dedup=False).nunique() == 20


def test_chuan_hoa_bao_cao():
    assert normalize_report("  Khong  CO\nrach   ") == "khong co rach"
    assert report_group_key("A  b") == report_group_key("\ta B\n")
    assert report_group_key("a b") != report_group_key("a c")


def test_chia_lai_cung_seed_ra_cung_ket_qua():
    """Split phai tai lap duoc, neu khong moi so do giua cac ngay deu khong so sanh duoc."""
    df = _gold_gia_lap()
    a = assign_folds(df, seed=42)["fold"].tolist()
    b = assign_folds(df, seed=42)["fold"].tolist()
    c = assign_folds(df, seed=7)["fold"].tolist()
    assert a == b
    assert a != c, "doi seed phai ra split khac"


def test_assign_folds_khong_sua_dataframe_goc():
    df = _gold_gia_lap()
    assign_folds(df)
    assert "fold" not in df.columns
