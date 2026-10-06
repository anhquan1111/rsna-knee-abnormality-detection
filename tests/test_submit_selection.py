"""Kiem doan chon series cua notebook nop bai.

Vi sao dang co test rieng: day la doan de hong mot cach AM THAM nhat trong ca repo.
Chon sai mat phang thi model van chay tron, van ghi ra submission.csv hop le, chi co
diem la thap - va khong co dong log nao noi vi sao. Da dinh dung loi nay mot lan: head
duoc train tren ba mat phang trong khi notebook van chi doc Sagittal.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "50_kaggle_submit.py"
PLANES = ["Sagittal", "Axial", "Coronal"]


def _load():
    # Ten file bat dau bang so nen khong import thang duoc.
    spec = importlib.util.spec_from_file_location("kaggle_submit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kaggle_submit"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def chon_series():
    return _load().chon_series


def _ts(rows):
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID",
                                       "Anatomical_Plane"])


def test_moi_mat_phang_dung_mot_series(chon_series):
    """Ca co hai series Sagittal chi duoc lay mot - neu khong, lat cua no bi dem hai lan."""
    ts = _ts([("S1", "a", "Sagittal"), ("S1", "b", "Sagittal"),
              ("S1", "c", "Axial"), ("S1", "d", "Coronal")])
    want, thieu = chon_series(ts, PLANES)

    assert len(want) == 3
    assert set(want["Anatomical_Plane"]) == set(PLANES)
    assert not want["SeriesInstanceUID"].duplicated().any()
    assert thieu == set()


def test_thieu_mat_phang_van_chay(chon_series):
    """Ca chi co Sagittal van phai co du doan, khong duoc rot ra ngoai."""
    ts = _ts([("S1", "a", "Sagittal"), ("S1", "b", "Axial"),
              ("S2", "c", "Sagittal")])
    want, thieu = chon_series(ts, PLANES)

    assert sorted(set(want["StudyInstanceUID"])) == ["S1", "S2"]
    assert len(want[want["StudyInstanceUID"] == "S2"]) == 1
    assert thieu == set()


def test_khong_co_mat_phang_nao_thi_rot_ve_series_bat_ky(chon_series):
    """Bo trang mot dong la mat diem dong do, khong phai mat diem mo hinh."""
    ts = _ts([("S1", "a", "Sagittal"), ("S2", "f", "Oblique"), ("S2", "g", "Oblique")])
    want, thieu = chon_series(ts, PLANES)

    assert thieu == {"S2"}
    s2 = want[want["StudyInstanceUID"] == "S2"]
    assert len(s2) == 1
    assert s2["Anatomical_Plane"].iat[0] == "Oblique"


def test_khong_sot_study_nao(chon_series):
    """Bat buoc: moi study trong test_series.csv deu phai co mat o dau ra."""
    ts = _ts([("S1", "a", "Sagittal"), ("S2", "b", "Axial"), ("S3", "c", "Coronal"),
              ("S4", "d", "Oblique"), ("S5", "e", "Sagittal"), ("S5", "f", "Axial")])
    want, _ = chon_series(ts, PLANES)

    assert set(want["StudyInstanceUID"]) == set(ts["StudyInstanceUID"])
    assert not want["SeriesInstanceUID"].duplicated().any()


def test_planes_khop_voi_luc_train(chon_series):
    """PLANES trong script phai la dung ba mat phang da dung luc trich dac trung.

    Neu ai do sua PLANES ve mot mat phang thi head van nap duoc va van chay - test nay
    la cho duy nhat bat duoc chuyen do truoc khi ton mot luot nop.
    """
    assert _load().PLANES == PLANES


# ---------------------------------------------------------------- SERIES_RULE "fluid"
def _ts_fs(rows):
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID",
                                       "Anatomical_Plane", "Fluid_Sensitive"])


def test_fluid_uu_tien_xung_nhay_dich(chon_series):
    """Series T1 dung truoc trong CSV: 'first' lay T1, 'fluid' phai lay series nhay dich."""
    ts = _ts_fs([("S1", "t1", "Sagittal", 0), ("S1", "fs", "Sagittal", 1),
                 ("S1", "ax", "Axial", 1), ("S1", "co", "Coronal", 0)])
    cu, _ = chon_series(ts, PLANES)
    moi, _ = chon_series(ts, PLANES, "fluid")

    sag = lambda w: w.loc[w["Anatomical_Plane"] == "Sagittal", "SeriesInstanceUID"].item()
    assert sag(cu) == "t1"
    assert sag(moi) == "fs"
    # Coronal chi co T1 - van phai giu, khong duoc bo trang mat phang
    assert set(moi["Anatomical_Plane"]) == set(PLANES)


def test_fluid_giu_thu_tu_csv_giua_cac_series_cung_loai(chon_series):
    """Hai series cung nhay dich: lay cai dung truoc, giong cach cu - khong doi ngau nhien."""
    ts = _ts_fs([("S1", "t1", "Sagittal", 0), ("S1", "fs_a", "Sagittal", 1),
                 ("S1", "fs_b", "Sagittal", 1)])
    moi, _ = chon_series(ts, PLANES, "fluid")
    assert moi["SeriesInstanceUID"].tolist() == ["fs_a"]


def test_mac_dinh_van_la_first_cho_head_hien_tai():
    """5 head dang nop train tren 'first'. Doi mac dinh truoc khi co head moi = diem sai am tham."""
    assert _load().SERIES_RULE == "first"


def test_script_06_va_50_chon_cung_series_tren_du_lieu_that(chon_series):
    """Train (06) va du doan (50) la hai ban sao rieng cua phep chon. Lech mot dong la head
    doc anh khac han luc train ma khong co gi bao."""
    csv = Path(__file__).resolve().parents[1] / "data" / "raw" / "train_series.csv"
    if not csv.exists():
        pytest.skip("chua co train_series.csv")
    m06 = _load_06()
    assert {pl for pl, rule in m06.JOBS if rule == "fluid"} == set(PLANES)

    ts = pd.read_csv(csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    want50, _ = chon_series(ts, PLANES, "fluid")
    for pl in PLANES:
        a = set(m06.chon_mot_series(ts[ts["Anatomical_Plane"] == pl], "fluid")["SeriesInstanceUID"])
        b = set(want50.loc[want50["Anatomical_Plane"] == pl, "SeriesInstanceUID"])
        assert a == b, pl


def _load_06():
    spec = importlib.util.spec_from_file_location(
        "kaggle_extract", SCRIPT.parent / "06_kaggle_extract_features.py")
    m06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m06)
    return m06


def test_t1_chi_lay_series_khong_nhay_dich_va_khong_lay_thay():
    """Goc nhin T1 ma lay series nhay dich thay vao khi thieu thi chi la ban sao cua 'fluid'."""
    m06 = _load_06()
    ts = _ts_fs([("S1", "t1", "Sagittal", 0), ("S1", "fs", "Sagittal", 1),
                 ("S2", "fs2", "Sagittal", 1)])
    w = m06.chon_mot_series(ts, "t1")
    assert w["SeriesInstanceUID"].tolist() == ["t1"]
