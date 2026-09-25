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
