"""Test cho chuan hoa cuong do MRI.

Khong can file DICOM that: cac test duoi dung mang numpy dung san de kiem dung CO CHE
da do o ngay 3 - cai gi duoc giu lai va cai gi bi xoa mat.
"""
from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.dicom_io import normalize_per_slice, normalize_series, resize_volume


def test_chuan_hoa_series_dua_ve_khoang_0_1():
    rng = np.random.default_rng(0)
    arr = (rng.random((8, 32, 32)) * 2000).astype(np.int16)
    out = normalize_series(arr)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_series_toan_mot_gia_tri_khong_sinh_nan():
    """`+ 1e-8` khong phai trang tri: anh den / anh hong cho p99 == p1 -> chia cho 0.

    Neu tran ra `inf`/`nan` thi no lan sang loss va lam sap ca vong train.
    """
    out = normalize_series(np.full((4, 16, 16), 7, dtype=np.int16))
    assert np.isfinite(out).all(), "series hang so phai cho ra so huu han"


def test_chuan_hoa_theo_slice_xoa_do_sang_tuong_doi_giua_cac_lat():
    """Bang chung cot loi cua ngay 3, o dang test tu dong.

    Dung 5 lat co do sang tang dan. Chuan hoa theo ca series thi thu bac do sang duoc giu;
    chuan hoa theo tung lat thi moi lat bi ep ve cung mot dai -> thu bac bien mat.
    """
    lats = [np.full((16, 16), 100 * (i + 1), dtype=np.int16) for i in range(5)]
    for i, lat in enumerate(lats):                 # them bien thien de percentile co nghia
        lat[:, :8] = 100 * (i + 1) // 2
    arr = np.stack(lats)

    theo_series = normalize_series(arr)
    theo_lat = normalize_per_slice(arr)

    p99_series = np.percentile(theo_series, 99, axis=(1, 2))
    p99_lat = np.percentile(theo_lat, 99, axis=(1, 2))

    assert p99_series.std() > 0.1, "chuan hoa theo series phai GIU thu bac do sang"
    assert p99_lat.std() < 0.01, "chuan hoa theo tung lat phai XOA thu bac do sang"


def test_chia_hang_so_vuot_khoi_khoang_0_1():
    """MRI khong phai anh 8-bit: dai gia tri len toi hang nghin."""
    arr = np.full((3, 8, 8), 2152, dtype=np.uint16)
    assert (arr.astype(np.float32) / 255.0).max() > 1.0


def test_resize_giu_so_lat_va_khoang_gia_tri():
    rng = np.random.default_rng(1)
    arr = normalize_series((rng.random((6, 64, 64)) * 1500).astype(np.int16))
    out = resize_volume(arr, 224)
    assert out.shape == (6, 224, 224)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_percentile_cat_duoc_diem_sang_bat_thuong():
    """Mot diem sang cuc manh khong duoc keo toan bo anh xuong toi.

    Nen anh phai co bien thien that: neu dung nen hang so thi p1 == p99 va phep chuan hoa
    suy bien ve 0 het - luc do test khong con do duoc dieu no muon do.
    """
    rng = np.random.default_rng(7)
    arr = (rng.normal(500, 60, (4, 32, 32))).astype(np.int16)
    sach = normalize_series(arr)

    co_nhieu = arr.copy()
    co_nhieu[0, 0, 0] = 30000                  # nhieu xung, khong phai mo
    out = normalize_series(co_nhieu)

    assert out.max() <= 1.0
    # Percentile 99 cat bo diem sang -> phan mo con lai gan nhu khong doi
    assert abs(out[1].mean() - sach[1].mean()) < 0.02, "diem sang khong duoc lam toi ca anh"


@pytest.mark.parametrize("size", [64, 224])
def test_thu_tu_chuan_hoa_roi_resize_khac_voi_lam_nguoc(size):
    """Lam nguoc thu tu thi noi suy tao gia tri moi o bien, keo percentile lech di.

    Dung anh TRON (gradient + khoi sang) chu khong phai nhieu trang: anh MRI that co cau
    truc khong gian, con nhieu trang la truong hop xau nhat cua phep thu nho - no cho lech
    gap doi va lam test do nham do manh cua nhieu thay vi do hieu ung thu tu xu ly.
    """
    yy, xx = np.mgrid[0:100, 0:100]
    nen = (yy * 8 + xx * 4).astype(np.float32)
    arr = np.stack([nen + i * 60 for i in range(4)])
    arr[:, 30:60, 30:60] += 700                      # mot khoi sang nhu vung ton thuong
    arr = arr.astype(np.int16)

    a = resize_volume(normalize_series(arr), size)
    b = normalize_series(resize_volume(arr.astype(np.float32), size))
    lech = np.abs(a - b).mean()
    assert lech > 0, "hai thu tu phai cho ket qua khac nhau"
    assert lech < 0.05, f"nhung khong duoc khac qua nhieu (do duoc {lech:.4f})"
