"""File OOF len repo cong khai nen khong mang nhan; doi lai, thu tu ca phai duoc kiem.

Lech thu tu mot ca thi du doan cua ca nay bi cham bang nhan cua ca kia - van ra mot so
AUC hop ly, chi la sai. Ma bam thu tu ca la cho duy nhat bat duoc chuyen do.
"""
from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.oof import ma_thu_tu, nap_oof


def test_doi_cho_hai_ca_la_doi_ma():
    assert ma_thu_tu(["a", "b", "c"]) != ma_thu_tu(["b", "a", "c"])
    assert ma_thu_tu(["a", "b"]) == ma_thu_tu(np.array(["a", "b"]))


def test_nap_oof_tu_choi_thu_tu_khac(tmp_path):
    uid, fold = np.array(["a", "b", "c"]), np.array([0, 1, 0])
    f = tmp_path / "oof.npz"
    np.savez(f, per_ep=np.zeros((2, 3, 12)), fold=fold, uid_md5=ma_thu_tu(uid))
    assert nap_oof(f, uid, fold).shape == (2, 3, 12)
    with pytest.raises(SystemExit):
        nap_oof(f, uid[::-1], fold)
    with pytest.raises(SystemExit):
        nap_oof(f, uid, np.array([1, 1, 0]))
