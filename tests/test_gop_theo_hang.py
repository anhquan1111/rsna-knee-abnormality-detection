"""Phep gop theo hang trong notebook nop bai co GIONG phep da dung de do 0.8112 khong?

Diem 0.8112 duoc do o may bang `scipy.stats.rankdata`. Notebook nop bai tu viet lai phep
xep hang bang `np.argsort` hai lan - vi khong muon phu thuoc scipy tren Kaggle.

Hai ban do neu lech nhau thi: notebook van chay, submission.csv van hop le, chi la diem
that KHAC diem da do. Va vi CV cua du an nay bam LB toi 0.003, mot sai lech nhu vay se
bi doc nham thanh "bang rieng khac bang cong khai" thay vi "code nop bai sai".

Test nay khoa hai ban lai voi nhau.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import LABELS

L = len(LABELS)


def xep_hang_nhu_notebook(z: np.ndarray) -> np.ndarray:
    """Sao y doan trong scripts/50_kaggle_submit.py."""
    hang = np.empty_like(z)
    for j in range(z.shape[1]):
        thu_tu = np.argsort(np.argsort(z[:, j]))
        hang[:, j] = (thu_tu + 1) / len(z)
    return hang


def xep_hang_nhu_luc_do(z: np.ndarray) -> np.ndarray:
    """Sao y doan da dung de do 0.8112 (scripts/43, 45)."""
    out = np.empty_like(z)
    for j in range(z.shape[1]):
        out[:, j] = rankdata(z[:, j]) / len(z)
    return out


def test_hai_ban_xep_hang_khop():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, L))
    assert np.allclose(xep_hang_nhu_notebook(z), xep_hang_nhu_luc_do(z))


def test_giu_nguyen_thu_hang():
    """Dieu duy nhat AUC quan tam: thu hang. Phep xep hang phai giu dung no."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(120, L))
    h = xep_hang_nhu_notebook(z)
    for j in range(L):
        assert np.array_equal(np.argsort(z[:, j]), np.argsort(h[:, j]))


def test_bat_bien_voi_thang_do_tung_model():
    """Cot loi cua viec gop THEO HANG thay vi theo gia tri.

    Nhan logit cua mot model len 100 lan (lam no "tu tin" hon han) KHONG duoc doi ket qua
    gop. Neu doi, tuc la dang gop theo gia tri va model on ao nhat se lan at.
    """
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(80, L)), rng.normal(size=(80, L))
    goc = (xep_hang_nhu_notebook(a) + xep_hang_nhu_notebook(b)) / 2
    moi = (xep_hang_nhu_notebook(a * 100) + xep_hang_nhu_notebook(b)) / 2
    assert np.allclose(goc, moi)

    # Doi lai: trung binh THEO GIA TRI thi bi lan at - day la thu ta co y tranh.
    tb_goc = (a + b) / 2
    tb_moi = (a * 100 + b) / 2
    assert not np.allclose(np.argsort(tb_goc[:, 0]), np.argsort(tb_moi[:, 0]))


def test_gop_khi_mot_head_khong_chay_duoc():
    """Ca thieu mat phang -> mot so head khong chay. Trung binh phai tinh tren so head
    THAT SU chay duoc, khong phai chia cung cho 5."""
    n = 50
    rng = np.random.default_rng(3)
    uids = [f"u{i}" for i in range(n)]
    # head A chay het, head B chi chay cho 30 ca dau
    m = {"A": {u: rng.normal(size=L) for u in uids},
         "B": {u: rng.normal(size=L) for u in uids[:30]}}

    tong, dem = np.zeros((n, L)), np.zeros((n, 1))
    for ten, d in m.items():
        co = [k for k, u in enumerate(uids) if u in d]
        z = np.stack([d[uids[k]] for k in co])
        tong[co] += xep_hang_nhu_notebook(z)
        dem[co] += 1
    prob = np.full((n, L), 0.5)
    co_du = dem[:, 0] > 0
    prob[co_du] = tong[co_du] / dem[co_du]

    assert (dem[:30] == 2).all() and (dem[30:] == 1).all()
    assert np.isfinite(prob).all()
    assert (prob >= 0).all() and (prob <= 1).all()


def test_ket_qua_nam_trong_khoang_hop_le():
    """Trung binh cac hang luon trong [0,1] -> qua duoc phep kiem truoc khi ghi file."""
    rng = np.random.default_rng(4)
    z = [rng.normal(size=(60, L)) * s for s in (1, 10, 0.01, 100, 5)]
    gop = sum(xep_hang_nhu_notebook(x) for x in z) / len(z)
    assert (gop > 0).all() and (gop <= 1).all()


@pytest.mark.parametrize("n", [2, 3, 10])
def test_chay_duoc_voi_it_ca(n):
    """Tap test hien thi luc soan notebook chi co 3 ca - phep xep hang phai khong vo."""
    rng = np.random.default_rng(5)
    z = rng.normal(size=(n, L))
    h = xep_hang_nhu_notebook(z)
    assert h.shape == (n, L)
    assert np.isfinite(h).all()
