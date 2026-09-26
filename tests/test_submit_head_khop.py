"""NGAY 11 - Ban SAO CHEP cua StudyHead trong notebook nop bai co khop repo khong?

Notebook nop bai khong import duoc package cua repo (Kaggle chi nhan mot cell code), nen
`StudyHead` va `MaskedPool` bi CHEP sang `scripts/50_kaggle_submit.py`. Code bi chep la
code se lech - cau hoi chi la lech luc nao.

Neu lech ten khoa thi `load_state_dict` bao loi ngay, con song duoc. Nhung neu lech PHEP
TINH ma ten khoa van trung - vi du quen `masked_fill` o pooling, hay doi thu tu dropout -
thi checkpoint van nap duoc, notebook van chay, submission.csv van hop le, chi la moi
con so deu khac voi luc do CV. Day dung la kieu hong khong co dau hieu nao.

Test nay so ca hai: cau truc (ten khoa) va phep tinh (dau ra tren cung trong so).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsna_knee.config import LABELS
from rsna_knee.head import StudyHead as HeadRepo

D = 32
POOLINGS = ["mean", "max", "attn", "per_label_attn"]


@pytest.fixture(scope="module")
def sub():
    spec = importlib.util.spec_from_file_location(
        "kaggle_submit_head", REPO / "scripts" / "50_kaggle_submit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kaggle_submit_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_thu_tu_nhan_khop(sub):
    """Lech thu tu nhan = moi cot trong submission.csv gan sai benh, khong bao loi gi."""
    assert sub.LABELS == list(LABELS)


@pytest.mark.parametrize("pooling", POOLINGS)
def test_ten_khoa_khop(sub, pooling):
    a = HeadRepo(D, pooling=pooling)
    b = sub.StudyHead(D, pooling=pooling)
    assert sorted(a.state_dict()) == sorted(b.state_dict()), \
        f"pooling {pooling}: ten khoa lech -> load_state_dict se bao thieu khoa"
    for k in a.state_dict():
        assert a.state_dict()[k].shape == b.state_dict()[k].shape, \
            f"pooling {pooling}: khoa {k} lech hinh dang"


@pytest.mark.parametrize("pooling", POOLINGS)
def test_phep_tinh_khop(sub, pooling):
    """Cung trong so, cung dau vao -> phai cung dau ra tung bit.

    Day la nua quan trong hon: ten khoa trung nhung phep tinh lech thi khong gi bao.
    """
    torch.manual_seed(0)
    a = HeadRepo(D, pooling=pooling).eval()
    b = sub.StudyHead(D, pooling=pooling).eval()
    b.load_state_dict(a.state_dict())          # khong strict=False: phai khop tuyet doi

    x = torch.randn(3, 11, D)
    mask = torch.zeros(3, 11, dtype=torch.bool)
    mask[:, :7] = True                          # co ca pad, de kiem duong masked_fill
    mask[2, :3] = True                          # moi ca mot so lat khac nhau

    with torch.no_grad():
        ra, rb = a(x, mask), b(x, mask)
    assert ra.shape == rb.shape == (3, len(LABELS))
    assert torch.allclose(ra, rb, atol=1e-6), \
        f"pooling {pooling}: cung trong so nhung ra khac -> phep tinh da lech"


@pytest.mark.parametrize("pooling", POOLINGS)
def test_bo_qua_lat_pad(sub, pooling):
    """Ban sao chep cung phai bo qua lat pad. Quen mask = pooling bi pha loang boi so 0."""
    torch.manual_seed(1)
    b = sub.StudyHead(D, pooling=pooling).eval()
    x = torch.randn(2, 8, D)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :5] = True

    with torch.no_grad():
        goc = b(x, mask)
        x2 = x.clone()
        x2[:, 5:] = 123.0
        moi = b(x2, mask)
    assert torch.allclose(goc, moi, atol=1e-5), \
        f"pooling {pooling}: lat pad van anh huong ket qua"


def test_chuan_hoa_anh_khop(sub):
    """`normalize_series` cua notebook phai giong het ban o repo.

    Lech mot chi tiet o day (vd percentile 2-98 thay vi 1-99) thi dac trung luc nop khac
    dac trung luc train, va diem tut ma khong hieu vi sao.
    """
    import numpy as np

    from rsna_knee.dicom_io import normalize_series as repo_fn

    rng = np.random.default_rng(0)
    arr = (rng.random((6, 20, 20)) * 4000).astype(np.int16)
    a = repo_fn(arr.astype(np.float32))
    b = sub.normalize_series(arr.astype(np.float32))
    assert a.dtype == b.dtype == np.float32
    assert np.allclose(a, b, atol=1e-6), "tien xu ly anh hai ben da lech"
