"""Test cho notebook 81 (tien xu ly + CNN 2.5D) - chay o may, khong can GPU hay anh.

Ba cho hong am tham:
  * chon nham series (T1 thay vi nhay dich) - model van train, chi kem di;
  * cua so 2.5D lay lat khong lien ke - van ra anh 3 kenh, chi la vo nghia;
  * ca thieu mat phang lam attention softmax tren toan -inf -> NaN lan vao loss.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _nap(ten: str):
    spec = importlib.util.spec_from_file_location(ten, SCRIPTS / f"{ten}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_81_uu_tien_series_nhay_dich_va_giu_duong_lui():
    m = _nap("81_kaggle_train_25d")
    ts = pd.DataFrame({"StudyInstanceUID": ["S"] * 3, "SeriesInstanceUID": ["t1", "fs", "t1b"],
                       "Anatomical_Plane": ["Sagittal"] * 3, "Fluid_Sensitive": [0, 1, 0]})
    thu_tu = m.thu_tu_series(ts)["SeriesInstanceUID"].tolist()
    assert thu_tu[0] == "fs", "series nhay dich phai duoc chon truoc"
    assert thu_tu[1:] == ["t1", "t1b"], "duong lui giu thu tu CSV"


def test_81_cua_so_25d_la_ba_lat_lien_ke(tmp_path):
    m = _nap("81_kaggle_train_25d")
    D = 6
    x = np.zeros((3, D, 4, 4), dtype=np.uint8)
    for d in range(D):
        x[:, d] = d                                   # gia tri pixel = so thu tu lat
    np.savez(tmp_path / "ca.npz", x=x, co=np.array([True, False, True]))
    w, co, _, _ = m.CaDataset(["ca"], np.zeros((1, 12)), tmp_path, train=False)[0]
    assert w.shape == (3, D, 3, 4, 4)
    tam = w[0, :, 1, 0, 0].numpy()
    assert tam.tolist() == list(range(D)), "khi cham, moi lat lam tam mot cua so"
    assert w[0, 2, :, 0, 0].tolist() == [1, 2, 3], "kenh = lat truoc, lat tam, lat sau"
    assert w[0, 0, :, 0, 0].tolist() == [0, 0, 1], "bien dau: kep lai, khong lay lat am"
    assert co.tolist() == [True, False, True]


def test_81_ca_khong_co_mat_phang_nao_khong_ra_nan():
    pytest.importorskip("timm")
    m = _nap("81_kaggle_train_25d")
    net = m.Mang25D("resnet10t", pretrained=False).eval()
    x = torch.rand(2, 3, 2, 3, 32, 32)
    co = torch.tensor([[False, False, False], [True, False, True]])
    with torch.no_grad():
        out = net(x, co)
    assert out.shape == (2, 12)
    assert torch.isfinite(out).all(), "softmax tren toan -inf phai co duong lui"


def test_81_mat_phang_bi_che_khong_anh_huong_du_doan():
    """Mat phang co=False la anh 0 do thieu series - doi noi dung cua no khong duoc doi logit."""
    pytest.importorskip("timm")
    m = _nap("81_kaggle_train_25d")
    torch.manual_seed(0)
    net = m.Mang25D("resnet10t", pretrained=False).eval()
    x = torch.rand(1, 3, 2, 3, 32, 32)
    co = torch.tensor([[True, False, True]])
    x2 = x.clone()
    x2[:, 1] = torch.rand_like(x2[:, 1])
    with torch.no_grad():
        assert torch.allclose(net(x, co), net(x2, co), atol=1e-5)
