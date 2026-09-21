"""Test cho pooling co mask va loss bo qua nhan chua gan.

Hai co che nay hong am tham: quen mask thi pooling van chay, loss van giam, chi co ket qua
te di mot cach kho hieu.
"""
from __future__ import annotations

import pytest
import torch

from rsna_knee.head import MaskedPool, StudyHead, collate_features, masked_bce


def test_mean_pooling_bo_qua_lat_pad():
    """Study 2 lat that + 2 lat pad phai cho ket qua y het study chi co 2 lat that.

    Quen mask thi trung binh bi pha loang boi cac so 0 - sai lech nay khong bao loi.
    """
    x = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, True, False, False]])
    out = MaskedPool("mean")(x, mask)
    assert torch.allclose(out, torch.tensor([[3.0, 6.0]]))

    khong_mask = (x * 1.0).mean(1)
    assert not torch.allclose(out, khong_mask), "phai khac voi mean khong mask"


def test_max_pooling_pad_bang_am_vo_cuc_khong_phai_0():
    """Dac trung toan AM -> pad bang 0 se tra ve 0 (sai). Pad -inf moi ra dung gia tri."""
    x = torch.tensor([[[-5.0, -2.0], [-3.0, -7.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    out = MaskedPool("max")(x, mask)
    assert torch.allclose(out, torch.tensor([[-3.0, -2.0]]))
    assert (out < 0).all(), "max khong duoc bi keo len 0 boi lat pad"


def test_attention_pooling_khong_phan_bo_trong_so_cho_lat_pad():
    torch.manual_seed(0)
    pool = MaskedPool("attn", dim=4)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, False, False, False],
                         [True, True, True, True, True]])
    out = pool(x, mask)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()

    # Doi noi dung lat pad khong duoc lam doi ket qua
    x2 = x.clone()
    x2[0, 2:] = 999.0
    assert torch.allclose(out[0], pool(x2, mask)[0], atol=1e-5)


def test_pooling_khong_hop_le_bi_chan():
    with pytest.raises(ValueError):
        MaskedPool("trung_binh_cong")
    with pytest.raises(ValueError):
        MaskedPool("attn")          # thieu `dim`


def test_masked_bce_bo_qua_nhan_nan():
    """Nhan `NaN` = chua gan. Dieu kien bat buoc de tron nhan nguoi va nhan may o ngay 8.

    Dung hai cot LECH HAN nhau: cot 0 doan rat sai, cot 1 doan rat dung. Neu hai cot doi
    xung thi bo mot cot di cung khong lam loss doi, va test se khong chung minh duoc gi.
    """
    logits = torch.tensor([[-3.0, 3.0]])
    y_biet = torch.tensor([[1.0, 1.0]])            # cot 0 sai nang, cot 1 dung
    y_bo_cot_dung = torch.tensor([[1.0, float("nan")]])
    y_bo_cot_sai = torch.tensor([[float("nan"), 1.0]])

    loss_biet = masked_bce(logits, y_biet)
    loss_chi_sai = masked_bce(logits, y_bo_cot_dung)
    loss_chi_dung = masked_bce(logits, y_bo_cot_sai)
    assert torch.isfinite(loss_chi_sai) and torch.isfinite(loss_chi_dung)

    # Bo cot doan dung -> chi con cot sai -> loss phai TANG, va nguoc lai
    assert loss_chi_sai > loss_biet > loss_chi_dung


def test_masked_bce_khong_nhan_gradient_tu_vi_tri_nan():
    logits = torch.tensor([[0.5, 0.5]], requires_grad=True)
    masked_bce(logits, torch.tensor([[1.0, float("nan")]])).backward()
    grad = logits.grad[0]
    assert grad[0] != 0, "vi tri co nhan phai sinh gradient"
    assert grad[1] == 0, "vi tri NaN tuyet doi khong duoc sinh gradient"


def test_masked_bce_khi_khong_co_nhan_nao():
    """Batch toan NaN -> loss bang 0 va van backward duoc, khong duoc ra NaN."""
    logits = torch.tensor([[0.3, -0.7]], requires_grad=True)
    loss = masked_bce(logits, torch.full((1, 2), float("nan")))
    assert torch.isfinite(loss) and loss.detach().item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_collate_pad_ve_lat_lon_nhat_va_tra_mask_dung():
    """Bay da gap that: study co 19-36 lat nen `default_collate` vo."""
    batch = [
        (torch.ones(2, 4), torch.zeros(12), "ca_ngan"),
        (torch.ones(5, 4), torch.zeros(12), "ca_dai"),
    ]
    x, mask, y, ids = collate_features(batch)
    assert x.shape == (2, 5, 4)
    assert mask.sum(1).tolist() == [2, 5]
    assert (x[0, 2:] == 0).all(), "phan pad phai la 0"
    assert ids == ["ca_ngan", "ca_dai"] and y.shape == (2, 12)


def test_head_tra_ve_logit_chu_khong_phai_xac_suat():
    """Loss dung `BCEWithLogits` nen head KHONG duoc tu sigmoid - se bi bop hai lan."""
    torch.manual_seed(0)
    head = StudyHead(dim=8, pooling="mean")
    x = torch.randn(3, 4, 8) * 50          # dau vao lon de logit vuot [0, 1]
    mask = torch.ones(3, 4, dtype=torch.bool)
    out = head(x, mask)
    assert out.shape == (3, 12)
    assert out.min() < 0, "dau ra phai la logit, co the am"
