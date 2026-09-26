"""Kiem pooling chu y RIENG TUNG BENH.

Diem de sai am tham nhat cua kieu pooling nay: tron vector gop cua benh nay vao logit
cua benh kia. Neu tron, model van chay, loss van giam, diem van ra - chi la moi benh
dang nhin vao mot chong lat khong phai cua no. Test 3 chot dung cho cho do.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import LABELS
from rsna_knee.head import MaskedPool, PerLabelAttnPool, StudyHead, masked_bce

L = len(LABELS)
D = 16


def test_hinh_dang_dau_ra():
    pool = PerLabelAttnPool(D, L)
    x = torch.randn(3, 7, D)
    mask = torch.ones(3, 7, dtype=torch.bool)
    assert pool(x, mask).shape == (3, L, D)

    head = StudyHead(D, pooling="per_label_attn")
    assert head(x, mask).shape == (3, L)


def test_bo_qua_vi_tri_pad():
    """Lat pad KHONG duoc dong gop. Doi gia tri o do thi ket qua phai y nguyen."""
    torch.manual_seed(0)
    pool = PerLabelAttnPool(D, L).eval()
    x = torch.randn(2, 6, D)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, :4] = True                      # 2 lat cuoi la pad

    with torch.no_grad():
        a = pool(x, mask)
        x2 = x.clone()
        x2[:, 4:] = 999.0                   # dap gia tri cuc doan vao phan pad
        b = pool(x2, mask)
    assert torch.allclose(a, b, atol=1e-5), "lat pad van anh huong ket qua"


def test_moi_benh_chu_y_lat_khac_nhau():
    """Cot loi cua kieu pooling nay: 12 benh phai co 12 bo trong so KHAC nhau.

    Neu chung giong het nhau thi no chi la `attn` cu doi ten - khong duoc loi gi ma lai
    ton them tham so.
    """
    torch.manual_seed(0)
    pool = PerLabelAttnPool(D, L)
    x = torch.randn(1, 20, D)
    mask = torch.ones(1, 20, dtype=torch.bool)
    w = pool.score(x).masked_fill(~mask.unsqueeze(-1), float("-inf")).softmax(dim=1)

    assert w.shape == (1, 20, L)
    assert torch.allclose(w.sum(dim=1), torch.ones(1, L), atol=1e-5), \
        "trong so phai cong lai bang 1 TREN CAC LAT"
    w = w.detach()
    khac = max(float((w[0, :, i] - w[0, :, j]).abs().max())
               for i in range(L) for j in range(i + 1, L))
    assert khac > 1e-4, "moi benh dang dung chung mot bo trong so"


def test_logit_khong_tron_giua_cac_benh():
    """Doi vector gop cua benh j thi CHI logit cua benh j duoc doi.

    Day la test quan trong nhat: `nn.Linear(d, L)` dat sau pooling (B,L,d) se lam moi
    logit an theo ca 12 vector gop - sai ma khong co dau hieu gi.
    """
    torch.manual_seed(0)
    head = StudyHead(D, pooling="per_label_attn", dropout=0.0).eval()
    pooled = torch.randn(1, L, D)

    with torch.no_grad():
        goc = (pooled * head.w).sum(-1) + head.b
        doi = pooled.clone()
        doi[0, 3] += 10.0                   # chi dung vao vector gop cua benh so 3
        moi = (doi * head.w).sum(-1) + head.b

    khac = (moi - goc).abs()[0]
    assert khac[3] > 1e-3, "benh bi doi le ra phai doi logit"
    assert khac[torch.arange(L) != 3].max() < 1e-6, \
        "benh KHAC cung bi doi logit -> dang tron giua cac benh"


def test_train_duoc_voi_nhan_khuyet():
    """Chay tron mot buoc train voi nhan NaN - dung dieu kien that cua ngay 8."""
    torch.manual_seed(0)
    head = StudyHead(D, pooling="per_label_attn")
    x = torch.randn(4, 9, D)
    mask = torch.ones(4, 9, dtype=torch.bool)
    y = torch.rand(4, L)
    y[y < 0.4] = float("nan")               # ~40% o chua ket luan duoc

    loss = masked_bce(head(x, mask), y)
    assert torch.isfinite(loss)
    loss.backward()
    g = [p.grad for p in head.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(t).all() for t in g)
    assert head.w.grad is not None and float(head.w.grad.abs().sum()) > 0


@pytest.mark.parametrize("mode", ["mean", "max", "attn"])
def test_khong_lam_hong_ba_kieu_cu(mode):
    """Checkpoint cu phai nap duoc y nguyen - khong duoc doi kien truc cua ba kieu kia."""
    head = StudyHead(D, pooling=mode)
    x = torch.randn(2, 5, D)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert head(x, mask).shape == (2, L)
    assert hasattr(head, "fc") and not hasattr(head, "w")
    assert isinstance(head.pool, MaskedPool)
