"""Test cho evaluator macro AUC.

Day la ham quan trong nhat cua ca du an: moi ket luan o ngay 5 va ngay 8 deu di qua no.
Cac test duoi khong kiem "ham co chay khong" ma kiem dung nhung HANH VI DE HONG AM THAM
da ghi trong tai lieu ngay 4 - nhung cho ma sai se cho ra mot con so dep chu khong phai
mot dong loi.
"""
from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.metrics import MacroAUCResult, macro_auc, prevalence_baseline

L3 = ["a", "b", "c"]


def test_diem_dung_tren_truong_hop_biet_truoc():
    """Xep hang hoan hao -> 1.0; xep hang nguoc han -> 0.0; trung binh dung.

    "Nguoc han" nghia la MOI ca duong tinh deu xep duoi MOI ca am tinh. Chi doi cho vai
    cap thi AUC ra 0.25 chu khong phai 0 - AUC dem ti le cap xep dung, khong phai so o sai.
    """
    y = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)
    p = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]], dtype=float)
    res = macro_auc(y, p, ["thuan", "nguoc"])
    assert res.per_label["thuan"] == pytest.approx(1.0)
    assert res.per_label["nguoc"] == pytest.approx(0.0)
    assert res.macro_auc == pytest.approx(0.5)
    assert res.n_scored == 2


def test_nhan_mot_lop_bi_bo_va_duoc_bao_cao():
    """Bay chinh cua ngay 4: `roc_auc_score` tra `nan` chu khong nem loi.

    Neu evaluator khong tu kiem truoc, `np.nanmean` se cho ra mot con so dep trong khi
    am tham bo nhan do khoi mau so - va khong co gi bao cho biet.
    """
    y = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=float)  # 'b' toan 1
    p = np.random.default_rng(0).random((4, 3))
    res = macro_auc(y, p, L3)

    assert "b" in res.skipped
    assert res.n_scored == 2, "nhan mot lop phai bi loai khoi mau so"
    assert "b" not in res.per_label
    assert not np.isnan(res.macro_auc)


def test_khong_bao_gio_tra_ve_nan():
    """Du lieu the nao cung phai ra so that hoac nem loi - tuyet doi khong tra `nan`."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = rng.integers(0, 2, (6, 3)).astype(float)
        p = rng.random((6, 3))
        try:
            res = macro_auc(y, p, L3)
        except ValueError:
            continue                      # khong nhan nao cham duoc - bao loi la dung
        assert not np.isnan(res.macro_auc)
        assert res.n_scored >= 1


def test_khong_cham_duoc_thi_nem_loi_chu_khong_tra_so():
    """Moi nhan deu mot lop -> phai nem ValueError, khong duoc tra ve mot so vo nghia."""
    y = np.ones((4, 3), dtype=float)
    p = np.random.default_rng(2).random((4, 3))
    with pytest.raises(ValueError):
        macro_auc(y, p, L3)


def test_logit_va_xac_suat_cho_cung_diem():
    """AUC bat bien qua phep don dieu tang -> khong can sigmoid truoc khi cham.

    Quan trong vi vong train cham AUC thang tren logit. Neu tinh chat nay bi pha (vi du
    ai do them clip), diem se lech ma khong co loi nao.
    """
    rng = np.random.default_rng(3)
    y = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=float)
    logits = rng.normal(size=y.shape)
    probs = 1 / (1 + np.exp(-logits))
    assert macro_auc(y, logits, ["x", "y"]).macro_auc == pytest.approx(
        macro_auc(y, probs, ["x", "y"]).macro_auc
    )


def test_nhan_chua_gan_bi_bo_qua_chu_khong_thanh_am_tinh():
    """`NaN` trong y_true nghia la CHUA GAN. Coi no la 0 se boi den diem mot cach sai lech."""
    y_full = np.array([[0.0], [1.0], [0.0], [1.0]])
    p = np.array([[0.1], [0.9], [0.2], [0.8]])
    y_partial = y_full.copy()
    y_partial[2, 0] = np.nan          # bo mot ca ra khoi phep cham

    assert macro_auc(y_full, p, ["z"]).macro_auc == pytest.approx(1.0)
    assert macro_auc(y_partial, p, ["z"]).macro_auc == pytest.approx(1.0)


def test_lech_shape_bi_chan_som():
    y = np.zeros((4, 3))
    with pytest.raises(ValueError):
        macro_auc(y, np.zeros((4, 2)), L3)          # y_score lech cot
    with pytest.raises(ValueError):
        macro_auc(y, np.zeros((4, 3)), ["a", "b"])  # so ten nhan lech


def test_baseline_hang_so_dung_bang_0_5():
    """Doan ti le mac benh cho moi ca -> moi ca cung diem -> AUC dung bang 0.5.

    Day la moc so sanh cua ngay 5: model nao khong vuot qua 0.5 la chua hoc duoc gi.
    """
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, (40, 3)).astype(float)
    res = macro_auc(y, prevalence_baseline(y, len(y)), L3)
    assert res.macro_auc == pytest.approx(0.5)
    assert res.n_scored == 3


def test_ket_qua_luon_mang_theo_mau_so():
    """`MacroAUCResult` phai in kem so nhan cham duoc - mot con so tran la vo nghia."""
    y = np.array([[0, 1], [1, 1], [0, 1], [1, 1]], dtype=float)
    res = macro_auc(y, np.random.default_rng(5).random((4, 2)), ["p", "q"])
    assert isinstance(res, MacroAUCResult)
    assert "1 nhan" in str(res) and "bo 1" in str(res)
    row = res.to_row(config="thu")
    assert row["n_scored"] == 1 and row["n_skipped"] == 1 and row["config"] == "thu"
