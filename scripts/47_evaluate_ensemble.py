"""Danh gia bo gop 5 model cuoi tren 58 ca gold.

    python scripts/47_evaluate_ensemble.py

Doc du doan out-of-fold cua 5 head (do 43_head_tung_mat_phang.py va cac thi nghiem ngay
11-12 ghi ra), gop theo hang dung nhu notebook nop bai, roi cham macro AUC + KTC 95%.
Khong train lai - 45_train_final_ensemble.py mat hon 10 phut.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.stats import rankdata

from rsna_knee.config import LABELS, REPORTS_DIR
from rsna_knee.metrics import macro_auc

L = list(LABELS)


def theo_hang(p: np.ndarray) -> np.ndarray:
    """Hang THO, khong chia cho n. Hang la boi so cua 0.5 nen cong 5 model van chinh xac
    tuyet doi; chia cho n roi cong so thuc se lam vo vai cap hoa nhau (diem gop co 66 cap)
    va AUC nhay o chu so thu tu. AUC khong doi khi nhan ca cot voi mot hang so."""
    out = np.empty_like(p, dtype=float)
    for j in range(p.shape[1]):
        out[:, j] = rankdata(p[:, j])
    return out


def main() -> None:
    mp = np.load(REPORTS_DIR / "day11_mat_phang_oof.npz")
    ck = np.load(REPORTS_DIR / "day11_ck_oof.npz")
    b336 = np.load(REPORTS_DIR / "day12_b336_oof.npz")
    y = mp["y"]
    assert np.array_equal(y, ck["y"], equal_nan=True) and np.array_equal(y, b336["y"], equal_nan=True)

    models = {
        "mp_sagittal  (chi Sagittal)": mp["sagittal"],
        "mp_axial     (chi Axial)": mp["axial"],
        "mp_coronal   (chi Coronal)": mp["coronal"],
        "chung_s224   (ca 3 mat phang)": ck["D_llm"],
        "chung_b336   (ca 3, backbone lon)": b336["D_llm"],
    }
    gop = np.mean([theo_hang(p) for p in models.values()], axis=0)

    print(f"Danh gia tren {len(y)} ca co nhan bac si (out-of-fold, 5 fold)\n")
    print(f"  {'Model':36s} macro AUC")
    print(f"  {'Doan bua (hang so)':36s}    0.5000")
    for ten, p in models.items():
        print(f"  {ten:36s}    {macro_auc(y, p, L).macro_auc:.4f}")
    kq = macro_auc(y, gop, L)
    print(f"  {'GOP CA 5 (trung binh theo hang)':36s}    {kq.macro_auc:.4f}   <-- model cuoi")

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        r = macro_auc(y[idx], gop[idx], L)
        if r.n_scored == len(L):
            boot.append(r.macro_auc)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  Khoang tin cay 95% (bootstrap theo ca): [{lo:.3f}, {hi:.3f}]")
    print("  Diem that tren Kaggle public LB        : 0.771")

    print("\nAUC tung benh cua model cuoi:")
    for nhan, auc in sorted(kq.per_label.items(), key=lambda kv: -kv[1]):
        print(f"  {nhan:18s} {auc:.3f}  {'#' * round(auc * 30)}")

    acc_hang_so = float(np.nanmean(y == 0))
    print("\nVi sao khong bao cao accuracy:")
    print(f"  Doan 'khong benh' cho MOI ca -> accuracy {acc_hang_so:.1%}, nhung AUC = 0.500")


if __name__ == "__main__":
    main()
