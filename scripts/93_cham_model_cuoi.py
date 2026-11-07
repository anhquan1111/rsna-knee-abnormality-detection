"""NGAY 15 - Cham model cuoi DUNG danh sach da chot o reports/CHOT_MODEL_CUOI.md.

    python scripts/93_cham_model_cuoi.py --tai   # tai du doan 3 lan chay Kaggle roi cham
    python scripts/93_cham_model_cuoi.py         # cham lai tu file da tai

Model chinh: gop theo hang 6 CNN (#1-#6). Kem: 4 CNN nhan cong khai, 2 CNN nhan tu gan
(so bat cap voi 2 CNN cong khai cung backbone/seed), moc cu 0.8848. Moi model lay EMA
epoch cuoi. Khong loai model vi diem thap - chi khi hong ky thuat (thieu file / thieu ca).

Ghi reports/day15_model_cuoi.json (chi so tong hop) va reports/day15_model_cuoi_du_doan.npz
(du doan gold tung model, KHONG nhan, chi ma bam thu tu ca).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.stats import rankdata

from rsna_knee.config import LABELS, REPORTS_DIR
from rsna_knee.metrics import macro_auc
from rsna_knee.oof import gold_theo_fold, ma_thu_tu

L = list(LABELS)
REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(REPO / ".venv" / "bin" / "kaggle")
# (so thu tu, lan chay Kaggle, thu muc tai ve, thu muc model trong output, nhan)
DANH_SACH = [
    (1, "rsna-knee-train-25d", "kaggle_cnn", "efficientnet_b3_s42", "cong_khai"),
    (2, "rsna-knee-train-25d", "kaggle_cnn", "tf_efficientnetv2_b2_s7", "cong_khai"),
    (3, "rsna-knee-train-25d-seed-moi", "kaggle_cnn_seed_moi", "efficientnet_b3_s123", "cong_khai"),
    (4, "rsna-knee-train-25d-seed-moi", "kaggle_cnn_seed_moi", "tf_efficientnetv2_b2_s2026", "cong_khai"),
    (5, "rsna-knee-train-25d-nhan-minh", "kaggle_cnn_nhan_minh", "efficientnet_b3_s42", "tu_gan"),
    (6, "rsna-knee-train-25d-nhan-minh", "kaggle_cnn_nhan_minh", "tf_efficientnetv2_b2_s7", "tu_gan"),
]


def hang(p):
    return np.column_stack([rankdata(p[:, j]) for j in range(p.shape[1])])


def gop(ps):
    return np.mean([hang(p) for p in ps], axis=0)


def _auc_nhanh(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def _macro(y, p):
    return float(np.mean([_auc_nhanh(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def _mau(y, n, seed):
    rng = np.random.default_rng(seed)
    while n:
        i = rng.integers(0, len(y), len(y))
        if all(0 < y[i, j].sum() < len(i) for j in range(y.shape[1])):
            n -= 1
            yield i


def ktc(y, p, n=2000):
    return [round(float(v), 4) for v in np.percentile([_macro(y[i], p[i]) for i in _mau(y, n, 0)], [2.5, 97.5])]


def ktc_cap(y, a, b, n=2000):
    d = [_macro(y[i], b[i]) - _macro(y[i], a[i]) for i in _mau(y, n, 0)]
    return [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]


def main() -> None:
    if "--tai" in sys.argv:
        for _, kernel, thu_muc, _, _ in DANH_SACH:
            ra = REPO / "data" / thu_muc
            ra.mkdir(parents=True, exist_ok=True)
            subprocess.run([KAGGLE, "kernels", "output", f"votrananhquan/{kernel}", "-p", str(ra), "-o",
                            "--file-pattern", r"(du_doan\.npz|nhat_ky\.json|\.log)$"], check=True)

    y, fold, uid = gold_theo_fold()
    P, hong = {}, {}
    for so, _, thu_muc, mo_hinh, nhan in DANH_SACH:
        f = REPO / "data" / thu_muc / mo_hinh / "du_doan.npz"
        if not f.exists():
            hong[so] = f"thieu {f.relative_to(REPO)}"
            continue
        z = np.load(f)
        vt = {u: k for k, u in enumerate(z["gold_uid"])}
        if any(u not in vt for u in uid):
            hong[so] = "thieu du doan cho ca gold"
            continue
        P[so] = z["gold_ema"][-1][[vt[u] for u in uid]]
    if hong:
        print("MODEL HONG KY THUAT (bi loai, ghi ro):", hong)

    nhom = {"gop_6_cnn (chinh)": [1, 2, 3, 4, 5, 6], "gop_4_cnn_cong_khai": [1, 2, 3, 4],
            "gop_2_cnn_tu_gan": [5, 6], "gop_2_cnn_cong_khai (moc cu)": [1, 2]}
    kq = {"danh_sach": [{"so": s, "kernel": k, "mo_hinh": m, "nhan": n} for s, k, _, m, n in DANH_SACH],
          "model_hong": hong, "tung_model": {}, "nhom": {}}
    for so, p in P.items():
        kq["tung_model"][so] = {"auc": round(_macro(y, p), 4), "ci": ktc(y, p)}
    G = {}
    for ten, ds in nhom.items():
        co = [s for s in ds if s in P]
        if not co:
            continue
        G[ten] = gop([P[s] for s in co])
        r = macro_auc(y, G[ten], L)
        kq["nhom"][ten] = {"model": co, "auc": round(r.macro_auc, 4), "ci": ktc(y, G[ten]),
                           "tung_benh": {k: round(v, 3) for k, v in r.per_label.items()}}
    moc = "gop_2_cnn_cong_khai (moc cu)"
    so_sanh = {}
    for ten in ("gop_6_cnn (chinh)", "gop_4_cnn_cong_khai", "gop_2_cnn_tu_gan"):
        if ten in G and moc in G:
            so_sanh[f"{ten} - moc cu"] = {"chenh": round(_macro(y, G[ten]) - _macro(y, G[moc]), 4),
                                          "ci": ktc_cap(y, G[moc], G[ten])}
    kq["so_sanh"] = so_sanh
    np.savez_compressed(REPORTS_DIR / "day15_model_cuoi_du_doan.npz", fold=fold, uid_md5=ma_thu_tu(uid),
                        **{f"model_{s}": p for s, p in P.items()})
    (REPORTS_DIR / "day15_model_cuoi.json").write_text(json.dumps(kq, indent=2, ensure_ascii=False))

    print("Tung model (EMA epoch cuoi):")
    for s, v in kq["tung_model"].items():
        _, k, _, m, n = DANH_SACH[s - 1]
        print(f"  #{s} {m:<28} nhan {n:<9} {v['auc']:.4f}  KTC {v['ci']}")
    print("\nNhom (gop theo hang, trong so bang nhau):")
    for ten, v in kq["nhom"].items():
        print(f"  {ten:<30} {v['auc']:.4f}  KTC {v['ci']}  model {v['model']}")
    print("\nSo voi moc cu 0.8848 (KTC bat cap):")
    for ten, v in so_sanh.items():
        print(f"  {ten:<40} {v['chenh']:+.4f}  {v['ci']}")
    if "gop_6_cnn (chinh)" in kq["nhom"]:
        print("\nTung benh, model chinh (gop 6):")
        for k, v in sorted(kq["nhom"]["gop_6_cnn (chinh)"]["tung_benh"].items(), key=lambda kv: -kv[1]):
            print(f"  {k:<18} {v:.3f}")


if __name__ == "__main__":
    main()
