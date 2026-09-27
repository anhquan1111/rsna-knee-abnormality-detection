"""Mot head chung cho ca ba mat phang, hay ba head rieng roi gop?

    python scripts/43_head_tung_mat_phang.py --epochs 80

Giao thuc y het ngay 8/10: cung 5 fold, cung seed, cung so epoch, cung nguon nhan (LLM),
cung 58 ca danh gia. Doi dung mot thu: cach dung ba mat phang.

HAI CACH, VA GIA THUYET DANG SAU
================================
  CHUNG  (dang dung)  Noi lat cua ca ba mat phang thanh mot chong, mot head doc het.
                      Gia thuyet: benh hien o mat phang nao cung duoc, cu de attention
                      tu tim.
  RIENG  (thu o day)  Ba head doc lap, moi head chi thay mot mat phang, roi GOP du doan.
                      Gia thuyet: moi mat phang co cach nhin rieng, va ep chung vao mot
                      head buoc no phai hoc mot ham dung chung cho ba loai anh khac nhau.

Vi sao dang thu: ngay 9 do duoc rang gop hai model SAI O CHO KHAC NHAU thi co lai
(+0.023), trong khi gop hai model nhin cung mot bo dac trung thi khong (0.7688 so voi
0.7698). Ba mat phang la ba goc nhin that su khac nhau, nen day la truong hop dau chu
khong phai truong hop sau.

GOP THEO HANG, khong theo gia tri: ba head train rieng nen thang do logit khong chung
goc. AUC von chi quan tam thu hang.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from rsna_knee.config import (
    DATA_INTERIM,
    LABELS,
    N_FOLDS,
    REPORT_COL,
    REPORTS_DIR,
    SEED,
    STUDY_COL,
)
from rsna_knee.dataset import FeatureDataset
from rsna_knee.head import StudyHead, collate_features, predict, train_head
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.metrics import macro_auc
from rsna_knee.splits import assert_no_leakage, assign_folds

L = list(LABELS)
REPORT_KEY = "report_key"
REPO = Path(__file__).resolve().parents[1]
LLM = REPO / "data" / "external" / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv"
NGUON = {
    "sagittal": DATA_INTERIM / "features_dinov2_vits14_224.npz.bak",
    "axial": REPO / "data" / "kaggle_out" / "features_dinov2_axial.npz",
    "coronal": REPO / "data" / "kaggle_out" / "features_dinov2_coronal.npz",
}


def chay_5fold(feats: dict, pool: pd.DataFrame, gold: pd.DataFrame, dim: int,
               epochs: int, lr: float, device: str, pooling: str):
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    assert_no_leakage(folded)
    oof = np.full((len(folded), len(L)), np.nan, dtype=np.float32)

    for fold in range(N_FOLDS):
        va = folded[folded.fold == fold]
        tr = pool[~pool[STUDY_COL].isin(set(va[STUDY_COL]))]
        # Chan duong ro ri cua ngay 8: nhan may suy tu bao cao nen ca co bao cao trung
        # tung ky tu voi ca val mang dap an vao train duoi UID khac.
        tr = tr[~tr[REPORT_KEY].isin(set(va[REPORT_KEY]))]

        tr_ds, va_ds = FeatureDataset(feats, tr), FeatureDataset(feats, va)
        if len(tr_ds) == 0 or len(va_ds) == 0:
            continue
        torch.manual_seed(SEED + fold)
        model = StudyHead(dim, pooling=pooling)
        tl = DataLoader(tr_ds, batch_size=4, shuffle=True, collate_fn=collate_features)
        vl = DataLoader(va_ds, batch_size=4, shuffle=False, collate_fn=collate_features)
        out = train_head(model, tl, vl, epochs=epochs, lr=lr, device=device, log_every=0)
        model.load_state_dict(out["best"]["state"])
        _, logits, ids = predict(model, vl, device)
        order = {u: k for k, u in enumerate(ids)}
        idx = folded.index[folded[STUDY_COL].isin(ids)]
        oof[idx] = logits[[order[u] for u in folded.loc[idx, STUDY_COL]]]
    return oof, folded


def theo_hang(p: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    for j in range(p.shape[1]):
        out[:, j] = rankdata(p[:, j]) / len(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pooling", default="per_label_attn")
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, series = load_raw()
    man = build_study_manifest(train, series)
    man[REPORT_KEY] = (man[REPORT_COL].astype(str).str.strip().str.lower()
                       .map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()))
    gold_all = gold_subset(man)

    thieu = [k for k, v in NGUON.items() if not v.exists()]
    if thieu:
        raise SystemExit(f"Thieu dac trung cua mat phang {thieu}")
    llm = pd.read_csv(LLM, dtype={STUDY_COL: str}).set_index(STUDY_COL)

    print("=== Dac trung tung mat phang ===")
    fp = {}
    for ten, path in NGUON.items():
        with np.load(path) as z:
            fp[ten] = {k: z[k] for k in z.files}
        n_lat = sum(v.shape[0] for v in fp[ten].values())
        print(f"  {ten:<10} {len(fp[ten]):,} ca | {n_lat:,} lat")

    chung = set.intersection(*(set(v) for v in fp.values()))
    gold = gold_all[gold_all[STUDY_COL].isin(chung)].reset_index(drop=True)
    extra = man[(~man[STUDY_COL].isin(gold_all[STUDY_COL]))
                & (man[STUDY_COL].isin(chung))].reset_index(drop=True)
    for lab in L:
        extra[lab] = extra[STUDY_COL].map(llm[lab])
    pool = pd.concat([gold, extra], ignore_index=True)
    dim = next(iter(fp["sagittal"].values())).shape[1]
    print(f"  gold {len(gold)} | ngoai gold {len(extra):,} | {dim} chieu")

    print(f"\n=== Ba head RIENG (pooling {args.pooling}) ===")
    oof_mp, folded = {}, None
    for ten in NGUON:
        t0 = time.time()
        o, folded = chay_5fold(fp[ten], pool, gold, dim, args.epochs, args.lr,
                               device, args.pooling)
        oof_mp[ten] = o
        keep = ~np.isnan(o).all(axis=1)
        y = folded[L].to_numpy(dtype=float)
        print(f"  {ten:<10} macro AUC {macro_auc(y[keep], o[keep], L).macro_auc:.4f}  "
              f"({time.time()-t0:.0f}s)")

    y = folded[L].to_numpy(dtype=float)
    keep = np.logical_and.reduce([~np.isnan(o).all(axis=1) for o in oof_mp.values()])
    gop = sum(theo_hang(o[keep]) for o in oof_mp.values()) / len(oof_mp)
    auc_gop = macro_auc(y[keep], gop, L).macro_auc
    print(f"\n  GOP ba head rieng (theo hang)  macro AUC {auc_gop:.4f}")

    # Moc so sanh: head CHUNG tren dac trung ba mat phang noi lat (da co tu scripts/41)
    moc_path = REPORTS_DIR / "day10b_oof.npz"
    if not moc_path.exists():
        print(f"\n  khong thay {moc_path.name} -> bo qua phep so voi head chung")
        auc_chung, lo, hi = float("nan"), float("nan"), float("nan")
    else:
        with np.load(moc_path) as z:
            chung_oof = z["D_llm"]
        k2 = keep & ~np.isnan(chung_oof).all(axis=1)
        auc_chung = macro_auc(y[k2], chung_oof[k2], L).macro_auc
        gop2 = sum(theo_hang(o[k2]) for o in oof_mp.values()) / len(oof_mp)
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(args.boot):
            i = rng.integers(0, int(k2.sum()), int(k2.sum()))
            try:
                d.append(macro_auc(y[k2][i], gop2[i], L).macro_auc
                         - macro_auc(y[k2][i], chung_oof[k2][i], L).macro_auc)
            except ValueError:
                continue
        lo, hi = np.percentile(d, (2.5, 97.5))
        obs = macro_auc(y[k2], gop2, L).macro_auc - auc_chung
        print(f"\n=== RIENG+gop co hon CHUNG khong? ===")
        print(f"  head CHUNG (noi lat ba mat phang) : {auc_chung:.4f}")
        print(f"  GOP ba head RIENG                 : {macro_auc(y[k2], gop2, L).macro_auc:.4f}")
        print(f"  chenh {obs:+.4f}  KTC 95% [{lo:+.4f}, {hi:+.4f}] (bat cap, {len(d)} lan)")
        print("  -> " + ("CHUA ket luan duoc" if lo < 0 < hi
                         else ("RIENG hon" if lo > 0 else "CHUNG hon")))

        # Gop TAT CA: ba head rieng + head chung. Bon goc nhin thay vi ba.
        tat_ca = (sum(theo_hang(o[k2]) for o in oof_mp.values())
                  + theo_hang(chung_oof[k2])) / (len(oof_mp) + 1)
        print(f"\n  GOP CA BON (ba rieng + mot chung): "
              f"{macro_auc(y[k2], tat_ca, L).macro_auc:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPORTS_DIR / "day11_mat_phang_oof.npz", y=y, **oof_mp)
    (REPORTS_DIR / "day11_mat_phang.json").write_text(json.dumps({
        "pooling": args.pooling, "epochs": args.epochs,
        "tung_mat_phang": {k: macro_auc(y[~np.isnan(v).all(axis=1)],
                                        v[~np.isnan(v).all(axis=1)], L).macro_auc
                           for k, v in oof_mp.items()},
        "gop_ba_head_rieng": auc_gop, "head_chung": auc_chung,
        "ci95_rieng_tru_chung": [float(lo), float(hi)],
    }, indent=2))
    print("\nDa ghi day11_mat_phang.json va day11_mat_phang_oof.npz")


if __name__ == "__main__":
    main()
