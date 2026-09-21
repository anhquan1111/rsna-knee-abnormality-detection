"""Ngay 4: chia fold chong ro ri va do hau qua cua cach chia sai.

Chay:  python scripts/11_build_splits.py
Ra:    data/interim/splits_gold.csv  (commit file nay - split phai co dinh giua cac lan chay)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rsna_knee.config import DATA_INTERIM, LABELS, N_FOLDS, SEED, STUDY_COL
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.metrics import macro_auc, prevalence_baseline
from rsna_knee.splits import (
    assert_no_leakage,
    assign_folds,
    build_groups,
    measure_series_split_leakage,
    normalize_report,
)


def main() -> None:
    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold = gold_subset(man)

    print("=== 1. Chia ngau nhien o CAP SERIES thi ro ri bao nhieu? ===")
    gold_series = series[series[STUDY_COL].isin(gold[STUDY_COL])]
    r = measure_series_split_leakage(gold_series, seed=SEED)
    print(f"  {len(gold_series)} series cua 58 study -> val {r['n_series_val']} series")
    print(f"  val chua {r['n_study_val']} study, trong do {r['n_study_leaked']} study "
          f"cung xuat hien o train  =>  ro ri {r['leak_rate']:.1%}")

    print("\n=== 2. Bao cao trung lap ===")
    norm = train["Report"].map(normalize_report)
    exact = int(train["Report"].duplicated(keep=False).sum())
    normed = int(norm.duplicated(keep=False).sum())
    print(f"  trung y nguyen           : {exact} dong")
    print(f"  trung sau khi chuan hoa  : {normed} dong")
    print(f"  nhom trung lon nhat      : {int(norm.value_counts().iloc[0])} study")
    n_gold_dup = int(norm[train[STUDY_COL].isin(gold[STUDY_COL])].duplicated(keep=False).sum())
    print(f"  trong 58 gold co trung   : {n_gold_dup}")

    print("\n=== 3. Chia fold dung: StratifiedGroupKFold tren 58 gold ===")
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    print(f"  kiem ro ri moi fold: {assert_no_leakage(folded)}  (0 la dat)")
    print(f"  nhom sau khi gop bao cao trung: {build_groups(folded).nunique()} nhom / {len(folded)} study")

    counts = folded.groupby("fold").size()
    print(f"  so study moi fold: {counts.to_dict()}")

    print("\n  Nhan bi mat (chi mot lop) trong tung fold val:")
    total_broken = 0
    for f in range(N_FOLDS):
        va = folded[folded.fold == f]
        broken = [l for l in LABELS if va[l].nunique() < 2]
        total_broken += len(broken)
        print(f"    fold {f} (n={len(va)}): {len(broken)} nhan -> {', '.join(broken) if broken else '-'}")
    print(f"  TONG: {total_broken} cap (fold, nhan) khong cham duoc")

    print("\n=== 3b. Stratify co that su giup khong? So voi KFold thuong ===")
    from sklearn.model_selection import KFold
    for k in (5, 10):
        for name, splitter in (("KFold       ", KFold(n_splits=k, shuffle=True, random_state=SEED)),
                               ("StratGroupKF", None)):
            if splitter is None:
                f2 = assign_folds(gold, n_splits=k, seed=SEED)
            else:
                f2 = gold.copy().reset_index(drop=True)
                f2["fold"] = -1
                for fi, (_, va) in enumerate(splitter.split(f2)):
                    f2.loc[va, "fold"] = fi
            broken = sum(
                sum(1 for l in LABELS if f2[f2.fold == fi][l].nunique() < 2)
                for fi in range(k)
            )
            print(f"  k={k:>2} {name}: {broken:>2} cap (fold, nhan) khong cham duoc")
    print("  -> stratify tren nhan hiem nhat (MCL) la thu giu cho macro AUC co du mau so")

    print("\n=== 4. roc_auc_score lam gi khi nhan chi co mot lop? ===")
    from sklearn.metrics import roc_auc_score
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = roc_auc_score([0, 0, 0, 0], [0.1, 0.9, 0.3, 0.7])
    print(f"  roc_auc_score(y toan 0) = {v}  (tra ve gia tri, KHONG nem ValueError)")
    print(f"  np.mean([0.9, nan]) = {np.mean([0.9, v])}   <- ca macro thanh nan")
    print(f"  np.nanmean([0.9, nan]) = {np.nanmean([0.9, v])}  <- ra so dep, am tham bo nhan")

    print("\n=== 5. Baseline hang so: AUC vs accuracy ===")
    y = gold[list(LABELS)].to_numpy(dtype=float)
    pred = prevalence_baseline(y, len(y))
    res = macro_auc(y, pred, LABELS)
    acc = float(((pred > 0.5).astype(float) == y).mean())
    print(f"  {res}")
    print(f"  accuracy cua chinh baseline do: {acc:.1%}")
    print(f"  -> accuracy {acc:.1%} nghe rat kha trong khi AUC dung bang 0.5. "
          f"Day la ly do khong bao cao accuracy.")

    print("\n=== 6. Macro AUC dao bao nhieu khi chi doi seed chia tap? ===")
    # Mo phong mot model co suc phan biet CO DINH: diem = nhieu manh + mot phan tin hieu.
    # Nhieu phai lan at tin hieu, neu khong AUC bao hoa o 1.0 va phep do mat y nghia.
    rng = np.random.default_rng(SEED)
    scores = []
    for s in range(30):
        f = assign_folds(gold, n_splits=N_FOLDS, seed=s)
        va = f[f.fold == 0]
        yv = va[list(LABELS)].to_numpy(dtype=float)
        fake = rng.normal(0, 1.0, yv.shape) + yv * 0.8
        try:
            scores.append(macro_auc(yv, fake, LABELS).macro_auc)
        except ValueError:
            pass
    scores = np.array(scores)
    print(f"  30 seed, cung mot model: min {scores.min():.3f} | max {scores.max():.3f} "
          f"| std {scores.std():.3f}")
    print(f"  -> chenh lech nho hon ~{2*scores.std():.2f} giua hai cau hinh la nhieu, khong phai cai tien")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out = DATA_INTERIM / "splits_gold.csv"
    folded[[STUDY_COL, "fold"]].to_csv(out, index=False)
    print(f"\nDa ghi {out}")


if __name__ == "__main__":
    main()
