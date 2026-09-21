"""Ngay 8: nhan may giup hay hai? Thi nghiem co kiem soat.

Giao thuc - ba dieu phai giu nguyen giua cac cau hinh, neu khong so sanh vo nghia:
  1. **Tap danh gia luon la study co nhan NGUOI GAN.** Nhan may khong duoc dung lam thuoc do
     chinh no. Study co nhan may chi duoc them vao phia TRAIN.
  2. **Cung split, cung seed, cung so epoch, cung backbone.** Chi doi dung mot thu: nhan.
  3. **Cung evaluator.** macro AUC voi mau so ghi ro.

Chay:
    python scripts/40_experiment_weak_labels.py --backbone dinov2_vits14 --epochs 80
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.config import (
    DATA_INTERIM,
    DATA_RAW,
    LABELS,
    N_FOLDS,
    REPORTS_DIR,
    SEED,
    STUDY_COL,
)
from rsna_knee.dataset import FeatureDataset, StudyDataset
from rsna_knee.features import ensure_features, usable_studies
from rsna_knee.head import StudyHead, collate_features, predict, train_head
from rsna_knee.manifest import (
    build_study_manifest,
    complete_local_studies,
    gold_subset,
    load_raw,
)
from rsna_knee.metrics import macro_auc
from rsna_knee.splits import assert_no_leakage, assign_folds

IMAGES = DATA_RAW / "images"


def available_studies(cache, extract_missing: bool) -> set[str]:
    """Study dung duoc: da co dac trung, hoac co du anh o may de trich.

    KHONG chi hoi "co anh o may khong": dac trung duoc trich tren Kaggle nen hang nghin
    study co dac trung ma khong co file .dcm nao o day. Hoi nham cau se lam thi nghiem
    chay tren mot phan nho du lieu ma khong co gi bao.
    """
    return usable_studies(cache, IMAGES, extract_missing=extract_missing)


def run_config(name: str, train_pool: pd.DataFrame, gold: pd.DataFrame, feats: dict,
               dim: int, epochs: int, lr: float, device: str) -> dict:
    """train_pool: tat ca study duoc phep vao TRAIN. Val luon lay tu gold, theo fold co dinh."""
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    assert_no_leakage(folded)
    oof = np.full((len(folded), len(LABELS)), np.nan, dtype=np.float32)

    for fold in range(N_FOLDS):
        va = folded[folded.fold == fold]
        va_ids = set(va[STUDY_COL])
        tr = train_pool[~train_pool[STUDY_COL].isin(va_ids)]   # khong bao gio de val lot vao train

        tr_ds, va_ds = FeatureDataset(feats, tr), FeatureDataset(feats, va)
        if len(tr_ds) == 0 or len(va_ds) == 0:
            continue
        torch.manual_seed(SEED + fold)
        model = StudyHead(dim, pooling="mean")
        tl = DataLoader(tr_ds, batch_size=4, shuffle=True, collate_fn=collate_features)
        vl = DataLoader(va_ds, batch_size=4, shuffle=False, collate_fn=collate_features)
        out = train_head(model, tl, vl, epochs=epochs, lr=lr, device=device, log_every=0)
        model.load_state_dict(out["best"]["state"])
        _, logits, ids = predict(model, vl, device)

        order = {u: k for k, u in enumerate(ids)}
        idx = folded.index[folded[STUDY_COL].isin(ids)]
        oof[idx] = logits[[order[u] for u in folded.loc[idx, STUDY_COL]]]

    y = folded[list(LABELS)].to_numpy(dtype=float)
    keep = ~np.isnan(oof).all(axis=1)
    res = macro_auc(y[keep], oof[keep], LABELS)
    return {"config": name, "result": res, "n_train": len(train_pool), "oof": oof, "keep": keep}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vits14", choices=["resnet18", "dinov2_vits14"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extract-missing", action="store_true",
                    help="trich them dac trung o may cho study co anh nhung chua co trong cache")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold_all = gold_subset(man)

    cache = DATA_INTERIM / f"features_{args.backbone}_{args.size}.npz"
    have = available_studies(cache, args.extract_missing)
    gold = gold_all[gold_all[STUDY_COL].isin(have)].reset_index(drop=True)
    extra = man[(~man[STUDY_COL].isin(gold_all[STUDY_COL]))
                & (man[STUDY_COL].isin(have))].reset_index(drop=True)

    weak_path = DATA_INTERIM / "weak_labels.csv"
    if not weak_path.exists():
        raise SystemExit("Chua co weak_labels.csv. Chay scripts/20_extract_weak_labels.py truoc.")
    weak = pd.read_csv(weak_path)

    print(f"=== Du lieu co anh ===")
    print(f"  study co nhan nguoi gan : {len(gold)}")
    print(f"  study chi co nhan may   : {len(extra)}")
    if len(extra) == 0:
        print("\n  CHUA co study nao ngoai gold -> khong chay duoc thi nghiem nay.")
        print("  Tai them: python scripts/04_download_from_listing.py --plane Sagittal --max-studies 40")
        return

    print("\n=== Trich dac trung ===")
    feats = ensure_features(pd.concat([gold, extra]), IMAGES, backbone=args.backbone,
                            size=args.size, device=device, cache_path=cache)
    dim = next(iter(feats.values())).shape[1]

    # Ghep nhan may vao cac study ngoai gold. Nhan `NaN` = chua ket luan duoc -> loss bo qua.
    weak_idx = weak.set_index(STUDY_COL)
    extra_weak = extra.copy()
    for label in LABELS:
        extra_weak[label] = extra_weak[STUDY_COL].map(weak_idx[label])

    extra_zero = extra_weak.copy()
    extra_zero[list(LABELS)] = extra_zero[list(LABELS)].fillna(0.0)

    known = extra_weak[list(LABELS)].notna().to_numpy()
    print(f"  nhan may: {known.mean():.1%} cap (study, nhan) ket luan duoc, "
          f"{int((extra_weak[list(LABELS)] == 1).sum().sum())} duong tinh")

    configs = [
        ("A_gold_only", gold),
        ("B_gold+weak", pd.concat([gold, extra_weak], ignore_index=True)),
        ("C_gold+weak_unknown_as_0", pd.concat([gold, extra_zero], ignore_index=True)),
    ]

    print("\n=== Ket qua (danh gia LUON tren study co nhan nguoi gan) ===")
    rows, results = [], {}
    for name, pool in configs:
        t0 = time.time()
        out = run_config(name, pool, gold, feats, dim, args.epochs, args.lr, device)
        dt = time.time() - t0
        results[name] = out
        rows.append(out["result"].to_row(config=name, n_train=out["n_train"], seconds=round(dt, 1)))
        print(f"  {name:<26} n_train={out['n_train']:>3}  {out['result']}  ({dt:.0f}s)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "day8_experiment.csv", index=False)

    base = results["A_gold_only"]["result"]
    print("\n=== Chenh lech so voi A, theo tung nhan ===")
    print(f"  {'nhan':<18} {'A':>7} {'B':>7} {'B-A':>7} {'C':>7} {'C-A':>7}")
    for label in LABELS:
        a = base.per_label.get(label)
        b = results["B_gold+weak"]["result"].per_label.get(label)
        c = results["C_gold+weak_unknown_as_0"]["result"].per_label.get(label)
        if a is None or b is None or c is None:
            continue
        print(f"  {label:<18} {a:>7.3f} {b:>7.3f} {b-a:>+7.3f} {c:>7.3f} {c-a:>+7.3f}")

    print("\n=== Chenh lech nay co vuot duoc nhieu khong? ===")
    y = results["A_gold_only"]["oof"]
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    y_true = folded[list(LABELS)].to_numpy(dtype=float)
    keep = results["A_gold_only"]["keep"]
    rng = np.random.default_rng(SEED)
    diffs = []
    pb = results["B_gold+weak"]["oof"]
    for _ in range(300):
        idx = rng.integers(0, int(keep.sum()), int(keep.sum()))
        try:
            da = macro_auc(y_true[keep][idx], y[keep][idx], LABELS).macro_auc
            db = macro_auc(y_true[keep][idx], pb[keep][idx], LABELS).macro_auc
            diffs.append(db - da)
        except ValueError:
            continue
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, (2.5, 97.5))
    obs = results["B_gold+weak"]["result"].macro_auc - base.macro_auc
    print(f"  B - A quan sat duoc : {obs:+.4f}")
    print(f"  khoang tin cay 95%  : [{lo:+.4f}, {hi:+.4f}] (bootstrap theo study)")
    verdict = "KHONG ket luan duoc" if lo < 0 < hi else ("nhan may GIUP" if lo > 0 else "nhan may HAI")
    print(f"  -> {verdict}")

    # Cau ket luan phai theo SO DO, khong duoc viet cung. Bang mot cau "chua du du lieu"
    # co dinh se sai ngay khi du lieu du - va do la luc nguoi doc can cau tra loi nhat.
    n_eval = int(keep.sum())
    n_weak = results["B_gold+weak"]["n_train"] - results["A_gold_only"]["n_train"]
    if lo < 0 < hi:
        print("  Khoang tin cay trum qua 0 nen chenh lech quan sat duoc CHUA la bang chung.")
        print(f"  Dang co {n_eval} study danh gia va {n_weak:,} study nhan may - can them ca hai.")
    else:
        print("  Khoang tin cay KHONG chua 0 -> day la bang chung that theo huong tren.")
        print(f"  Dua tren {n_eval} study danh gia (nhan nguoi) va {n_weak:,} study nhan may.")
        print(f"  Gioi han con lai: tap danh gia toi da chi 58 study, nen be rong khoang")
        print(f"  tin cay ({hi - lo:.3f}) chi hep lai duoc bang cach co them nhan NGUOI,")
        print(f"  them nhan may khong giup gi cho phan nay.")


if __name__ == "__main__":
    main()
