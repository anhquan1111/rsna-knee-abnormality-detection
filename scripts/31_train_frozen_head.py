"""Ngay 5: backbone dong bang -> pooling slice->study -> head 12 chieu.

Chay:
    python scripts/31_train_frozen_head.py                      # resnet18, ca 3 kieu pooling
    python scripts/31_train_frozen_head.py --backbone dinov2_vits14
    python scripts/31_train_frozen_head.py --pooling mean --epochs 120

Ra: data/interim/features_<backbone>.npz, reports/day5_*.csv, reports/day5_curves.png
"""
from __future__ import annotations

import argparse
import json
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
from rsna_knee.metrics import macro_auc, prevalence_baseline
from rsna_knee.splits import assert_no_leakage, assign_folds

IMAGES = DATA_RAW / "images"


def get_features(manifest: pd.DataFrame, backbone: str, size: int, device: str) -> dict:
    """Trich dac trung mot lan roi cache ra .npz. Lan sau doc lai trong mot nhay mat."""
    cache = DATA_INTERIM / f"features_{backbone}_{size}.npz"
    t0 = time.time()
    feats = ensure_features(manifest, IMAGES, backbone=backbone, size=size,
                            device=device, cache_path=cache)
    # Cache co the chua rat nhieu study hon so study se dung (4,407 vs 58). Dem lat theo
    # DUNG phan se dung, neu khong bao cao ra mot con so khong lien quan gi toi thi nghiem.
    from rsna_knee.config import STUDY_COL as _SC
    dung = [u for u in manifest[_SC].astype(str) if u in feats]
    n_slices = sum(feats[u].shape[0] for u in dung)
    print(f"  cache {len(feats):,} study | dung {len(dung)} study / {n_slices:,} lat "
          f"| {time.time()-t0:.1f}s | file {cache.stat().st_size/1024**2:.0f} MB")
    return feats


def plot_curves(curves: pd.DataFrame, out_path: Path) -> None:
    """Ve learning curve trung binh qua cac fold, mot cot cho moi kieu pooling.

    Thu can nhin o day khong phai duong nao thap hon, ma la khoang cach train/val doang
    ra tu epoch nao - do la luc model bat dau khop thuoc thay vi hoc.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poolings = list(dict.fromkeys(curves["pooling"]))
    fig, axes = plt.subplots(2, len(poolings), figsize=(5 * len(poolings), 7), squeeze=False)
    for j, pooling in enumerate(poolings):
        sub = curves[curves.pooling == pooling].groupby("epoch").mean(numeric_only=True)
        axes[0][j].plot(sub.index, sub["train_loss"], label="train loss")
        axes[0][j].plot(sub.index, sub["val_loss"], label="val loss")
        axes[0][j].set_title(f"pooling = {pooling}")
        axes[0][j].set_xlabel("epoch"); axes[0][j].legend(); axes[0][j].grid(alpha=.3)
        axes[1][j].plot(sub.index, sub["val_auc"], color="tab:green")
        axes[1][j].axhline(0.5, ls="--", c="gray", lw=1)
        axes[1][j].set_xlabel("epoch"); axes[1][j].set_ylabel("val macro AUC")
        axes[1][j].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_cv(feats: dict, manifest: pd.DataFrame, pooling: str, dim: int,
           epochs: int, lr: float, device: str) -> dict:
    """Cross-validation tra ve du doan out-of-fold cho TOAN BO study.

    Voi vai chuc study, AUC tinh rieng tung fold dao rat manh (moi fold chi 3-4 ca). Gop
    du doan out-of-fold roi cham MOT LAN tren toan bo la cach dung duoc o co mau nay.
    """
    folded = assign_folds(manifest, n_splits=N_FOLDS, seed=SEED)
    assert_no_leakage(folded)

    n_lab = len(LABELS)
    oof = np.full((len(folded), n_lab), np.nan, dtype=np.float32)
    curves = []

    for fold in range(N_FOLDS):
        tr = folded[folded.fold != fold]
        va = folded[folded.fold == fold]
        if len(va) == 0:
            continue
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
        pos = folded[STUDY_COL].isin(ids)
        order = {u: k for k, u in enumerate(ids)}
        idx = folded.index[pos]
        oof[idx] = logits[[order[u] for u in folded.loc[idx, STUDY_COL]]]

        for e, tlo, vlo, auc in zip(out["history"]["epoch"], out["history"]["train_loss"],
                                    out["history"]["val_loss"], out["history"]["val_auc"]):
            curves.append({"fold": fold, "epoch": e, "train_loss": tlo,
                           "val_loss": vlo, "val_auc": auc, "pooling": pooling})

    y = folded[list(LABELS)].to_numpy(dtype=float)
    keep = ~np.isnan(oof).all(axis=1)
    res = macro_auc(y[keep], oof[keep], LABELS)
    return {"result": res, "oof": oof, "folded": folded, "curves": curves, "n_scored": int(keep.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pooling", default=None, choices=["mean", "max", "attn"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extract-missing", action="store_true",
                    help="trich them dac trung o may cho study co anh nhung chua co trong cache")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, series = load_raw()
    gold = gold_subset(build_study_manifest(train, series))

    # Study dung duoc = da co dac trung (trich tren Kaggle) HOAC co du anh o may de trich.
    # Chi hoi "co anh o may khong" se bo qua gan het du lieu trich tren Kaggle.
    cache = DATA_INTERIM / f"features_{args.backbone}_{args.size}.npz"
    have = usable_studies(cache, IMAGES, extract_missing=args.extract_missing)
    gold = gold[gold[STUDY_COL].isin(have)].reset_index(drop=True)
    if len(gold) < 6:
        raise SystemExit(f"Moi co {len(gold)} study gold dung duoc - can them dac trung hoac anh")
    print(f"=== Du lieu: {len(gold)} / 58 study co nhan nguoi gan ===")
    print(f"  duong tinh moi nhan: "
          f"{ {l: int(gold[l].sum()) for l in LABELS} }")

    print("\n=== 1. Trich dac trung (backbone dong bang) ===")
    feats = get_features(gold, args.backbone, args.size, device)
    dim = next(iter(feats.values())).shape[1]
    n_slices = sum(feats[u].shape[0] for u in gold[STUDY_COL].astype(str) if u in feats)
    print(f"  {len(gold)} study dung de train | {n_slices:,} lat | {dim} chieu")

    print("\n=== 2. Moc so sanh: baseline hang so ===")
    y = gold[list(LABELS)].to_numpy(dtype=float)
    base = macro_auc(y, prevalence_baseline(y, len(y)), LABELS)
    print(f"  {base}")

    print("\n=== 3. So sanh ba kieu pooling (cung split, cung seed, cung so epoch) ===")
    poolings = [args.pooling] if args.pooling else ["mean", "max", "attn"]
    rows, all_curves, best_run = [], [], None
    for pooling in poolings:
        t0 = time.time()
        run = run_cv(feats, gold, pooling, dim, args.epochs, args.lr, device)
        dt = time.time() - t0
        res = run["result"]
        rows.append(res.to_row(pooling=pooling, seconds=round(dt, 1), n_studies=run["n_scored"]))
        all_curves.extend(run["curves"])
        print(f"  {pooling:<5} {res}  ({dt:.1f}s)")
        if best_run is None or res.macro_auc > best_run["result"].macro_auc:
            best_run, best_pooling = run, pooling

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Ten file kem backbone: chay hai backbone ma dung chung ten thi ket qua truoc bi de mat.
    tag = f"{args.backbone}_{args.size}"
    pd.DataFrame(rows).to_csv(REPORTS_DIR / f"day5_pooling_{tag}.csv", index=False)
    curves = pd.DataFrame(all_curves)
    curves.to_csv(REPORTS_DIR / f"day5_curves_{tag}.csv", index=False)
    plot_curves(curves, REPORTS_DIR / f"day5_curves_{tag}.png")
    print(f"  learning curve -> day5_curves_{tag}.png")

    print(f"\n=== 4. AUC tung nhan - pooling tot nhat ({best_pooling}) ===")
    res = best_run["result"]
    for name, v in sorted(res.per_label.items(), key=lambda kv: -kv[1]):
        n_pos = int(gold[name].sum())
        print(f"  {name:<18} {v:.3f}   ({n_pos} duong tinh / {len(gold)})")
    if res.skipped:
        print(f"  bo qua (chi mot lop): {', '.join(res.skipped)}")
    print(f"  -> trung binh {res.macro_auc:.4f} tren {res.n_scored} nhan. "
          f"Voi {len(gold)} study, khoang tin cay rat rong - xem muc 6.")

    print("\n=== 5. Luu va nap lai checkpoint - du doan co giong het khong? ===")
    torch.manual_seed(SEED)
    model = StudyHead(dim, pooling=best_pooling)
    ds = FeatureDataset(feats, gold)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_features)
    _, p1, _ = predict(model, loader, device)
    ckpt = DATA_INTERIM / f"head_{args.backbone}_{best_pooling}.pt"
    torch.save({"state_dict": model.state_dict(), "dim": dim, "pooling": best_pooling,
                "labels": list(LABELS), "backbone": args.backbone, "size": args.size}, ckpt)

    blob = torch.load(ckpt, weights_only=True)
    model2 = StudyHead(blob["dim"], pooling=blob["pooling"])
    model2.load_state_dict(blob["state_dict"])
    _, p2, _ = predict(model2, loader, device)
    print(f"  lech lon nhat giua hai lan du doan: {np.abs(p1-p2).max():.2e}")
    print(f"  thu tu nhan trong checkpoint khop config: {blob['labels'] == list(LABELS)}")
    print(f"  ghi {ckpt.name} ({ckpt.stat().st_size/1024:.0f} KB)")

    print("\n=== 6. Do bat dinh: bootstrap theo study ===")
    y_all = best_run["folded"][list(LABELS)].to_numpy(dtype=float)
    oof = best_run["oof"]
    keep = ~np.isnan(oof).all(axis=1)
    yk, pk = y_all[keep], oof[keep]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(300):
        idx = rng.integers(0, len(yk), len(yk))
        try:
            boots.append(macro_auc(yk[idx], pk[idx], LABELS).macro_auc)
        except ValueError:
            continue
    boots = np.array(boots)
    lo, hi = np.percentile(boots, (2.5, 97.5))
    print(f"  macro AUC {res.macro_auc:.3f}, khoang tin cay 95% [{lo:.3f}, {hi:.3f}] "
          f"(bootstrap {len(boots)} lan theo STUDY)")
    print(f"  -> be rong {hi-lo:.3f}. Chenh lech nho hon the nay giua hai cau hinh "
          f"khong ket luan duoc gi.")

    summary = {
        "n_studies": len(gold), "n_slices": int(n_slices), "backbone": args.backbone,
        "dim": dim, "best_pooling": best_pooling, "macro_auc": res.macro_auc,
        "ci95": [float(lo), float(hi)], "baseline_auc": base.macro_auc,
        "epochs": args.epochs, "device": device,
    }
    (REPORTS_DIR / f"day5_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDa ghi day5_summary_{tag}.json")


if __name__ == "__main__":
    main()
