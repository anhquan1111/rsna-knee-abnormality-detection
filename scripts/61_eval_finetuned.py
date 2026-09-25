"""NGAY 9 - Cham dac trung tu backbone da fine-tune, doi chieu voi backbone dong bang.

Chay sau khi tai ket qua tu Kaggle (scripts/60_kaggle_finetune.py) ve:

    python scripts/61_eval_finetuned.py data/kaggle_out_ft

Giao thuc: giong HET ngay 5. Cung fold, cung seed, cung so epoch, cung pooling, cung
tap danh gia (58 ca nhan bac si). Doi dung MOT thu: dac trung den tu backbone dong bang
hay backbone da fine-tune. Neu doi hai thu thi chenh lech do duoc khong quy ve dau duoc.

Bootstrap o day la BOOTSTRAP BAT CAP: moi lan lay mau lai, CA HAI cau hinh duoc cham
tren cung tap ca duoc chon. Cach nay tru bot phuong sai do "gap tap ca de/kho", nen do
duoc chenh lech chinh xac hon nhieu so voi viec so hai khoang tin cay roi nhau - hai
khoang chong lan nhau VAN co the ung voi mot chenh lech chac chan khac 0.
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

from rsna_knee.config import DATA_INTERIM, LABELS, N_FOLDS, REPORTS_DIR, SEED, STUDY_COL
from rsna_knee.dataset import FeatureDataset
from rsna_knee.head import StudyHead, collate_features, predict, train_head
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.metrics import macro_auc
from rsna_knee.splits import assert_no_leakage, assign_folds


def chay_5fold(feats: dict, gold: pd.DataFrame, epochs: int, lr: float,
               device: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Tra ve (du doan out-of-fold, bang gold da gan fold) - giong het ngay 5."""
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    assert_no_leakage(folded)
    oof = np.full((len(folded), len(LABELS)), np.nan, dtype=np.float32)

    for fold in range(N_FOLDS):
        va = folded[folded.fold == fold]
        tr = folded[folded.fold != fold]
        tr_ds, va_ds = FeatureDataset(feats, tr), FeatureDataset(feats, va)
        if len(tr_ds) == 0 or len(va_ds) == 0:
            continue
        torch.manual_seed(SEED + fold)
        model = StudyHead(next(iter(feats.values())).shape[1], pooling="mean")
        tl = DataLoader(tr_ds, batch_size=4, shuffle=True, collate_fn=collate_features)
        vl = DataLoader(va_ds, batch_size=4, shuffle=False, collate_fn=collate_features)
        out = train_head(model, tl, vl, epochs=epochs, lr=lr, device=device, log_every=0)
        model.load_state_dict(out["best"]["state"])
        _, logits, ids = predict(model, vl, device)

        order = {u: k for k, u in enumerate(ids)}
        idx = folded.index[folded[STUDY_COL].isin(ids)]
        oof[idx] = logits[[order[u] for u in folded.loc[idx, STUDY_COL]]]
    return oof, folded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", help="thu muc chua features_finetuned_*.npz tai tu Kaggle")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser()
    ft_files = sorted(src.glob("features_finetuned_*.npz"))
    if not ft_files:
        raise SystemExit(f"Khong thay features_finetuned_*.npz trong {src}\n"
                         f"Co: {sorted(p.name for p in src.glob('*'))[:20]}")
    ft_path = ft_files[0]

    tt = src / "finetune_summary.json"
    if tt.exists():
        d = json.loads(tt.read_text())
        print("=== Tom tat tu Kaggle ===")
        for k in ("stage", "n_study", "n_gold", "n_slice", "n_hong", "val_auc_weak",
                  "unfreeze_blocks", "n_train_studies", "epochs", "phut_tong"):
            if k in d:
                print(f"  {k:<16} {d[k]}")
        if d.get("lich_su"):
            print("  qua trinh train:")
            for h in d["lich_su"]:
                print(f"    epoch {h['epoch']:>2} | loss {h['loss']:.4f} | "
                      f"val AUC (nhan may) {h['val_auc']:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, series = load_raw()
    gold_all = gold_subset(build_study_manifest(train, series))

    with np.load(ft_path) as z:
        ft = {k: z[k] for k in z.files}
    dong_bang_path = DATA_INTERIM / f"features_dinov2_vits14_{args.size}.npz"
    if not dong_bang_path.exists():
        raise SystemExit(f"Khong thay {dong_bang_path} - can no lam moc so sanh.")
    with np.load(dong_bang_path) as z:
        db = {k: z[k] for k in z.files if k in ft}

    chung = sorted(set(ft) & set(db) & set(gold_all[STUDY_COL]))
    gold = gold_all[gold_all[STUDY_COL].isin(chung)].reset_index(drop=True)
    print(f"\n=== Du lieu ===")
    print(f"  ca gold cham duoc ca hai ben: {len(gold)}/58")
    if len(gold) < 58:
        print("  CANH BAO: thieu ca gold -> so sanh van hop le nhung KTC rong hon.")
    n_ft = sum(ft[u].shape[0] for u in chung)
    n_db = sum(db[u].shape[0] for u in chung)
    print(f"  so lat: fine-tune {n_ft:,} | dong bang {n_db:,}"
          + ("  <- KHAC NHAU, phai xem lai tien xu ly" if n_ft != n_db else "  <- khop"))

    print(f"\n=== Chay 5-fold, {args.epochs} epoch, pooling mean ===")
    ket = {}
    for ten, f in (("dong bang (ngay 5/8)", db), ("fine-tune (ngay 9)", ft)):
        t0 = time.time()
        oof, folded = chay_5fold(f, gold, args.epochs, args.lr, device)
        y = folded[list(LABELS)].to_numpy(dtype=float)
        keep = ~np.isnan(oof).all(axis=1)
        res = macro_auc(y[keep], oof[keep], LABELS)
        ket[ten] = {"oof": oof, "keep": keep, "y": y, "res": res}
        print(f"  {ten:<22} macro AUC {res.macro_auc:.4f}  ({time.time()-t0:.0f}s)")

    a = ket["dong bang (ngay 5/8)"]
    b = ket["fine-tune (ngay 9)"]
    print("\n=== Chenh lech tung nhan ===")
    print(f"  {'nhan':<18} {'dong bang':>10} {'fine-tune':>10} {'chenh':>8}")
    for lab in LABELS:
        x, y2 = a["res"].per_label.get(lab), b["res"].per_label.get(lab)
        if x is None or y2 is None:
            continue
        print(f"  {lab:<18} {x:>10.3f} {y2:>10.3f} {y2-x:>+8.3f}")

    print("\n=== Chenh lech co vuot duoc nhieu khong? ===")
    keep = a["keep"] & b["keep"]
    yt = a["y"][keep]
    pa, pb = a["oof"][keep], b["oof"][keep]
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(args.boot):
        idx = rng.integers(0, len(yt), len(yt))
        try:
            diffs.append(macro_auc(yt[idx], pb[idx], LABELS).macro_auc
                         - macro_auc(yt[idx], pa[idx], LABELS).macro_auc)
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, (2.5, 97.5))
    obs = b["res"].macro_auc - a["res"].macro_auc
    print(f"  fine-tune - dong bang : {obs:+.4f}")
    print(f"  khoang tin cay 95%    : [{lo:+.4f}, {hi:+.4f}] "
          f"(bootstrap bat cap {len(diffs)} lan theo ca)")
    if lo < 0 < hi:
        ket_luan = "CHUA ket luan duoc"
        print(f"  -> {ket_luan}: khoang tin cay trum qua 0.")
        print(f"     Voi {int(keep.sum())} ca danh gia, chenh lech phai lon hon "
              f"~{max(abs(lo), abs(hi)):.3f} moi do duoc.")
    else:
        ket_luan = "fine-tune GIUP" if lo > 0 else "fine-tune HAI"
        print(f"  -> {ket_luan}: khoang tin cay khong chua 0.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ra = REPORTS_DIR / "day9_finetune.csv"
    pd.DataFrame([
        a["res"].to_row(config="dong_bang", n_gold=len(gold)),
        b["res"].to_row(config="fine_tune", n_gold=len(gold)),
    ]).to_csv(ra, index=False)
    (REPORTS_DIR / "day9_finetune.json").write_text(json.dumps({
        "dong_bang": a["res"].macro_auc, "fine_tune": b["res"].macro_auc,
        "chenh": obs, "ci95": [float(lo), float(hi)], "ket_luan": ket_luan,
        "n_gold": len(gold), "epochs": args.epochs, "boot": len(diffs),
    }, indent=2))
    print(f"\nDa ghi {ra.name} va day9_finetune.json")


if __name__ == "__main__":
    main()
