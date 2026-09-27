"""Chay lai Y HET mot cau hinh voi nhieu seed - lech bao nhieu?

    python scripts/44_phuong_sai_seed.py --seeds 5 --epochs 80

VI SAO DAY LA PHEP DO QUAN TRONG NHAT CUA CA DU AN
==================================================
Suot 11 ngay, moi ket luan deu dua tren CHENH LECH giua cac cau hinh:

    pooling max hon mean      +0.019   (ngay 5)
    3 mat phang hon 1         +0.061   (vong 3)
    nhan LLM hon tu dien      +0.054   (ngay 10)
    per_label_attn hon mean   +0.023   (ngay 10)
    gop 4 model hon head chung +0.036  (ngay 11)

Nhung chua bao gio hoi: **chay lai y het, chi doi seed, thi lech bao nhieu?**

Neu rieng seed da cho +-0.03 thi mot nua bang tren nam gon trong nhieu - chung khong noi
model tot hon, chung noi ta tung xuc xac mot lan. Neu seed chi cho +-0.005 thi ca bang
dung vung.

Day la cay thuoc, KHONG phai cai nut de van.

KHONG DUOC CHON SEED TOT NHAT
=============================
Chon seed co diem cao nhat la chon tren TAP DANH GIA. Diem CV se dep len va LB thi khong,
vi cai dep do chi la nhat dung lan tung may. Script nay tinh san con so "neu nhat seed tot
nhat thi diem phong len bao nhieu" de thay ro cai gia cua viec do.

Hai cach dung HOP LE:
  1. Lam thuoc do doc lai cac so cu.
  2. TRUNG BINH du doan qua cac seed. Trung binh thi khong nhin dap an luc nao ca, khac
     han voi chon. Day la gop model, dung thu ngay 9 do duoc la co lai.

CO DINH FOLD, CHI DOI SEED CUA MODEL
====================================
`assign_folds` giu nguyen SEED, chi `torch.manual_seed` doi. Neu doi ca fold thi moi lan
chay lai cham tren mot tap danh gia khac, va con so thu duoc tron hai nguon dao dong -
khong tra loi duoc cau dang hoi la "cung du lieu, cung split, chay lai thi lech bao nhieu".
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


def mot_lan(seed_model: int, feats: dict, pool: pd.DataFrame, gold: pd.DataFrame,
            dim: int, epochs: int, lr: float, device: str, pooling: str):
    # Fold GIU NGUYEN o SEED goc - chi doi seed cua model.
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    assert_no_leakage(folded)
    oof = np.full((len(folded), len(L)), np.nan, dtype=np.float32)

    for fold in range(N_FOLDS):
        va = folded[folded.fold == fold]
        tr = pool[~pool[STUDY_COL].isin(set(va[STUDY_COL]))]
        tr = tr[~tr[REPORT_KEY].isin(set(va[REPORT_KEY]))]
        tr_ds, va_ds = FeatureDataset(feats, tr), FeatureDataset(feats, va)
        if len(tr_ds) == 0 or len(va_ds) == 0:
            continue
        torch.manual_seed(seed_model * 1000 + fold)
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
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pooling", default="mean",
                    help="mac dinh 'mean' cho re; phuong sai theo seed la tinh chat cua "
                         "quy trinh train chu khong cua rieng mot kieu pooling")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, series = load_raw()
    man = build_study_manifest(train, series)
    man[REPORT_KEY] = (man[REPORT_COL].astype(str).str.strip().str.lower()
                       .map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()))
    gold_all = gold_subset(man)

    cache = DATA_INTERIM / f"features_dinov2_vits14_{args.size}.npz"
    with np.load(cache) as z:
        feats = {k: z[k] for k in z.files}
    dim = next(iter(feats.values())).shape[1]

    gold = gold_all[gold_all[STUDY_COL].isin(feats)].reset_index(drop=True)
    extra = man[(~man[STUDY_COL].isin(gold_all[STUDY_COL]))
                & (man[STUDY_COL].isin(feats))].reset_index(drop=True)
    llm = pd.read_csv(LLM, dtype={STUDY_COL: str}).set_index(STUDY_COL)
    for lab in L:
        extra[lab] = extra[STUDY_COL].map(llm[lab])
    pool = pd.concat([gold, extra], ignore_index=True)
    print(f"=== Cau hinh D_llm | pooling {args.pooling} | {len(pool):,} ca train ===")
    print(f"  chay {args.seeds} seed, FOLD GIU NGUYEN, chi doi seed cua model\n")

    diem, oofs, folded = [], [], None
    for s in range(1, args.seeds + 1):
        t0 = time.time()
        oof, folded = mot_lan(s, feats, pool, gold, dim, args.epochs, args.lr,
                              device, args.pooling)
        y = folded[L].to_numpy(dtype=float)
        keep = ~np.isnan(oof).all(axis=1)
        a = macro_auc(y[keep], oof[keep], L).macro_auc
        diem.append(a)
        oofs.append(oof)
        print(f"  seed {s}  macro AUC {a:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    d = np.asarray(diem)
    y = folded[L].to_numpy(dtype=float)
    keep = np.logical_and.reduce([~np.isnan(o).all(axis=1) for o in oofs])

    print(f"\n=== Phuong sai chi do SEED ===")
    print(f"  trung binh      {d.mean():.4f}")
    print(f"  do lech chuan   {d.std(ddof=1):.4f}")
    print(f"  thap nhat       {d.min():.4f}")
    print(f"  cao nhat        {d.max():.4f}")
    print(f"  BE RONG         {d.max()-d.min():.4f}  <- chi doi seed, khong doi gi khac")

    print(f"\n=== Doc lai cac ket luan cu bang thuoc nay ===")
    cu = [("pooling max hon mean (ngay 5)", 0.019),
          ("3 mat phang hon 1 (vong 3)", 0.061),
          ("nhan LLM hon tu dien (ngay 10)", 0.054),
          ("per_label_attn hon mean (ngay 10)", 0.023),
          ("gop 4 model hon head chung (ngay 11)", 0.036)]
    be_rong = d.max() - d.min()
    for ten, ch in cu:
        cho = "NAM TRONG nhieu seed" if ch <= be_rong else "vuot nhieu seed"
        print(f"  {ten:<40} {ch:+.3f}  -> {cho}")

    print(f"\n=== Neu nhat seed tot nhat thi phong len bao nhieu? ===")
    print(f"  trung binh cac seed : {d.mean():.4f}")
    print(f"  seed tot nhat       : {d.max():.4f}")
    print(f"  PHONG LEN           : {d.max()-d.mean():+.4f}")
    print("  Do la phan diem KHONG chuyen sang LB duoc: no den tu viec nhin dap an")
    print("  roi nhat, chu khong tu model tot hon.")

    gop = sum(theo_hang(o[keep]) for o in oofs) / len(oofs)
    auc_gop = macro_auc(y[keep], gop, L).macro_auc
    print(f"\n=== Trung binh du doan qua cac seed (HOP LE) ===")
    print(f"  gop {len(oofs)} seed : {auc_gop:.4f}  ({auc_gop-d.mean():+.4f} so voi trung binh)")
    print("  Khac han voi nhat seed: trung binh khong nhin dap an luc nao ca.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "day12_phuong_sai_seed.json").write_text(json.dumps({
        "pooling": args.pooling, "epochs": args.epochs, "n_seed": args.seeds,
        "diem_tung_seed": [float(x) for x in d],
        "trung_binh": float(d.mean()), "do_lech_chuan": float(d.std(ddof=1)),
        "be_rong": float(be_rong),
        "phong_len_neu_nhat_seed_tot_nhat": float(d.max() - d.mean()),
        "gop_cac_seed": float(auc_gop),
    }, indent=2))
    np.savez_compressed(REPORTS_DIR / "day12_seed_oof.npz", y=y,
                        **{f"seed{i+1}": o for i, o in enumerate(oofs)})
    print("\nDa ghi day12_phuong_sai_seed.json va day12_seed_oof.npz")


if __name__ == "__main__":
    main()
