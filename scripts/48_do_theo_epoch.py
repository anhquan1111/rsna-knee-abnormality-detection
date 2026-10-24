"""NGAY 13 - Diem CV phu thuoc vao cach chon epoch bao nhieu?

    python scripts/48_do_theo_epoch.py --ten chung_s224 --epochs 25   # OOF moi epoch
    python scripts/48_do_theo_epoch.py --ten chung_s224 --khong-gold  # chon epoch, khong nhin gold
    python scripts/48_do_theo_epoch.py --ten mp_axial --epochs 9 --seed 7  # them mot seed
    python scripts/48_do_theo_epoch.py --cham                         # cham bo gop, ~1 phut

CHUYEN DA XAY RA
================
Moi thi nghiem truoc day goi `train_head(model, tl, vl)` voi `vl` la fold gold DANG CHAM,
va lay epoch co AUC val cao nhat. Fold co 11-12 ca, nen do la chon tren tap danh gia -
cung loi voi chon seed tot nhat, chi la lap lai 5 lan moi lan chay. No chon epoch 2-22.

Ban nop (script 45) thi khac: train 80 epoch tren gold + nhan LLM, ma nhan LLM la nhan mem
nen AUC = nan moi epoch -> `train_head` lui ve chon theo loss -> gan epoch 80. Ma head
cang train lau cang TE tren gold (chung_s224: 0.735 o epoch 5, 0.674 o epoch 80).

Script nay ghi logit val o MOI epoch trong mot lan chay, de tu cung mot lan train tinh duoc
ca "best epoch tung fold" (cach cu) lan "epoch co dinh" (cach dung).

CHON EPOCH KHONG NHIN GOLD (--khong-gold)
=========================================
Bo 58 ca gold ra hoan toan. Train tren 90% ca nhan LLM, cham AUC so voi nhan LLM (nhi
phan hoa o 0.5) tren 10% con lai. Day khong phai diem that - nhan LLM chi dat 0.893 tren
gold - nhung no tra loi duoc cau "head bat dau hoc vet nhieu tu luc nao" ma khong dong vao
tap danh gia. Do duoc: dinh o epoch 9 (0.7664), giam dan con 0.735 o epoch 25.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from rsna_knee.head import StudyHead, collate_features, masked_bce, predict
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.metrics import macro_auc
from rsna_knee.oof import gold_theo_fold, ma_thu_tu, nap_oof
from rsna_knee.splits import assert_no_leakage, assign_folds

L = list(LABELS)
REPO = Path(__file__).resolve().parents[1]
LLM = REPO / "data" / "external" / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv"
# Cung 5 thanh phan, cung file dac trung voi 45_train_final_ensemble.py
NGUON = {
    "mp_sagittal": DATA_INTERIM / "features_dinov2_vits14_224_sagittal.npz",
    "mp_axial": REPO / "data" / "kaggle_out" / "features_dinov2_axial.npz",
    "mp_coronal": REPO / "data" / "kaggle_out" / "features_dinov2_coronal.npz",
    "chung_s224": DATA_INTERIM / "features_dinov2_vits14_224.npz",
    "chung_b336": DATA_INTERIM / "features_dinov2_vitb14_336.npz",
}
EPOCH_CHOT = 9   # tu --khong-gold, KHONG tu diem gold


def nap(ten: str):
    train, series = load_raw()
    man = build_study_manifest(train, series)
    man["report_key"] = (man[REPORT_COL].astype(str).str.strip().str.lower()
                         .map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()))
    gold_all = gold_subset(man)
    with np.load(NGUON[ten]) as z:
        feats = {k: z[k] for k in z.files}
    llm = pd.read_csv(LLM, dtype={STUDY_COL: str}).set_index(STUDY_COL)
    gold = gold_all[gold_all[STUDY_COL].isin(feats)].reset_index(drop=True)
    extra = man[(~man[STUDY_COL].isin(gold_all[STUDY_COL]))
                & man[STUDY_COL].isin(feats)].reset_index(drop=True)
    for lab in L:
        extra[lab] = extra[STUDY_COL].map(llm[lab])
    return feats, gold, extra


def mot_epoch(model, loader, opt, device):
    model.train()
    for x, m, y, _ in loader:
        opt.zero_grad()
        masked_bce(model(x.to(device), m.to(device)), y.to(device)).backward()
        opt.step()


def ten_file(ten: str, seed: int) -> Path:
    # Seed mac dinh giu ten cu de cac file da ghi van doc duoc
    duoi = "" if seed == SEED else f"_seed{seed}"
    return REPORTS_DIR / f"day13_epoch_{ten}{duoi}.npz"


def do_oof_theo_epoch(ten: str, epochs: int, lr: float, device: str, seed: int = SEED) -> None:
    """Giong run_config cua 41 (D_llm, per_label_attn), ghi logit val o moi epoch.

    `seed` chi doi khoi tao + thu tu xao tron; split fold LUON co dinh theo SEED (giong
    44_phuong_sai_seed.py) de chenh lech giua cac seed chi con la nhieu cua viec train."""
    feats, gold, extra = nap(ten)
    pool = pd.concat([gold, extra], ignore_index=True)
    dim = next(iter(feats.values())).shape[1]
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    assert_no_leakage(folded)
    per_ep = np.full((epochs, len(folded), len(L)), np.nan, dtype=np.float32)
    print(f"{ten}: gold {len(gold)} | ngoai gold {len(extra):,} | {dim} chieu", flush=True)

    for fold in range(N_FOLDS):
        va = folded[folded.fold == fold]
        tr = pool[~pool[STUDY_COL].isin(set(va[STUDY_COL]))]
        tr = tr[~tr["report_key"].isin(set(va["report_key"]))]   # ro ri trung bao cao
        torch.manual_seed(seed + fold)
        model = StudyHead(dim, pooling="per_label_attn").to(device)
        tl = DataLoader(FeatureDataset(feats, tr), batch_size=4, shuffle=True,
                        collate_fn=collate_features)
        vl = DataLoader(FeatureDataset(feats, va), batch_size=4, shuffle=False,
                        collate_fn=collate_features)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(epochs):
            mot_epoch(model, tl, opt, device)
            _, logits, ids = predict(model, vl, device)
            order = {u: k for k, u in enumerate(ids)}
            idx = folded.index[folded[STUDY_COL].isin(ids)]
            per_ep[ep, idx] = logits[[order[u] for u in folded.loc[idx, STUDY_COL]]]
        print(f"  fold {fold} xong", flush=True)

    out = ten_file(ten, seed)
    # KHONG luu nhan gold hay UID: file nay len repo cong khai (xem rsna_knee/oof.py)
    np.savez_compressed(out, per_ep=per_ep, fold=folded["fold"].to_numpy(),
                        uid_md5=ma_thu_tu(folded[STUDY_COL]))
    print(f"  ghi {out.name}")


def chon_epoch_khong_gold(ten: str, epochs: int, lr: float, device: str) -> None:
    feats, _, extra = nap(ten)
    dim = next(iter(feats.values())).shape[1]
    rng = np.random.default_rng(SEED)
    la_val = rng.random(len(extra)) < 0.1
    tr, va = extra[~la_val], extra[la_val].copy()
    va[L] = (va[L].to_numpy(float) >= 0.5).astype(float)

    torch.manual_seed(SEED)
    model = StudyHead(dim, pooling="per_label_attn").to(device)
    tl = DataLoader(FeatureDataset(feats, tr), batch_size=4, shuffle=True,
                    collate_fn=collate_features)
    vl = DataLoader(FeatureDataset(feats, va), batch_size=16, shuffle=False,
                    collate_fn=collate_features)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print(f"{ten}: train {len(tr):,} | val nhan LLM {len(va)} | KHONG co ca gold nao")
    curve = []
    for ep in range(1, epochs + 1):
        mot_epoch(model, tl, opt, device)
        yt, lg, _ = predict(model, vl, device)
        curve.append(macro_auc(yt, lg, L).macro_auc)
        print(f"  epoch {ep:>2}  AUC so voi nhan LLM {curve[-1]:.4f}", flush=True)
    dinh = int(np.argmax(curve)) + 1
    print(f"  dinh o epoch {dinh}")
    (REPORTS_DIR / f"day13_khong_gold_{ten}.json").write_text(json.dumps(
        {"ten": ten, "auc_theo_epoch": curve, "epoch_dinh": dinh}, indent=2))


def theo_hang(p: np.ndarray) -> np.ndarray:
    """Hang tho, giong 47_evaluate_ensemble.py va notebook nop bai."""
    return np.column_stack([rankdata(p[:, j]) for j in range(p.shape[1])])


def best_tung_fold(P: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Cach cu: moi fold lay epoch co AUC cao nhat TREN CHINH FOLD DO."""
    out = np.empty_like(P[0])
    for k in np.unique(fold):
        m = fold == k
        s = []
        for e in range(P.shape[0]):
            try:
                s.append(macro_auc(y[m], P[e, m], L).macro_auc)
            except ValueError:
                s.append(-1.0)
        out[m] = P[int(np.argmax(s)), m]
    return out


def cham() -> None:
    y, fold, uid = gold_theo_fold()
    z = {t: {"per_ep": nap_oof(ten_file(t, SEED), uid, fold)} for t in NGUON}
    n_ep = min(v["per_ep"].shape[0] for v in z.values())

    cach = {
        "best epoch tung fold (cach cu)": lambda P: best_tung_fold(P[:n_ep], y, fold),
        f"epoch co dinh {EPOCH_CHOT} (chot khong nhin gold)": lambda P: P[EPOCH_CHOT - 1],
        f"epoch co dinh {n_ep} (train lau, giong ban da nop)": lambda P: P[n_ep - 1],
    }
    print(f"Cham tren {len(y)} ca gold, out-of-fold. Cot: {', '.join(NGUON)}\n")
    gop_chot = None
    for ten_cach, f in cach.items():
        preds = {t: f(v["per_ep"]) for t, v in z.items()}
        gop = np.mean([theo_hang(p) for p in preds.values()], axis=0)
        rieng = "  ".join(f"{macro_auc(y, p, L).macro_auc:.3f}" for p in preds.values())
        print(f"  {ten_cach:<46} gop {macro_auc(y, gop, L).macro_auc:.4f} | {rieng}")
        if "chot" in ten_cach:
            gop_chot = gop

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        r = macro_auc(y[i], gop_chot[i], L)
        if r.n_scored == len(L):
            boot.append(r.macro_auc)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  KTC 95% bo gop epoch {EPOCH_CHOT}: [{lo:.3f}, {hi:.3f}]")

    # Nhieu seed (neu da chay --seed): diem tung seed cho biet do on dinh, va gop qua seed
    # la cach tang diem HOP LE - khong nhin tap danh gia de chon gi ca.
    seeds = sorted({int(f.stem.split("_seed")[1]) for f in REPORTS_DIR.glob("day13_epoch_*_seed*.npz")})
    seeds = [s for s in seeds
             if all(ten_file(t, s).exists() for t in NGUON)]
    if not seeds:
        return
    print(f"\n  Epoch {EPOCH_CHOT}, nhieu seed (fold co dinh): {[SEED] + seeds}")
    hang_moi_seed = []
    for s in [SEED] + seeds:
        hs = [theo_hang(nap_oof(ten_file(t, s), uid, fold)[EPOCH_CHOT - 1]) for t in NGUON]
        hang_moi_seed.append(hs)
        print(f"    seed {s:<4} gop {macro_auc(y, np.mean(hs, axis=0), L).macro_auc:.4f}")
    diem = [macro_auc(y, np.mean(h, axis=0), L).macro_auc for h in hang_moi_seed]
    print(f"    trung binh {np.mean(diem):.4f} | do lech chuan {np.std(diem, ddof=1):.4f}")
    gop_seed = np.mean([h for hs in hang_moi_seed for h in hs], axis=0)
    print(f"    GOP {len(hang_moi_seed)} seed x 5 model: {macro_auc(y, gop_seed, L).macro_auc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", choices=list(NGUON))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--khong-gold", action="store_true", dest="khong_gold")
    ap.add_argument("--cham", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.cham:
        return cham()
    if not args.ten:
        raise SystemExit("Can --ten <thanh phan> hoac --cham")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.khong_gold:
        chon_epoch_khong_gold(args.ten, args.epochs, args.lr, device)
    else:
        do_oof_theo_epoch(args.ten, args.epochs, args.lr, device, args.seed)


if __name__ == "__main__":
    main()
