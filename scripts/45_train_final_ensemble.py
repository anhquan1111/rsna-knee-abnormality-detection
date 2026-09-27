"""Train va luu DU 5 head cua bo gop cuoi cung.

    python scripts/45_train_final_ensemble.py --epochs 80

NAM MODEL, VA VI SAO DUNG NAM CAI NAY
=====================================
    mp_sagittal    head rieng, chi thay mat phang Sagittal   (S@224)  0.7106
    mp_axial       head rieng, chi thay mat phang Axial      (S@224)  0.7691
    mp_coronal     head rieng, chi thay mat phang Coronal    (S@224)  0.7394
    chung_s224     head chung, noi lat ca ba mat phang       (S@224)  0.7698
    chung_b336     head chung, noi lat ca ba mat phang       (B@336)  0.7481
    ------------------------------------------------------------------------
    GOP CA NAM (trung binh theo hang)                                 0.8113

Khong cai nao trong nam dat 0.78 khi dung mot minh. Gop lai duoc 0.8113 vi chung sai o
NHUNG CHO KHAC NHAU - dung dieu kien ngay 9 do duoc. Vi du B@336 hon han o PF OA (+0.069)
va Effusion (+0.047) nhung kem han o MCL (-0.184) va Medial Meniscus (-0.117).

DUNG "GOP TAT CA", KHONG PHAI "GOP TO HOP TOT NHAT"
===================================================
Da thu ca 26 to hop con co the co. Lay cai diem cao nhat trong 26 la CHON TREN TAP DANH
GIA - cung loi voi viec chon seed tot nhat, va phan diem phong len do khong chuyen sang
bang xep hang duoc.

"Gop tat ca model da train" la mot quy tac dat truoc, khong nhin diem. No tinh co cung
la to hop tot nhat trong 26, nhung day la he qua chu khong phai ly do chon.

TRUNG BINH THEO HANG, khong theo gia tri: nam head train rieng nen thang do logit cua
chung khong chung goc. Trung binh logit se de head nao "tu tin" hon lan at, ma do tu tin
do khong so duoc giua cac model khac nhau. AUC von chi quan tam thu hang.
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
from torch.utils.data import DataLoader

from rsna_knee.config import (
    DATA_INTERIM,
    LABELS,
    REPORT_COL,
    SEED,
    STUDY_COL,
)
from rsna_knee.dataset import FeatureDataset
from rsna_knee.head import StudyHead, collate_features, predict, train_head
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw

L = list(LABELS)
REPORT_KEY = "report_key"
REPO = Path(__file__).resolve().parents[1]
LLM = REPO / "data" / "external" / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv"

# (ten, duong dan dac trung, backbone, size, mat phang, diem CV da do)
THANH_PHAN = [
    ("mp_sagittal", DATA_INTERIM / "features_dinov2_vits14_224.npz.bak",
     "dinov2_vits14", 224, "sagittal", 0.7106),
    ("mp_axial", REPO / "data" / "kaggle_out" / "features_dinov2_axial.npz",
     "dinov2_vits14", 224, "axial", 0.7691),
    ("mp_coronal", REPO / "data" / "kaggle_out" / "features_dinov2_coronal.npz",
     "dinov2_vits14", 224, "coronal", 0.7394),
    ("chung_s224", DATA_INTERIM / "features_dinov2_vits14_224.npz",
     "dinov2_vits14", 224, "all", 0.7698),
    ("chung_b336", DATA_INTERIM / "features_dinov2_vitb14_336.npz",
     "dinov2_vitb14", 336, "all", 0.7481),
]
OUT = DATA_INTERIM / "ensemble"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pooling", default="per_label_attn")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, series = load_raw()
    man = build_study_manifest(train, series)
    man[REPORT_KEY] = (man[REPORT_COL].astype(str).str.strip().str.lower()
                       .map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()))
    gold_all = gold_subset(man)
    llm = pd.read_csv(LLM, dtype={STUDY_COL: str}).set_index(STUDY_COL)

    thieu = [t[0] for t in THANH_PHAN if not t[1].exists()]
    if thieu:
        raise SystemExit(f"Thieu dac trung cho: {thieu}")

    OUT.mkdir(parents=True, exist_ok=True)
    ghi = []
    for ten, path, backbone, size, plane, cv in THANH_PHAN:
        t0 = time.time()
        with np.load(path) as z:
            feats = {k: z[k] for k in z.files}
        dim = next(iter(feats.values())).shape[1]

        gold = gold_all[gold_all[STUDY_COL].isin(feats)].reset_index(drop=True)
        extra = man[(~man[STUDY_COL].isin(gold_all[STUDY_COL]))
                    & (man[STUDY_COL].isin(feats))].reset_index(drop=True)
        for lab in L:
            extra[lab] = extra[STUDY_COL].map(llm[lab])
        pool = pd.concat([gold, extra], ignore_index=True)

        # Train tren TOAN BO pool - khong chia fold. Diem CV da do o buoc thi nghiem roi;
        # buoc nay chi de lay trong so dung het du lieu.
        torch.manual_seed(SEED)
        ds = FeatureDataset(feats, pool)
        loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_features)
        model = StudyHead(dim, pooling=args.pooling)
        out = train_head(model, loader, loader, epochs=args.epochs, lr=args.lr,
                         device=device, log_every=0)
        model.load_state_dict(out["best"]["state"])

        # Chot chan "head chua train" cua ngay 5: phep kiem luu/nap lai KHONG bat duoc
        # loi nay vi no chi kiem doc ghi.
        _, logits, _ = predict(model, DataLoader(ds, batch_size=4, shuffle=False,
                                                 collate_fn=collate_features), device)
        do_lech = float(logits.std(axis=0).mean())
        if do_lech < 0.01:
            raise SystemExit(f"{ten}: head hinh nhu CHUA TRAIN (do lech {do_lech:.5f})")

        f = OUT / f"head_{ten}.pt"
        torch.save({"state_dict": model.state_dict(), "dim": dim,
                    "pooling": args.pooling, "labels": L,
                    "ten": ten, "backbone": backbone, "size": size, "plane": plane,
                    "cv_macro_auc": cv, "n_train_studies": len(pool),
                    "epochs": args.epochs, "nguon_nhan": "llm_labels_v4_blend"}, f)
        ghi.append({"ten": ten, "backbone": backbone, "size": size, "plane": plane,
                    "dim": dim, "cv": cv, "file": f.name,
                    "kb": round(f.stat().st_size / 1024)})
        print(f"  {ten:<14} {backbone} @{size} | {plane:<8} | {dim} chieu | "
              f"CV {cv:.4f} | do lech {do_lech:.3f} | {time.time()-t0:.0f}s", flush=True)
        del feats

    (OUT / "ensemble.json").write_text(json.dumps({
        "pooling": args.pooling, "epochs": args.epochs,
        "cach_gop": "trung binh theo hang (rank average)",
        "cv_gop_ca_nam": 0.8113,
        "thanh_phan": ghi,
    }, indent=2))
    print(f"\nDa ghi {len(ghi)} head vao {OUT}")
    print("Buoc tiep: python scripts/49_pack_submission_assets.py")


if __name__ == "__main__":
    main()
