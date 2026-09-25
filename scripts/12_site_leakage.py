"""Ngay 4 - phan con treo: ro ri tang 3 (theo may chup).

Ngay 4 da chan duoc ro ri tang 1 (slice -> study) va do duoc bang mot con so bang 0.
Tang 3 - "may A co mat o ca train lan val nen model hoc dau van tay may thay vi benh ly"
- luc do KHONG kiem duoc, vi CSV khong co cot may chup. Header DICOM thi co.

Script nay tra loi ba cau, theo thu tu:
  1. Co bao nhieu hang may that su, sau khi chuan hoa ten?
  2. Dau van tay may co NAM TRONG dac trung khong? (do bang cach thu doan hang tu dac trung)
  3. Neu co thi chan duoc khong, va chan thi mat gi?

Cau 2 la cau quan trong nhat va thuong bi bo qua: neu model khong doan noi hang may tu
dac trung thi ro ri tang 3 chi la noi lo ly thuyet, khong can chan.

Chay:  python scripts/12_site_leakage.py
Can:   data/manifest/study_manufacturer.csv (tu scripts/07_ingest_kaggle_features.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from rsna_knee.config import (
    DATA_INTERIM,
    DATA_MANIFEST,
    LABELS,
    N_FOLDS,
    REPORTS_DIR,
    SEED,
    STUDY_COL,
)
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.splits import assert_no_leakage, assign_folds


def main() -> None:
    vendor_csv = DATA_MANIFEST / "study_manufacturer.csv"
    if not vendor_csv.exists():
        raise SystemExit(f"Chua co {vendor_csv}.\n"
                         "Chay scripts/07_ingest_kaggle_features.py truoc.")

    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold = gold_subset(man).reset_index(drop=True)

    vendors = pd.read_csv(vendor_csv).set_index(STUDY_COL)["manufacturer_norm"]
    gold["vendor"] = gold[STUDY_COL].map(vendors)
    co_vendor = gold["vendor"].notna()

    print("=== 1. Co bao nhieu hang may that su? ===")
    print(f"  study gold biet hang may: {int(co_vendor.sum())} / {len(gold)}")
    if int(co_vendor.sum()) < 20:
        raise SystemExit("Qua it study gold biet hang may - can trich them dac trung truoc.")
    g = gold[co_vendor].reset_index(drop=True)
    print(g["vendor"].value_counts().to_string())
    n_vendor = g["vendor"].nunique()
    print(f"  -> {n_vendor} hang")

    print("\n=== 2. Dau van tay may co nam trong dac trung khong? ===")
    npz = DATA_INTERIM / "features_dinov2_vits14_224.npz"
    if not npz.exists():
        print(f"  bo qua - chua co {npz.name}")
    else:
        with np.load(npz) as z:
            feats = {k: z[k] for k in z.files}
        # Gop dac trung tung lat len cap study dung cach giong het luc train (mean pooling)
        co_dt = [u for u in g[STUDY_COL] if u in feats]
        # cache dac trung luu fp16 (xem scripts/07) - ep fp32 truoc khi dua vao sklearn
        X = np.stack([feats[u].astype(np.float32).mean(axis=0) for u in co_dt])
        y = g.set_index(STUDY_COL).loc[co_dt, "vendor"].to_numpy()
        print(f"  {len(co_dt)} study co dac trung | {len(set(y))} hang")

        # Chi giu hang co it nhat 2 ca, neu khong khong chia fold duoc
        du = pd.Series(y).value_counts()
        keep = np.isin(y, du[du >= 2].index)
        X, y = X[keep], y[keep]

        if len(set(y)) < 2:
            print("  khong du hang de thu doan - bo qua")
        else:
            # Doan hang may tu dac trung, danh gia out-of-fold
            skf = StratifiedGroupKFold(n_splits=min(5, int(pd.Series(y).value_counts().min())),
                                       shuffle=True, random_state=SEED)
            dung = 0
            for tr, va in skf.split(X, y, groups=np.arange(len(y))):
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(X[tr], y[tr])
                dung += int((clf.predict(X[va]) == y[va]).sum())
            acc = dung / len(y)
            # Moc so sanh: doan luon hang pho bien nhat
            moc = float(pd.Series(y).value_counts(normalize=True).max())
            print(f"  doan dung hang may tu dac trung : {acc:.1%}")
            print(f"  doan bua (luon chon hang dong nhat): {moc:.1%}")
            if acc > moc + 0.15:
                print("  -> Dac trung CO MANG dau van tay may chup. Ro ri tang 3 la RUI RO THAT.")
            else:
                print("  -> Khong hon moc bao nhieu; dau van tay may yeu trong dac trung nay.")

    print("\n=== 3. Split hien tai co tron hang may giua train/val khong? ===")
    folded = assign_folds(g, n_splits=N_FOLDS, seed=SEED)
    assert_no_leakage(folded)
    tron = 0
    for f in range(N_FOLDS):
        va_v = set(folded[folded.fold == f]["vendor"])
        tr_v = set(folded[folded.fold != f]["vendor"])
        chung = va_v & tr_v
        tron += len(chung)
        print(f"  fold {f}: {len(va_v)} hang o val, {len(chung)} hang cung co o train")
    print(f"  -> {tron} cap (fold, hang) bi tron. Split theo study KHONG chan tang 3.")

    print("\n=== 4. Chan duoc khong, va chan thi mat gi? ===")
    # Nhom theo hang: moi hang nam tron ven trong mot fold
    from sklearn.model_selection import GroupKFold

    k_toi_da = min(n_vendor, N_FOLDS)
    print(f"  so fold toi da neu nhom theo hang: {k_toi_da} (bang so hang, khong the hon)")
    gkf = GroupKFold(n_splits=k_toi_da)
    h = g.copy()
    h["fold"] = -1
    for fi, (_, va) in enumerate(gkf.split(h, groups=h["vendor"])):
        h.loc[va, "fold"] = fi

    hong_study, hong_vendor = 0, 0
    print(f"  {'fold':>5} {'n':>4} {'hang':<28} {'nhan khong cham duoc':>22}")
    for fi in range(k_toi_da):
        va = h[h.fold == fi]
        vo = [l for l in LABELS if va[l].nunique() < 2]
        hong_vendor += len(vo)
        print(f"  {fi:>5} {len(va):>4} {','.join(sorted(set(va['vendor']))):<28} {len(vo):>22}")
    for fi in range(N_FOLDS):
        va = folded[folded.fold == fi]
        hong_study += sum(1 for l in LABELS if va[l].nunique() < 2)

    print(f"\n  Nhom theo STUDY (hien tai) : {hong_study:>3} cap (fold, nhan) khong cham duoc")
    print(f"  Nhom theo HANG MAY         : {hong_vendor:>3} cap (fold, nhan) khong cham duoc")
    print("\n  Danh doi: chan duoc ro ri tang 3 nhung fold bi lech han ve co mau va phan bo")
    print("  nhan, va so fold bi tran bang so hang. Voi co mau nay, cai gia thuong dat hon")
    print("  cai duoc - ghi lai de quyet dinh co can cu, khong chon theo cam tinh.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "day4_site_leakage.csv"
    pd.DataFrame([{
        "n_gold_co_vendor": int(co_vendor.sum()),
        "n_vendor": n_vendor,
        "cap_fold_hang_bi_tron": tron,
        "nhan_hong_nhom_theo_study": hong_study,
        "nhan_hong_nhom_theo_hang": hong_vendor,
    }]).to_csv(out, index=False)
    print(f"\nDa ghi {out}")


if __name__ == "__main__":
    main()
