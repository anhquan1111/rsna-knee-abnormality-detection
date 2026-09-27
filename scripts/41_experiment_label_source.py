"""NGAY 10a - Nguon nhan nao tot hon: tu dien luat cua ta, hay nhan LLM cong khai?

Giao thuc y het ngay 8, doi DUNG MOT THU: nguon nhan may dua vao train.
Tap danh gia luon la 58 ca nhan bac si. Cung split, cung seed, cung so epoch, cung pooling.

    python scripts/41_experiment_label_source.py --epochs 80

BA CAU HINH
===========
  A_gold_only     58 ca nhan bac si                      (moc duoi)
  B_tu_dien       + 4.349 ca nhan tu dien cua ta         (= cau hinh B ngay 8, 0.6923)
  D_llm           + 4.349 ca nhan LLM cong khai          (moi)

KHAC BIET GIUA HAI NGUON NHAN
=============================
                     tu dien cua ta      LLM cong khai
    do phu               30,8%               100%
    dang             cung 0/1/NaN       MEM, 211 muc 0.005-1.0
    AUC tren 58 ca gold    -                 0.893

Nhan mem dung truc tiep duoc voi BCE: `binary_cross_entropy_with_logits` nhan target
trong [0,1] chu khong bat buoc 0 hoac 1. Nhan 0.25 nghia la "hoi nghieng am tinh" va
gradient se nho hon nhan 0.0 - dung la thu ta muon khi nguon nhan khong chac chan.

CANH BAO DA KIEM CHUNG THAT
===========================
Trong hai bo nhan cong khai tai ve, `report_labels_v5.csv` KHONG DUNG DUOC:

    v4_blend | AUC tren gold 0.8927 | 0.1% gia tri la 0/1 tuyet doi
    v5       | AUC tren gold 1.0000 | 100% gia tri la 0/1 tuyet doi, ngoai gold 0%

v5 chep thang 58 nhan gold vao file. Ai dung no de tu cham se thay diem hoan hao va tin
la minh gioi. Script nay CHI nhan v4_blend, va co kiem lai truoc khi chay.
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
    DATA_RAW,
    LABELS,
    N_FOLDS,
    REPORT_COL,
    REPORTS_DIR,
    SEED,
    STUDY_COL,
)
from rsna_knee.dataset import FeatureDataset
from rsna_knee.features import usable_studies
from rsna_knee.head import StudyHead, collate_features, predict, train_head
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.metrics import macro_auc
from rsna_knee.splits import assert_no_leakage, assign_folds

IMAGES = DATA_RAW / "images"
REPORT_KEY = "report_key"
LLM_PATH = (Path(__file__).resolve().parents[1] / "data" / "external"
            / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv")
L = list(LABELS)


def kiem_nguon_nhan(d: pd.DataFrame, gold: pd.DataFrame, ten: str) -> float:
    """Bo nhan nay co CHEP nhan gold khong? Tra ve AUC tren gold.

    Dau hieu sao chep khong phai la "AUC cao" ma la "AUC cao VA gia tri tren gold dung
    bang 0 hoac 1 tuyet doi trong khi ngoai gold thi khong". Mot bo nhan that su tot van
    se co gia tri lung chung o nhung ca kho.
    """
    g = gold.set_index(STUDY_COL)
    chung = [u for u in g.index if u in d.index]
    if len(chung) < 10:
        raise SystemExit(f"{ten}: chi khop {len(chung)} ca gold, khong kiem duoc")
    yt = g.loc[chung, L].to_numpy(dtype=float)
    yp = d.loc[chung, L].to_numpy(dtype=float)

    # `macro_auc` bo qua NaN o phia NHAN THAT, nhung o day NaN nam o phia NGUON NHAN
    # (tu dien chi ket luan duoc 30,8%). Phai bo ca hai phia, va cham tren dung phan
    # nguon nhan DAM ket luan - do moi la cau hoi: "no ket luan co dung khong".
    from sklearn.metrics import roc_auc_score
    diem = []
    for j in range(len(L)):
        m = np.isfinite(yt[:, j]) & np.isfinite(yp[:, j])
        if m.sum() < 2 or len(np.unique(yt[m, j])) < 2:
            continue
        diem.append(roc_auc_score(yt[m, j], yp[m, j]))
    auc = float(np.mean(diem)) if diem else float("nan")
    yp = np.nan_to_num(yp, nan=0.5)   # chi de dem ty le 0/1 tuyet doi ben duoi

    tren_gold = float(np.mean((yp == 0) | (yp == 1)))
    ngoai = d.loc[[u for u in d.index if u not in set(g.index)], L].to_numpy(dtype=float)
    ngoai_gold = float(np.mean((ngoai == 0) | (ngoai == 1)))
    print(f"  {ten:<10} AUC tren gold {auc:.4f} | 0/1 tuyet doi: tren gold "
          f"{tren_gold:.1%}, ngoai gold {ngoai_gold:.1%}")
    if auc > 0.99 and tren_gold > 0.9 and ngoai_gold < 0.5:
        raise SystemExit(
            f"{ten} CHEP nhan gold: AUC {auc:.4f}, {tren_gold:.0%} gia tri tren gold la "
            f"0/1 tuyet doi nhung ngoai gold chi {ngoai_gold:.0%}.\n"
            "  Dung bo nay se cho diem gia. Khong chay tiep."
        )
    return auc


def luu_cau_hinh_thang(name: str, pool: pd.DataFrame, feats: dict, dim: int,
                       oof_auc: float, epochs: int, lr: float, device: str,
                       pooling: str, backbone: str) -> Path:
    """Train lai cau hinh thang tren TOAN BO pool cua no roi ghi checkpoint.

    Ngay 8 da dinh dung loi nay mot lan: vong lap thi nghiem chi CHAM DIEM roi vut model
    cua tung fold di, nen cau hinh tot nhat khong co file trong so nao - va bai nop se
    chay bang head cu ma khong co dong log nao bao. Do mot dang, nop mot neo.
    """
    torch.manual_seed(SEED)
    ds = FeatureDataset(feats, pool)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_features)
    model = StudyHead(dim, pooling=pooling)
    out = train_head(model, loader, loader, epochs=epochs, lr=lr, device=device, log_every=0)
    model.load_state_dict(out["best"]["state"])

    # Phep kiem I/O (luu -> nap lai -> so sanh) KHONG bat duoc head chua train, vi no chi
    # kiem doc ghi. Do lech chuan du doan giua cac ca moi phan biet duoc "da hoc" voi
    # "tra ve hang so". Ngay 5 da mat mot luot nop vi thieu chot nay.
    _, logits, _ = predict(model, DataLoader(ds, batch_size=4, shuffle=False,
                                             collate_fn=collate_features), device)
    do_lech = float(logits.std(axis=0).mean())
    print(f"  do lech chuan du doan giua cac ca: {do_lech:.4f}")
    if do_lech < 0.01:
        raise SystemExit("Head hinh nhu CHUA TRAIN: du doan gan nhu giong nhau cho moi ca.")

    path = DATA_INTERIM / f"head_{backbone}_{pooling}_llm.pt"
    torch.save({"state_dict": model.state_dict(), "dim": dim, "pooling": pooling,
                "labels": L, "backbone": backbone,
                "oof_macro_auc": round(float(oof_auc), 4),
                "n_train_studies": len(pool), "epochs": epochs,
                "config": name, "nguon_nhan": "llm_labels_v4_blend"}, path)
    return path


def run_config(name: str, pool: pd.DataFrame, gold: pd.DataFrame, feats: dict,
               dim: int, epochs: int, lr: float, device: str,
               pooling: str = "mean") -> dict:
    """Y het run_config cua ngay 8, ke ca buoc chan trung bao cao voi val."""
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    assert_no_leakage(folded)
    oof = np.full((len(folded), len(L)), np.nan, dtype=np.float32)
    n_dup = 0

    for fold in range(N_FOLDS):
        va = folded[folded.fold == fold]
        tr = pool[~pool[STUDY_COL].isin(set(va[STUDY_COL]))]
        if REPORT_KEY in tr.columns and REPORT_KEY in va.columns:
            truoc = len(tr)
            tr = tr[~tr[REPORT_KEY].isin(set(va[REPORT_KEY]))]
            n_dup += truoc - len(tr)

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

    y = folded[L].to_numpy(dtype=float)
    keep = ~np.isnan(oof).all(axis=1)
    return {"config": name, "result": macro_auc(y[keep], oof[keep], L),
            "n_train": len(pool), "oof": oof, "keep": keep, "y": y, "n_dup": n_dup}


def ktc_batcap(y, pa, pb, boot: int, seed: int = SEED):
    """Bootstrap bat cap: moi lan lay mau, CA HAI cau hinh cham tren cung tap ca."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        try:
            d.append(macro_auc(y[i], pb[i], L).macro_auc - macro_auc(y[i], pa[i], L).macro_auc)
        except ValueError:
            continue
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--cache", default=None,
                    help="duong dan npz dac trung; mac dinh suy tu backbone+size")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--pooling", default="mean",
                    choices=["mean", "max", "attn", "per_label_attn"])
    ap.add_argument("--configs", nargs="+", default=None,
                    help="chi chay mot so cau hinh, vd: --configs A_gold_only D_llm")
    ap.add_argument("--tag", default="day10a", help="tien to ten file ket qua")
    ap.add_argument("--chi-luu-head", action="store_true",
                    dest="chi_luu_head",
                    help="bo qua 5-fold, chi train head tren toan bo pool va ghi checkpoint")
    ap.add_argument("--oof-auc", type=float, default=None, dest="oof_auc",
                    help="diem CV da do truoc do, de ghi vao checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, series = load_raw()
    man = build_study_manifest(train, series)
    man[REPORT_KEY] = (man[REPORT_COL].astype(str).str.strip().str.lower()
                       .map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()))
    gold_all = gold_subset(man)

    cache = (Path(args.cache) if args.cache
             else DATA_INTERIM / f"features_{args.backbone}_{args.size}.npz")
    have = usable_studies(cache, IMAGES, extract_missing=False)
    gold = gold_all[gold_all[STUDY_COL].isin(have)].reset_index(drop=True)
    extra = man[(~man[STUDY_COL].isin(gold_all[STUDY_COL]))
                & (man[STUDY_COL].isin(have))].reset_index(drop=True)
    with np.load(cache) as z:
        feats = {k: z[k] for k in z.files}
    dim = next(iter(feats.values())).shape[1]
    print(f"=== Du lieu ===\n  gold {len(gold)} | ngoai gold {len(extra):,} | {dim} chieu")

    # --- hai nguon nhan ---
    print("\n=== Kiem hai nguon nhan truoc khi dung ===")
    tu_dien = pd.read_csv(DATA_INTERIM / "weak_labels.csv",
                          dtype={STUDY_COL: str}).set_index(STUDY_COL)
    if not LLM_PATH.exists():
        raise SystemExit(f"Khong thay {LLM_PATH}")
    llm = pd.read_csv(LLM_PATH, dtype={STUDY_COL: str}).set_index(STUDY_COL)
    auc_td = kiem_nguon_nhan(tu_dien, gold, "tu_dien")
    auc_llm = kiem_nguon_nhan(llm, gold, "llm")

    def gan(nguon: pd.DataFrame) -> pd.DataFrame:
        out = extra.copy()
        for lab in L:
            out[lab] = out[STUDY_COL].map(nguon[lab])
        return out

    e_td, e_llm = gan(tu_dien), gan(llm)
    for ten, e in (("tu_dien", e_td), ("llm", e_llm)):
        v = e[L].to_numpy(dtype=float)
        print(f"  {ten:<10} do phu {np.isfinite(v).mean():.1%} | "
              f"{len(np.unique(v[np.isfinite(v)]))} muc gia tri khac nhau")

    # Cau hinh E tach duoc mot chuyen ma D khong tach duoc.
    #
    # D hon B vi hai ly do tron lan nhau: nhan LLM vua PHU RONG HON (100% so voi 30,7%)
    # vua CHAT LUONG HON (AUC 0.893 so voi 0.736). Nhin rieng D-B thi khong biet phan nao
    # den tu dau - ma hai nguyen nhan do dan toi hai viec lam tiep hoan toan khac nhau:
    # neu la do do phu thi nen di mo rong tu dien; neu la do chat luong thi mo rong tu
    # dien chi nhoi them nhan sai.
    #
    # E = nhan LLM nhung BIT LAI dung cho ma tu dien khong ket luan duoc. Cung do phu voi
    # B, chi khac chat luong. Vay:
    #     E - B  =  phan den tu CHAT LUONG nhan
    #     D - E  =  phan den tu DO PHU
    e_llm_bit = e_llm.copy()
    mat_na = ~np.isfinite(e_td[L].to_numpy(dtype=float))
    v = e_llm_bit[L].to_numpy(dtype=float).copy()   # to_numpy co the tra ve view chi doc
    v[mat_na] = np.nan
    e_llm_bit[L] = v
    print(f"  llm_bit    do phu {np.isfinite(v).mean():.1%}  (bit theo dung mat na cua tu dien)")

    configs = [
        ("A_gold_only", gold),
        ("B_tu_dien", pd.concat([gold, e_td], ignore_index=True)),
        ("D_llm", pd.concat([gold, e_llm], ignore_index=True)),
        ("E_llm_bit_theo_tu_dien", pd.concat([gold, e_llm_bit], ignore_index=True)),
    ]

    if args.configs:
        thieu = [c for c in args.configs if c not in dict(configs)]
        if thieu:
            raise SystemExit(f"Khong co cau hinh {thieu}. "
                             f"Co: {[c for c, _ in configs]}")
        configs = [(c, p_) for c, p_ in configs if c in args.configs]
        print(f"\n  --configs: chi chay {[c for c, _ in configs]}")

    # `--chi-luu-head` bo qua vong 5-fold va chi train lai head tren toan bo pool roi ghi
    # checkpoint. Dung khi da biet diem CV roi (da chay truoc do) va chi con thieu file
    # trong so de nop - vong CV mat 42 phut, buoc nay mat 5.
    if args.chi_luu_head:
        if len(configs) != 1:
            raise SystemExit("--chi-luu-head can dung mot cau hinh: them --configs <ten>")
        ten_ch, pool_ch = configs[0]
        print(f"\n=== Chi train lai head de ghi checkpoint ({ten_ch}) ===")
        print(f"  KHONG chay 5-fold. Diem CV lay tu --oof-auc = {args.oof_auc}")
        if args.oof_auc is None:
            raise SystemExit("Can --oof-auc <so> de ghi vao checkpoint "
                             "(lay tu lan chay CV truoc).")
        ck = luu_cau_hinh_thang(ten_ch, pool_ch, feats, dim, args.oof_auc,
                                args.epochs, args.lr, device, args.pooling, args.backbone)
        print(f"  ghi {ck.name} ({ck.stat().st_size/1024:.0f} KB)")
        return

    print("\n=== Ket qua (danh gia LUON tren 58 ca nhan bac si) ===")
    rows, res = [], {}
    for name, pool in configs:
        t0 = time.time()
        o = run_config(name, pool, gold, feats, dim, args.epochs, args.lr, device,
                       pooling=args.pooling)
        res[name] = o
        rows.append(o["result"].to_row(config=name, n_train=o["n_train"],
                                       seconds=round(time.time() - t0, 1)))
        print(f"  {name:<14} n_train={o['n_train']:>5}  {o['result']}  "
              f"({time.time()-t0:.0f}s)"
              + (f"  [bo {o['n_dup']} dong trung bao cao]" if o["n_dup"] else ""))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / f"{args.tag}_label_source.csv", index=False)

    # Phan so sanh duoi day can du ca bon cau hinh. Khi chay voi --configs (vd de thu
    # mot kieu pooling khac tren rieng D_llm) thi bo qua, chi ghi ket qua tho.
    du_bon = all(k in res for k in
                 ("A_gold_only", "B_tu_dien", "D_llm", "E_llm_bit_theo_tu_dien"))
    if not du_bon:
        np.savez_compressed(REPORTS_DIR / f"{args.tag}_oof.npz",
                            y=next(iter(res.values()))["y"],
                            **{k: v["oof"] for k, v in res.items()})
        (REPORTS_DIR / f"{args.tag}_label_source.json").write_text(json.dumps({
            "pooling": args.pooling, "epochs": args.epochs,
            **{k: v["result"].macro_auc for k, v in res.items()},
        }, indent=2))
        print(f"\n  chi chay {list(res)} -> bo qua phan tach nguyen nhan.")
        print(f"  da ghi {args.tag}_oof.npz va {args.tag}_label_source.json")
        return

    a, b, d = res["A_gold_only"], res["B_tu_dien"], res["D_llm"]
    print("\n=== Chenh lech tung nhan ===")
    print(f"  {'nhan':<18} {'A':>7} {'B tu dien':>10} {'E bit':>8} {'D llm':>8} {'D-B':>8}")
    for lab in L:
        va = a["result"].per_label.get(lab)
        vb = b["result"].per_label.get(lab)
        vd = d["result"].per_label.get(lab)
        ve = res["E_llm_bit_theo_tu_dien"]["result"].per_label.get(lab)
        if None in (va, vb, vd, ve):
            continue
        print(f"  {lab:<18} {va:>7.3f} {vb:>10.3f} {ve:>8.3f} {vd:>8.3f} {vd-vb:>+8.3f}")

    e = res["E_llm_bit_theo_tu_dien"]

    print("\n=== Tach nguyen nhan: chat luong nhan hay do phu? ===")
    so_sanh = [
        ("D - B  (LLM day du vs tu dien)", b, d),
        ("E - B  (CHAT LUONG, cung do phu)", b, e),
        ("D - E  (DO PHU, cung chat luong)", e, d),
    ]
    tom = {}
    for ten, x, y2 in so_sanh:
        keep = x["keep"] & y2["keep"]
        lo, hi, n = ktc_batcap(x["y"][keep], x["oof"][keep], y2["oof"][keep], args.boot)
        obs = y2["result"].macro_auc - x["result"].macro_auc
        ket = ("chua ket luan duoc" if lo < 0 < hi
               else ("CO, theo huong tot" if lo > 0 else "CO, theo huong xau"))
        print(f"  {ten:<36} {obs:+.4f}  KTC [{lo:+.4f}, {hi:+.4f}]  -> {ket}")
        tom[ten.split()[0] + "-" + ten.split()[2]] = {
            "chenh": obs, "ci95": [lo, hi], "ket_luan": ket, "boot": n}

    keep = b["keep"] & d["keep"]
    lo, hi, n = ktc_batcap(b["y"][keep], b["oof"][keep], d["oof"][keep], args.boot)
    obs = d["result"].macro_auc - b["result"].macro_auc
    ket = ("CHUA ket luan duoc" if lo < 0 < hi
           else ("nhan LLM HON tu dien" if lo > 0 else "nhan LLM KEM hon tu dien"))

    # Luu du doan out-of-fold de cac buoc sau (error analysis ngay 10, gop model) khong
    # phai train lai 20 phut moi lan muon hoi mot cau moi.
    np.savez_compressed(REPORTS_DIR / f"{args.tag}_oof.npz",
                        y=b["y"], **{k: v["oof"] for k, v in res.items()})
    print(f"\n  da luu du doan out-of-fold -> {args.tag}_oof.npz "
          f"({', '.join(res)})")

    (REPORTS_DIR / f"{args.tag}_label_source.json").write_text(json.dumps({
        "A_gold_only": a["result"].macro_auc,
        "B_tu_dien": b["result"].macro_auc,
        "D_llm": d["result"].macro_auc,
        "E_llm_bit": e["result"].macro_auc,
        "auc_nguon_tren_gold": {"tu_dien": auc_td, "llm": auc_llm},
        "chenh_D_B": obs, "ci95": [lo, hi], "ket_luan": ket,
        "tach_nguyen_nhan": tom,
        "epochs": args.epochs, "boot": n,
    }, indent=2))
    print(f"\nDa ghi {args.tag}_label_source.csv va .json")

    print("\n=== Ghi checkpoint cho cau hinh thang ===")
    ten_thang = max(res, key=lambda k: res[k]["result"].macro_auc)
    pool_thang = dict(configs)[ten_thang]
    auc_thang = res[ten_thang]["result"].macro_auc
    print(f"  thang: {ten_thang} (macro AUC {auc_thang:.4f}, "
          f"{len(pool_thang):,} ca train, pooling {args.pooling})")
    ck = luu_cau_hinh_thang(ten_thang, pool_thang, feats, dim, auc_thang,
                            args.epochs, args.lr, device, args.pooling, args.backbone)
    print(f"  ghi {ck.name} ({ck.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
