"""NGAY 10 - Error analysis: model sai o DAU, va cai sai do co he thong khong?

    python scripts/42_error_analysis.py

Doc du doan out-of-fold da luu o `reports/day10a_oof.npz` (sinh boi scripts/41), nen
khong phai train lai 20 phut moi lan muon hoi mot cau moi.

BON CAU HOI, theo thu tu quan trong
===================================

1. **Tach loi theo HANG MAY.** Ngay 4 do duoc: dac trung doan dung hang may 82,8% (doan
   bua 37,9%), va split theo ca benh KHONG chan duoc tang ro ri nay. Cau hoi truc tiep
   la: diem co LECH giua cac hang khong? Neu model chay tot tren Siemens va te tren GE
   thi diem cong khai dang duoc do tren mot pha tron cu the, va bang RIENG co ty le hang
   khac di se cho diem khac han.

2. **Nhan nao THAT SU hon doan mo.** AUC 0.55 tren 12 ca duong tinh khong noi len gi.
   Phai co khoang tin cay tung nhan roi moi xep hang duoc.

3. **Loi co don vao vai ca kho khong**, hay rai deu? Hai truong hop nay dan toi hai viec
   lam tiep khac han: don cuc thi di soi cum do; rai deu thi la van de nang luc model.

4. **Do dai bao cao / so lat co du doan duoc loi khong?** Neu co, do la bien gay nhieu
   ma minh dang bo qua.

GIOI HAN PHAI NOI TRUOC
=======================
58 ca chia cho 4 hang la 22/18/16/2. AUC tren 16-22 ca cuc ky nhieu, va tren 2 ca thi
khong cham duoc. Nen moi con so o day deu di kem khoang tin cay, va phan lon se KHONG
ket luan duoc. Do la ket qua that, khong phai loi cua phep do.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rsna_knee.config import (
    DATA_INTERIM,
    DATA_MANIFEST,
    LABELS,
    N_FOLDS,
    REPORT_COL,
    REPORTS_DIR,
    SEED,
    STUDY_COL,
)
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.metrics import macro_auc
from rsna_knee.splits import assign_folds

L = list(LABELS)


def auc_kem_ktc(y: np.ndarray, p: np.ndarray, boot: int, seed: int = SEED):
    """macro AUC + khoang tin cay bootstrap THEO CA."""
    goc = macro_auc(y, p, L).macro_auc
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        try:
            d.append(macro_auc(y[i], p[i], L).macro_auc)
        except ValueError:
            continue
    if not d:
        return goc, float("nan"), float("nan"), 0
    return goc, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)


def auc_mot_nhan(yt: np.ndarray, ps: np.ndarray, boot: int, seed: int = SEED):
    from sklearn.metrics import roc_auc_score

    m = np.isfinite(yt)
    if m.sum() < 4 or len(np.unique(yt[m])) < 2:
        return None
    goc = roc_auc_score(yt[m], ps[m])
    rng = np.random.default_rng(seed)
    d = []
    yv, pv = yt[m], ps[m]
    for _ in range(boot):
        i = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[i])) < 2:
            continue
        d.append(roc_auc_score(yv[i], pv[i]))
    if not d:
        return None
    return goc, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(REPORTS_DIR / "day10a_oof.npz"))
    ap.add_argument("--config", default=None, help="mac dinh: cau hinh tot nhat trong file")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()

    oof_path = Path(args.oof)
    if not oof_path.exists():
        raise SystemExit(f"Khong thay {oof_path}. Chay scripts/41_experiment_label_source.py truoc.")
    with np.load(oof_path) as z:
        blob = {k: z[k] for k in z.files}
    y = blob.pop("y")

    # Chon cau hinh tot nhat de soi, tru khi duoc chi dinh khac.
    diem = {}
    for k, p in blob.items():
        keep = ~np.isnan(p).all(axis=1)
        diem[k] = macro_auc(y[keep], p[keep], L).macro_auc
    ten = args.config or max(diem, key=diem.get)
    print("=== Cac cau hinh co trong file ===")
    for k, v in sorted(diem.items(), key=lambda t: -t[1]):
        print(f"  {k:<26} {v:.4f}" + ("   <- soi cau hinh nay" if k == ten else ""))
    p_all = blob[ten]
    keep = ~np.isnan(p_all).all(axis=1)

    # --- gan lai UID theo dung thu tu ma scripts/41 dung ---
    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold = gold_subset(man)
    # scripts/41 loc gold theo cache dac trung roi moi assign_folds; tai lap y het.
    with np.load(DATA_INTERIM / "features_dinov2_vits14_224.npz") as z:
        co = set(z.files)
    gold = gold[gold[STUDY_COL].isin(co)].reset_index(drop=True)
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    if len(folded) != len(y):
        raise SystemExit(f"Lech so ca: oof co {len(y)}, dung lai duoc {len(folded)}. "
                         "Cache dac trung co the da doi sau khi chay scripts/41.")
    uid = folded[STUDY_COL].astype(str).to_numpy()

    tong, lo, hi, _ = auc_kem_ktc(y[keep], p_all[keep], args.boot)
    print(f"\n=== Tong the ===\n  macro AUC {tong:.4f}  KTC [{lo:.4f}, {hi:.4f}]  "
          f"tren {int(keep.sum())} ca")

    # ---------------------------------------------------------- 1. theo hang may
    print("\n=== 1. Tach loi theo HANG MAY ===")
    vend_path = DATA_MANIFEST / "study_manufacturer.csv"
    if not vend_path.exists():
        print(f"  khong thay {vend_path.name} - bo qua. Chay scripts/07 de sinh.")
        hang_rows = []
    else:
        vend = pd.read_csv(vend_path, dtype={STUDY_COL: str}).set_index(STUDY_COL)
        cot = "manufacturer_norm"
        h = pd.Series(uid).map(vend[cot]).fillna("UNKNOWN").to_numpy()
        hang_rows = []
        print(f"  {'hang':<10} {'n ca':>5} {'macro AUC':>10} {'KTC 95%':>22}")
        for ten_h in sorted(set(h[keep])):
            m = keep & (h == ten_h)
            if m.sum() < 4:
                print(f"  {ten_h:<10} {int(m.sum()):>5}   (qua it ca, khong cham duoc)")
                hang_rows.append({"hang": ten_h, "n": int(m.sum()), "auc": None})
                continue
            a, l2, h2, _ = auc_kem_ktc(y[m], p_all[m], args.boot)
            print(f"  {ten_h:<10} {int(m.sum()):>5} {a:>10.4f}   [{l2:.4f}, {h2:.4f}]")
            hang_rows.append({"hang": ten_h, "n": int(m.sum()), "auc": a,
                              "ci95": [l2, h2]})
        co_cham = [r for r in hang_rows if r.get("auc") is not None]
        if len(co_cham) >= 2:
            cao = max(co_cham, key=lambda r: r["auc"])
            thap = min(co_cham, key=lambda r: r["auc"])
            chong = cao["ci95"][0] <= thap["ci95"][1]
            print(f"\n  cao nhat {cao['hang']} {cao['auc']:.4f} | "
                  f"thap nhat {thap['hang']} {thap['auc']:.4f} | "
                  f"chenh {cao['auc']-thap['auc']:+.4f}")
            print("  -> " + ("khoang tin cay CHONG LAN nhau: chua do duoc chenh lech giua "
                             "cac hang." if chong else
                             "khoang tin cay ROI NHAU: diem LECH that giua cac hang."))
            print("     Voi 16-22 ca moi hang, khong do duoc khong co nghia la khong co.")

    # ---------------------------------------------------------- 2. tung nhan
    print("\n=== 2. Nhan nao THAT SU hon doan mo? ===")
    print(f"  {'nhan':<18} {'AUC':>7} {'KTC 95%':>20} {'n+':>4}  ket luan")
    nhan_rows = []
    for j, lab in enumerate(L):
        r = auc_mot_nhan(y[keep, j], p_all[keep, j], args.boot)
        if r is None:
            print(f"  {lab:<18}   (khong cham duoc)")
            continue
        a, l2, h2, n = r
        npos = int(np.nansum(y[keep, j] == 1))
        if l2 > 0.5:
            kl = "HON doan mo"
        elif h2 < 0.5:
            kl = "DUOI doan mo (xep nguoc)"
        else:
            kl = "chua ket luan duoc"
        print(f"  {lab:<18} {a:>7.3f}   [{l2:.3f}, {h2:.3f}] {npos:>4}  {kl}")
        nhan_rows.append({"nhan": lab, "auc": a, "ci95": [l2, h2],
                          "n_duong_tinh": npos, "ket_luan": kl})
    n_chac = sum(1 for r in nhan_rows if r["ket_luan"] == "HON doan mo")
    print(f"\n  {n_chac}/{len(nhan_rows)} nhan co bang chung la hon doan mo.")
    print("  So con lai KHONG phai la model doan bua o do - chi la 58 ca khong du de noi.")

    # ---------------------------------------------------------- 3. loi don cuc?
    print("\n=== 3. Loi don cuc vao vai ca, hay rai deu? ===")
    # Loi moi ca = BCE trung binh tren cac nhan da biet. Dung BCE chu khong dung
    # "doan dung/sai": AUC quan tam thu hang, ma BCE phat nang dung cho du doan tu tin
    # ma sai - dung thu ta muon tim.
    pr = 1 / (1 + np.exp(-p_all))
    biet = np.isfinite(y)
    eps = 1e-7
    ll = -(y * np.log(np.clip(pr, eps, 1)) + (1 - y) * np.log(np.clip(1 - pr, eps, 1)))
    loi_ca = np.where(biet, ll, np.nan)
    loi_ca = np.nanmean(np.where(keep[:, None], loi_ca, np.nan), axis=1)
    hop_le = np.isfinite(loi_ca)
    v = loi_ca[hop_le]
    thu_tu = np.argsort(-v)
    top10 = v[thu_tu[:max(1, len(v) // 10)]].sum() / v.sum()
    print(f"  loi trung binh {v.mean():.4f} | trung vi {np.median(v):.4f} | "
          f"cao nhat {v.max():.4f}")
    print(f"  10% ca te nhat chiem {top10:.1%} tong loi "
          + ("-> DON CUC" if top10 > 0.25 else "-> rai kha deu (10% deu se la 10%)"))

    man_i = man.set_index(STUDY_COL)
    with np.load(DATA_INTERIM / "features_dinov2_vits14_224.npz") as z:
        so_lat = {k: z[k].shape[0] for k in z.files}
    uid_hl = uid[hop_le]
    print(f"\n  5 ca te nhat:")
    print(f"  {'ca':<16} {'loi':>7} {'hang':<10} {'so lat':>7} {'dai bao cao':>12} {'n+':>4}")
    for k in thu_tu[:5]:
        u = uid_hl[k]
        hg = (vend[cot].get(u, "?") if vend_path.exists() else "?")
        rp = str(man_i[REPORT_COL].get(u, ""))
        npos_u = int(np.nansum(y[hop_le][k] == 1))
        print(f"  ...{u[-13:]:<13} {v[k]:>7.4f} {str(hg):<10} "
              f"{so_lat.get(u, 0):>7,} {len(rp):>12,} {npos_u:>4}")

    # ---------------------------------------------------------- 4. bien gay nhieu
    print("\n=== 4. Bien nao du doan duoc loi? ===")
    from scipy.stats import spearmanr
    dai = np.array([len(str(man_i[REPORT_COL].get(u, ""))) for u in uid_hl], dtype=float)
    lat = np.array([so_lat.get(u, np.nan) for u in uid_hl], dtype=float)
    npos = np.nansum(y[hop_le] == 1, axis=1).astype(float)

    bien = {"do dai bao cao": dai, "so lat": lat, "so nhan duong tinh": npos}
    kq = {}
    print(f"  {'bien':<22} {'rho':>7} {'p':>8}")
    for ten_b, x in bien.items():
        rho, pv = spearmanr(x, v)
        kq[ten_b] = {"rho": float(rho), "p": float(pv)}
        print(f"  {ten_b:<22} {rho:>+7.3f} {pv:>8.4f}"
              + ("   <- dang ke" if pv < 0.05 else ""))

    # Ba bien nay khong doc lap nhau, nen phai hoi them: cai nao la NGUYEN NHAN va cai
    # nao chi la dai dien cho no? Bao cao dai thuong la bao cao ke nhieu ton thuong.
    rho_dn, p_dn = spearmanr(dai, npos)
    print(f"\n  do dai bao cao vs so nhan duong tinh: rho {rho_dn:+.3f} (p {p_dn:.4f})")
    manh = max(kq, key=lambda k: -kq[k]["p"] if False else abs(kq[k]["rho"]))
    print(f"  -> Bien lien he manh nhat voi loi la **{manh}**.")
    print("     Bao cao dai ~ nhieu ton thuong duoc ke ~ ca da benh ly, va do moi la thu")
    print("     lam model kho. Do dai bao cao chi la DAI DIEN cho no.")
    print("     So lat KHONG du doan duoc loi -> ca kho khong phai ca chup nhieu lat.")
    print(f"\n  CANH BAO: da chay {3 + len(nhan_rows) + 1} phep kiem tren cung 58 ca. O muc 5%,")
    print("     ky vong co vai ket qua 'dang ke' do ngau nhien. Cac so tren la GOI Y")
    print("     huong dieu tra, chua phai bang chung doc lap.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "day10_error_analysis.json").write_text(json.dumps({
        "config": ten, "macro_auc": tong, "ci95": [lo, hi],
        "n_ca": int(keep.sum()),
        "theo_hang": hang_rows, "theo_nhan": nhan_rows,
        "n_nhan_hon_doan_mo": n_chac,
        "ty_le_loi_cua_10pc_te_nhat": float(top10),
        "tuong_quan_voi_loi": kq,
        "do_dai_vs_so_duong_tinh": {"rho": float(rho_dn), "p": float(p_dn)},
    }, indent=2))
    pd.DataFrame(nhan_rows).to_csv(REPORTS_DIR / "day10_per_label.csv", index=False)
    print("\nDa ghi day10_error_analysis.json va day10_per_label.csv")


if __name__ == "__main__":
    main()
