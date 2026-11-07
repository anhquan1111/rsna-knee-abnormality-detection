"""NGAY 15 - Nhan LLM tu gan dung toi dau? So voi nhan bac si tren 58 ca gold.

    python scripts/91_so_sanh_nhan.py

Bon nguon nhan, cung 58 ca, cung nhan bac si lam dap an:
    nhan LLM tu gan   scripts/90 (Gemini Flash, prompt cua minh)
    nhan LLM cong khai stevenleehans/rsna-knee-llm-report-labels, llm_labels_v4_blend.csv
    tu dien tu viet   src/rsna_knee/reports (chi ket luan duoc mot phan; o "chua ro" = 0.5)

Do ba kieu, vi "dung bao nhieu" co nhieu nghia:
  * AUC tung benh va macro AUC - kha nang xep ca co benh tren ca khong (khong can nguong);
  * nguong 0.5: % o (ca x benh) khop bac si, do nhay, do dac hieu - de doc nhu "dung X%";
  * do lech: o nao LLM noi nguoc bac si, dem theo benh.
KTC 95% bootstrap theo ca (2.000 lan), va KTC BAT CAP cho chenh tu gan - cong khai.

Them: hai bo LLM khop nhau toi dau tren 4.349 ca KHONG co nhan bac si (khong can dap an).

Ghi reports/day15_nhan_llm.json - CHI so tong hop, khong UID, khong van ban bao cao.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rsna_knee.config import LABELS, REPORTS_DIR
from rsna_knee.reports.extract import extract_frame

L = list(LABELS)
REPO = Path(__file__).resolve().parents[1]
NGUON = {
    "tu_gan": REPO / "data" / "nhan_llm_cua_minh" / "llm_labels_cua_minh.csv",
    "cong_khai": REPO / "data" / "external" / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv",
}
TEN = {"tu_gan": "LLM tự gắn (Gemini Flash)", "cong_khai": "LLM công khai", "tu_dien": "Từ điển tự viết"}


def _auc(y, s):
    """AUC = thong ke Mann-Whitney tren hang (co xu ly hoa) - bang roc_auc_score, nhanh hon
    nhieu lan nen bootstrap 2.000 lan chay trong vai giay."""
    r = rankdata(s)
    n1 = int(y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def auc_tung(y, p):
    return {k: float(_auc(y[:, j], p[:, j])) for j, k in enumerate(L) if 0 < y[:, j].sum() < len(y)}


def macro(y, p):
    return float(np.mean(list(auc_tung(y, p).values())))


def ktc(y, p, n=2000, seed=0):
    rng, b = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if all(0 < y[i, j].sum() < len(i) for j in range(len(L))):
            b.append(macro(y[i], p[i]))
    return [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])]


def ktc_cap(y, a, b, n=2000, seed=0):
    rng, d = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if all(0 < y[i, j].sum() < len(i) for j in range(len(L))):
            d.append(macro(y[i], b[i]) - macro(y[i], a[i]))
    return [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]


def nguong(y, p):
    """Nguong 0.5: % o khop, do nhay, do dac hieu (gop moi benh)."""
    d = p >= 0.5
    tp, tn = int((d & (y == 1)).sum()), int((~d & (y == 0)).sum())
    fp, fn = int((d & (y == 0)).sum()), int((~d & (y == 1)).sum())
    return {"khop": round((tp + tn) / y.size, 4), "do_nhay": round(tp / (tp + fn), 4),
            "do_dac_hieu": round(tn / (tn + fp), 4), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sai_tung_benh": {k: int(((p[:, j] >= 0.5) != (y[:, j] == 1)).sum()) for j, k in enumerate(L)}}


def main() -> None:
    train = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str})
    la_gold = train[L].notna().any(axis=1)
    gold = train[la_gold].set_index("StudyInstanceUID")
    uid = gold.index.tolist()
    y = gold[L].to_numpy(float)

    p = {k: pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID") for k, f in NGUON.items()}
    td = extract_frame(gold.reset_index()).set_index(gold.index)[L].to_numpy(float)
    phu_td = float(np.isfinite(td).mean())
    P = {k: v.loc[uid, L].to_numpy(float) for k, v in p.items()}
    P["tu_dien"] = np.where(np.isfinite(td), td, 0.5)

    kq = {"n_gold": len(uid), "do_phu_tu_dien_tren_gold": round(phu_td, 4), "nguon": {}}
    for k, pk in P.items():
        a = auc_tung(y, pk)
        kq["nguon"][k] = {"ten": TEN[k], "macro_auc": round(float(np.mean(list(a.values()))), 4),
                          "ci": ktc(y, pk), "tung_benh": {b: round(v, 3) for b, v in a.items()},
                          "nguong_05": nguong(y, pk)}
    kq["tu_gan_tru_cong_khai"] = {
        "chenh": round(kq["nguon"]["tu_gan"]["macro_auc"] - kq["nguon"]["cong_khai"]["macro_auc"], 4),
        "ci": ktc_cap(y, P["cong_khai"], P["tu_gan"])}
    kq["tu_gan_tru_tu_dien"] = {
        "chenh": round(kq["nguon"]["tu_gan"]["macro_auc"] - kq["nguon"]["tu_dien"]["macro_auc"], 4),
        "ci": ktc_cap(y, P["tu_dien"], P["tu_gan"])}

    # Hai bo LLM tren 4.349 ca ngoai gold - khong dap an, chi xem co doc bao cao giong nhau
    ngoai = train.loc[~la_gold, "StudyInstanceUID"]
    ngoai = ngoai[ngoai.isin(p["cong_khai"].index) & ngoai.isin(p["tu_gan"].index)]
    A, B = p["tu_gan"].loc[ngoai, L].to_numpy(float), p["cong_khai"].loc[ngoai, L].to_numpy(float)
    kq["hai_bo_llm_ngoai_gold"] = {
        "n_ca": len(ngoai),
        "khop_nguong_05": round(float(((A >= 0.5) == (B >= 0.5)).mean()), 4),
        "tuong_quan_hang": {k: round(float(pd.Series(A[:, j]).corr(pd.Series(B[:, j]), method="spearman")), 3)
                            for j, k in enumerate(L)},
        "ty_le_duong_05": {"tu_gan": {k: round(float((A[:, j] >= 0.5).mean()), 3) for j, k in enumerate(L)},
                           "cong_khai": {k: round(float((B[:, j] >= 0.5).mean()), 3) for j, k in enumerate(L)}},
    }
    (REPORTS_DIR / "day15_nhan_llm.json").write_text(json.dumps(kq, indent=2, ensure_ascii=False))

    print(f"58 ca gold. Tu dien ket luan duoc {phu_td:.1%} so o, o con lai tinh 0.5.\n")
    print(f"{'nguon':<28} {'macro AUC':>9}  {'KTC 95%':<18} {'khop@0.5':>8} {'nhay':>6} {'dac hieu':>8}")
    for k, v in kq["nguon"].items():
        t = v["nguong_05"]
        print(f"{v['ten']:<28} {v['macro_auc']:>9.4f}  {str(v['ci']):<18} {t['khop']:>8.1%} {t['do_nhay']:>6.1%} {t['do_dac_hieu']:>8.1%}")
    for k in ("tu_gan_tru_cong_khai", "tu_gan_tru_tu_dien"):
        print(f"\n{k}: {kq[k]['chenh']:+.4f}  KTC bat cap {kq[k]['ci']}")
    print(f"\n{'benh':<18} {'tu gan':>7} {'cong khai':>9} {'tu dien':>8} | sai@0.5 (tu gan / cong khai)")
    for b in L:
        n = kq["nguon"]
        print(f"{b:<18} {n['tu_gan']['tung_benh'].get(b, float('nan')):>7.3f} {n['cong_khai']['tung_benh'].get(b, float('nan')):>9.3f}"
              f" {n['tu_dien']['tung_benh'].get(b, float('nan')):>8.3f} | {n['tu_gan']['nguong_05']['sai_tung_benh'][b]:>3} / "
              f"{n['cong_khai']['nguong_05']['sai_tung_benh'][b]}")
    h = kq["hai_bo_llm_ngoai_gold"]
    print(f"\nHai bo LLM tren {h['n_ca']:,} ca ngoai gold: khop {h['khop_nguong_05']:.1%} o (nguong 0.5)")


if __name__ == "__main__":
    main()
