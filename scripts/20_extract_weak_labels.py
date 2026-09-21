"""Ngay 6-7: rut nhan yeu tu 4,407 bao cao va cham tren 58 study co nhan nguoi gan.

Chay:  python scripts/20_extract_weak_labels.py
       python scripts/20_extract_weak_labels.py --config win40_mildN
Ra:    data/interim/weak_labels.csv, reports/weak_label_scores.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rsna_knee.config import DATA_INTERIM, LABELS, REPORTS_DIR, STUDY_COL
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.reports.evaluate import (
    detect_languages,
    per_label_scores,
    per_language_coverage,
    summarize,
)
from rsna_knee.reports.extract import ExtractConfig, explain, extract_frame

# Cac cau hinh duoc so sanh - moi cai tat/bat DUNG MOT co che de quy duoc nguyen nhan.
CONFIGS = {
    "baseline": ExtractConfig(),
    "no_negation": ExtractConfig(negation_window=0, use_normal_cues=False),
    "narrow_window": ExtractConfig(negation_window=20),
    "wide_window": ExtractConfig(negation_window=200),
    "mild_oa_negative": ExtractConfig(mild_oa_negative=True),
    "unknown_as_zero": ExtractConfig(unknown_as_negative=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline", choices=list(CONFIGS))
    ap.add_argument("--skip-language", action="store_true")
    args = ap.parse_args()

    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold = gold_subset(man).reset_index(drop=True)

    print("=== 1. So sanh cac cau hinh tren 58 study co nhan nguoi gan ===")
    print(f"{'cau hinh':<18} {'F1(da tra loi)':>15} {'F1(ke ca bo sot)':>17} {'coverage':>9} "
          f"{'TP':>5} {'FP':>5} {'unk&pos':>8}")
    rows = []
    for name, cfg in CONFIGS.items():
        pred = extract_frame(gold, cfg)
        sc = per_label_scores(gold, pred)
        s = summarize(sc)
        rows.append({"config": name, **s})
        print(f"{name:<18} {s['macro_f1_known']:>15.3f} {s['macro_f1_all']:>17.3f} "
              f"{s['mean_coverage']:>9.1%} {s['total_TP']:>5} {s['total_FP']:>5} "
              f"{s['total_unknown_but_positive']:>8}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "weak_label_configs.csv", index=False)

    cfg = CONFIGS[args.config]
    print(f"\n=== 2. Chi tiet tung nhan - cau hinh '{args.config}' ({cfg.tag()}) ===")
    pred_gold = extract_frame(gold, cfg)
    scores = per_label_scores(gold, pred_gold)
    print(scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    scores.to_csv(REPORTS_DIR / "weak_label_scores.csv", index=False)

    print("\n=== 3. Vi sao khong dung so khop tu khoa tran ===")
    import re
    frac_re = re.compile(r"fractur", re.I)
    y = gold["Fracture"].to_numpy()
    mention = gold["Report"].map(lambda t: bool(frac_re.search(str(t)))).to_numpy()
    print(f"  Bao cao co chua chu 'fractur*' : {int(mention.sum())} / 58")
    print(f"    trong do gold Fracture=1     : {int((mention & (y == 1)).sum())}")
    print(f"    trong do gold Fracture=0     : {int((mention & (y == 0)).sum())}")
    print(f"  Gold Fracture=1 ma KHONG co chu: {int(((~mention) & (y == 1)).sum())}")
    prec_naive = (mention & (y == 1)).sum() / max(mention.sum(), 1)
    print(f"  -> precision cua 'co chu la duong tinh' = {prec_naive:.1%}")
    f_row = scores[scores.label == "Fracture"].iloc[0]
    print(f"  -> co xu ly phu dinh: precision {f_row['precision']:.1%}, "
          f"recall tren ca da tra loi {f_row['recall_known']:.1%}, "
          f"recall ke ca bo sot {f_row['recall_all']:.1%}")

    print("\n=== 4. Phu dinh dong gop bao nhieu? ===")
    a = summarize(per_label_scores(gold, extract_frame(gold, CONFIGS["baseline"])))
    b = summarize(per_label_scores(gold, extract_frame(gold, CONFIGS["no_negation"])))
    print(f"  co xu ly phu dinh : macro F1 {a['macro_f1_known']:.3f} | FP {a['total_FP']}")
    print(f"  tat xu ly phu dinh: macro F1 {b['macro_f1_known']:.3f} | FP {b['total_FP']}")
    print(f"  -> tat phu dinh lam FP tang {b['total_FP'] - a['total_FP']:+d}")

    if not args.skip_language:
        print("\n=== 5. Coverage theo ngon ngu (tren 58 gold) ===")
        langs = detect_languages(gold)
        print(per_language_coverage(gold, pred_gold, langs).to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))

        print("\n=== 6. Ngon ngu tren toan bo 4,407 bao cao ===")
        t0 = time.time()
        all_langs = detect_languages(man)
        vc = all_langs.value_counts()
        for lang, n in vc.items():
            n_gold = int((all_langs[man[STUDY_COL].isin(gold[STUDY_COL]).to_numpy()] == lang).sum())
            print(f"  {lang:<8} {n:>5,} bao cao ({n/len(man):>5.1%})  | co nhan nguoi gan: {n_gold}")
        print(f"  (nhan dien mat {time.time()-t0:.1f}s)")
        man["language"] = all_langs.to_numpy()

    print("\n=== 7. Rut nhan cho ca 4,407 study ===")
    t0 = time.time()
    pred_all = extract_frame(man, cfg)
    dt = time.time() - t0
    print(f"  xong trong {dt:.1f}s ({len(man)/dt:,.0f} bao cao/s)")
    known = pred_all[list(LABELS)].notna()
    print(f"  ti le cap (study, nhan) ket luan duoc : {known.to_numpy().mean():.1%}")
    print(f"  study co it nhat 1 nhan duong tinh    : {int((pred_all[list(LABELS)] == 1).any(axis=1).sum()):,}")
    print(f"  study khong ket luan duoc nhan nao    : {int((~known).all(axis=1).sum()):,}")

    print("\n  So nhan duong tinh may rut ra vs ti le tren gold:")
    for label in LABELS:
        mach = float((pred_all[label] == 1).mean())
        human = float(gold[label].mean())
        print(f"    {label:<18} may {mach:>6.1%} | nguoi {human:>6.1%} | lech {mach-human:>+6.1%}")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out = DATA_INTERIM / "weak_labels.csv"
    pred_all.to_csv(out, index=False)
    print(f"\nDa ghi {out} ({len(pred_all):,} dong)")

    worst = scores.sort_values("f1_all").iloc[0]
    print(f"\n=== 8. Nhan te nhat: {worst['label']} "
          f"(F1 ke ca bo sot {worst['f1_all']:.3f}) - soi mot ca sai ===")
    lab = worst["label"]
    bad = gold[(pred_gold[lab] == 1).to_numpy() & (gold[lab] == 0).to_numpy()]
    if len(bad):
        text = bad.iloc[0]["Report"]
        for verdict, sent in explain(text, lab, cfg)[:3]:
            print(f"  may cham {verdict} vi cau: {sent[:160]!r}")
        print(f"  nhung nguoi gan {lab}=0 cho ca nay")


if __name__ == "__main__":
    main()
