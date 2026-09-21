# -*- coding: utf-8 -*-
"""Cham chat luong nhan may tren gold set 58 study (ngay 7).

Nguyen tac khong duoc pha:
  * Chi 58 study co nhan nguoi gan. Do la THAM CHIEU DUY NHAT. Moi lan dung no de sua tu
    dien la mot lan no bot tinh doc lap - sau vai vong no khong con la tap test nua.
  * Bao cao theo TUNG NHAN, khong gop mot con so. Mot bo rut nhan co the rat tot o
    "Fracture" va vo hoan toan o "Lateral OA"; trung binh che mat dieu do.
  * Bao cao ca COVERAGE (ti le khong ket luan duoc). Mot bo doan bua it nhung bo qua 80%
    ca thi khong dung duoc de mo rong du lieu.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LABELS, REPORT_COL, STUDY_COL


def per_label_scores(gold: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """gold, pred: cung so dong, cung thu tu study. pred co the chua NaN (unknown)."""
    if not (gold[STUDY_COL].to_numpy() == pred[STUDY_COL].to_numpy()).all():
        raise ValueError("gold va pred khong cung thu tu study - phai merge truoc khi cham")

    rows = []
    for label in LABELS:
        y = gold[label].to_numpy(dtype=float)
        p = pred[label].to_numpy(dtype=float)
        known = ~np.isnan(p)

        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        missed_unknown = int((np.isnan(p) & (y == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) else np.nan
        # recall_known: chi tinh tren cac ca bo rut nhan DAM ket luan. Con so nay de dep
        # gia tao - mot bo chi tra loi khi chac chan se co recall gan 1.0.
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        # recall_all: coi "khong ket luan duoc" la bo sot. Day moi la con so dung de quyet
        # dinh co dung nhan may de mo rong du lieu hay khong.
        n_pos = int((y == 1).sum())
        rec_all = tp / n_pos if n_pos else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else np.nan
        f1_all = (2 * prec * rec_all / (prec + rec_all)
                  if (prec and rec_all and not np.isnan(prec) and not np.isnan(rec_all)) else np.nan)
        acc = (tp + tn) / known.sum() if known.sum() else np.nan

        rows.append({
            "label": label,
            "n_gold_pos": n_pos,
            "coverage": float(known.mean()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "unknown_but_positive": missed_unknown,
            "precision": prec, "recall_known": rec, "recall_all": rec_all,
            "f1_known": f1, "f1_all": f1_all, "accuracy_on_known": acc,
        })
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> dict:
    """Gop lai nhung KHONG giau mau so - luon kem so nhan tinh duoc."""
    f1 = scores["f1_known"].dropna()
    f1a = scores["f1_all"].dropna()
    return {
        "macro_f1_known": float(f1.mean()) if len(f1) else np.nan,
        "macro_f1_all": float(f1a.mean()) if len(f1a) else np.nan,
        "n_labels_with_f1": int(len(f1)),
        "mean_coverage": float(scores["coverage"].mean()),
        "total_TP": int(scores["TP"].sum()),
        "total_FP": int(scores["FP"].sum()),
        "total_FN": int(scores["FN"].sum()),
        "total_unknown_but_positive": int(scores["unknown_but_positive"].sum()),
    }


def detect_languages(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Doan ngon ngu tung bao cao. Chi de bao cao theo nhom, khong dung lam feature.

    Dung `langdetect`, vo tinh khong on dinh neu khong khoa seed - khoa lai de chay hai lan
    ra cung ket qua.
    """
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = seed
    out = []
    for text in df[REPORT_COL].astype(str):
        try:
            out.append(detect(text))
        except Exception:
            out.append("unknown")
    return pd.Series(out, index=df.index, name="language")


def per_language_coverage(gold: pd.DataFrame, pred: pd.DataFrame, languages: pd.Series) -> pd.DataFrame:
    """Coverage va do chinh xac tach theo ngon ngu - de lo cho tu dien chua phu."""
    rows = []
    pred_labels = pred[list(LABELS)].to_numpy(dtype=float)
    gold_labels = gold[list(LABELS)].to_numpy(dtype=float)
    for lang, idx in languages.groupby(languages).groups.items():
        pos = gold.index.get_indexer(idx)
        p, y = pred_labels[pos], gold_labels[pos]
        known = ~np.isnan(p)
        correct = (p == y) & known
        rows.append({
            "language": lang,
            "n_reports": len(pos),
            "coverage": float(known.mean()),
            "accuracy_on_known": float(correct.sum() / known.sum()) if known.sum() else np.nan,
            "n_gold_pos": int(np.nansum(y)),
        })
    return pd.DataFrame(rows).sort_values("n_reports", ascending=False).reset_index(drop=True)
