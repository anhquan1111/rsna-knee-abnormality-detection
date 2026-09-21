"""Evaluator macro AUC-ROC cho bai multi-label 12 nhan.

Viet tay thay vi goi thang `roc_auc_score(..., average='macro')` vi hai ly do do thuc
nghiem chi ra (xem tai lieu ngay 4):
  1. Khi mot nhan chi co mot lop trong tap danh gia, sklearn tra ve `nan` chu KHONG nem loi.
     Gop bang `np.nanmean` se ra mot so dep nhung am tham bo nhan khoi mau so.
  2. Ba quy uoc xu ly nhan mot lop (bo / cho 0.5 / cho 0.0) cho ba diem khac han nhau tren
     cung mot model. Diem macro khong kem mau so la mot con so vo nghia.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class MacroAUCResult:
    """Ket qua day du - luon mang theo mau so va danh sach nhan bi bo."""

    macro_auc: float
    n_scored: int
    per_label: dict[str, float] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        skip = f" | bo {len(self.skipped)}: {', '.join(self.skipped)}" if self.skipped else ""
        return f"macro AUC {self.macro_auc:.4f} tren {self.n_scored} nhan{skip}"

    def to_row(self, **extra) -> dict:
        row = {"macro_auc": self.macro_auc, "n_scored": self.n_scored,
               "n_skipped": len(self.skipped), "skipped": "|".join(self.skipped)}
        row.update({f"auc_{k}": v for k, v in self.per_label.items()})
        row.update(extra)
        return row


def macro_auc(y_true: np.ndarray, y_score: np.ndarray, labels) -> MacroAUCResult:
    """y_true, y_score: (n_study, n_label). y_score co the la logit - AUC khong doi qua sigmoid."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"Lech shape: y_true {y_true.shape} vs y_score {y_score.shape}")
    if y_true.shape[1] != len(labels):
        raise ValueError(f"y_true co {y_true.shape[1]} cot nhung nhan co {len(labels)} ten")

    per_label, skipped = {}, []
    for j, name in enumerate(labels):
        col = y_true[:, j]
        mask = ~np.isnan(col)          # nhan chua duoc gan thi khong tinh diem
        if mask.sum() == 0 or len(np.unique(col[mask])) < 2:
            skipped.append(name)
            continue
        per_label[name] = float(roc_auc_score(col[mask], y_score[mask, j]))

    if not per_label:
        raise ValueError("Khong nhan nao co du hai lop - tap danh gia nay khong cham duoc")
    return MacroAUCResult(float(np.mean(list(per_label.values()))), len(per_label), per_label, skipped)


def prevalence_baseline(y_true_train: np.ndarray, n_eval: int) -> np.ndarray:
    """Baseline ngu nhat: doan ti le duong tinh cua tap train cho moi study.

    Diem AUC cua no la 0.5 dung bang dinh nghia (moi study cung mot diem -> khong xep hang
    duoc). Gia tri cua baseline nay la de doi chieu accuracy: accuracy cua no rat cao vi
    nhieu nhan mat can bang, chung minh accuracy khong dung duoc cho bai nay.
    """
    rates = np.nanmean(y_true_train, axis=0)
    return np.tile(rates, (n_eval, 1))
