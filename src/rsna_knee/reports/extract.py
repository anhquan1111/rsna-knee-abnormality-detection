# -*- coding: utf-8 -*-
"""Bo rut nhan yeu tu bao cao chan doan (ngay 6).

Dau ra la BA trang thai cho moi cap (study, nhan): 1 / 0 / unknown. Day la diem khac biet
quan trong nhat so voi mot parser tu khoa thong thuong:

    "khong tim thay bang chung duong tinh" KHONG dong nghia voi "am tinh".

Neu ep unknown -> 0, moi nhan ma tu dien chua phu (vd: bien the tieng Phap) se bi gan am
tinh hang loat va model hoc mot quy luat sai ma khong co gi bao loi. Vi vay `unknown` duoc
giu nguyen la `NaN`, va viec co ep ve 0 hay khong la mot CAU HINH duoc do bang thuc nghiem
o ngay 8, khong phai mot gia dinh am tham.

Thuat toan cho moi cau:
  1. Tim `concept` (cau truc giai phau). Khong co -> cau nay khong noi gi ve nhan do.
  2. Tim `finding` (dau hieu benh ly). Voi nhan TYPE_B thi concept chinh la finding.
  3. Kiem phu dinh trong cua so `negation_window` ky tu NGAY TRUOC vi tri finding.
  4. Neu co concept nhung khong co finding, ma cau co tu khang dinh binh thuong
     ("intact", "sin alteraciones", "uredan") -> am tinh.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import LABELS, REPORT_COL, STUDY_COL
from .lexicon import LEXICON, MILD_CUES, NEGATION_CUES, NORMAL_CUES, TYPE_B

_FLAGS = re.IGNORECASE | re.UNICODE


def _compile(patterns) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), _FLAGS)


_NEG_RE = _compile(NEGATION_CUES)
_NORMAL_RE = _compile(NORMAL_CUES)
_MILD_RE = _compile(MILD_CUES)
_LABEL_RE = {
    name: {
        "concept": _compile(spec["concept"]),
        "finding": _compile(spec["finding"]) if spec["finding"] else None,
        "type": spec["type"],
    }
    for name, spec in LEXICON.items()
}

# Ngat cau: xuong dong, dau cham, cham phay, va cac dau bullet that su xuat hien trong
# du lieu (">" o dau dong, "1." dau muc trong phan Impression).
_SENT_SPLIT = re.compile(r"(?:\r?\n|(?<=[.;:])\s+|\s*>\s*)")


@dataclass(frozen=True)
class ExtractConfig:
    """Cac cong tac duoc do bang thuc nghiem o ngay 8, khong phai gia dinh co dinh."""

    negation_window: int = 60      # so ky tu truoc finding duoc coi la pham vi phu dinh
    mild_oa_negative: bool = False  # "mild chondropathy" tinh la am tinh hay duong tinh?
    unknown_as_negative: bool = False  # ep unknown -> 0 hay giu NaN?
    use_normal_cues: bool = True    # cho phep "intact"/"uredan" ket luan am tinh

    def tag(self) -> str:
        return (f"win{self.negation_window}"
                f"_mild{'N' if self.mild_oa_negative else 'P'}"
                f"_unk{'0' if self.unknown_as_negative else 'NaN'}"
                f"_norm{'on' if self.use_normal_cues else 'off'}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if s and s.strip()]


def _is_negated(sentence: str, start: int, window: int) -> bool:
    """Phu dinh co hieu luc khi tu phu dinh nam NGAY TRUOC dau hieu benh ly.

    Dung cua so thay vi "ca cau" vi bao cao y khoa hay ghep nhieu menh de:
      "No fracture is seen. ACL is intact. Horizontal tear at anterior horn..."
    Neu quet ca cau, tu "No" o dau se lat nham ket luan cua ve sau.
    """
    left = sentence[max(0, start - window):start]
    return _NEG_RE.search(left) is not None


def label_sentence(sentence: str, label: str, cfg: ExtractConfig) -> int | None:
    """Tra ve 1 (duong tinh), 0 (am tinh) hoac None (cau nay khong ket luan duoc)."""
    spec = _LABEL_RE[label]
    concept_hits = list(spec["concept"].finditer(sentence))
    if not concept_hits:
        return None

    if spec["type"] == TYPE_B:
        finding_hits = concept_hits          # ban than tu da la bat thuong
    else:
        finding_hits = list(spec["finding"].finditer(sentence)) if spec["finding"] else []

    saw_negated = False
    for hit in finding_hits:
        if _is_negated(sentence, hit.start(), cfg.negation_window):
            saw_negated = True
            continue
        if cfg.mild_oa_negative and label.endswith("OA") and _MILD_RE.search(sentence):
            saw_negated = True
            continue
        return 1

    if saw_negated:
        return 0
    if cfg.use_normal_cues and _NORMAL_RE.search(sentence):
        return 0
    return None


def extract_report(text: str, cfg: ExtractConfig | None = None) -> dict[str, float]:
    """Rut 12 nhan tu mot bao cao. Gia tri: 1.0, 0.0 hoac np.nan."""
    cfg = cfg or ExtractConfig()
    sentences = split_sentences(text)
    out: dict[str, float] = {}
    for label in LABELS:
        verdicts = [v for v in (label_sentence(s, label, cfg) for s in sentences) if v is not None]
        if 1 in verdicts:
            out[label] = 1.0        # mot cau khang dinh duong tinh thang moi cau am tinh
        elif verdicts:
            out[label] = 0.0
        else:
            out[label] = 0.0 if cfg.unknown_as_negative else np.nan
    return out


def extract_frame(df: pd.DataFrame, cfg: ExtractConfig | None = None) -> pd.DataFrame:
    """Rut nhan cho ca bang. Tra ve DataFrame co StudyInstanceUID + 12 cot nhan."""
    cfg = cfg or ExtractConfig()
    rows = [extract_report(t, cfg) for t in df[REPORT_COL]]
    out = pd.DataFrame(rows, columns=list(LABELS))
    out.insert(0, STUDY_COL, df[STUDY_COL].to_numpy())
    out["provenance"] = "machine"
    out["extractor_config"] = cfg.tag()
    return out


def explain(text: str, label: str, cfg: ExtractConfig | None = None) -> list[tuple[int, str]]:
    """Tra ve cac cau da quyet dinh ket luan - dung khi soi mot ca bi cham sai."""
    cfg = cfg or ExtractConfig()
    hits = []
    for s in split_sentences(text):
        v = label_sentence(s, label, cfg)
        if v is not None:
            hits.append((v, s))
    return hits
