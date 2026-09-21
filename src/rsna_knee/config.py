"""Hang so dung chung: nhan, duong dan, seed.

Moi module khac import tu day thay vi tu viet lai danh sach nhan. Ly do: thu tu 12 nhan
phai giong het nhau giua manifest, nhan yeu, head va submission - lech thu tu mot lan la
model hoc nham nhan ma khong bao loi o bat ky dau.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
DATA_MANIFEST = REPO_ROOT / "data" / "manifest"
REPORTS_DIR = REPO_ROOT / "reports"

# Thu tu nay la thu tu cot trong train.csv va sample_submission.csv. KHONG sap xep lai.
LABELS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

STUDY_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
REPORT_COL = "Report"

SEED = 42
N_FOLDS = 5
# Nhan hiem nhat trong 58 study co nhan nguoi gan -> dung de stratify.
STRATIFY_LABEL = "MCL"


def check_label_order(columns) -> None:
    """Chan som loi lech thu tu nhan giua cac file."""
    missing = [c for c in LABELS if c not in columns]
    if missing:
        raise ValueError(f"Thieu cot nhan: {missing}")


def short_uid(uid: str, n: int = 12) -> str:
    """Rut gon UID lam ten thu muc tren dia.

    Bat buoc tren Windows: duong dan goc cua Kaggle la
    `train_series/<64 ky tu>/<64 ky tu>/<64 ky tu>.dcm`. Cong voi thu muc repo la vuot gioi
    han MAX_PATH 260 ky tu - va loi bao ra la `FileNotFoundError` chu khong phai mot thong
    bao ve do dai duong dan, rat de chan doan nham thanh loi mang.

    12 ky tu cuoi cua UID la phan ngau nhien, du de phan biet trong pham vi mot dataset.
    Anh xa day du luu o `data/manifest/local_images.csv`.
    """
    return str(uid)[-n:]
