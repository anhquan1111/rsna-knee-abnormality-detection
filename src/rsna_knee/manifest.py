"""Dung manifest cap study va cap series tu CSV goc (+ header DICOM neu da co anh).

Manifest la mot file duy nhat lam nguon su that cho moi buoc sau: split, train, danh gia.
Khong module nao duoc doc thang `train.csv` de tu suy ra nhan.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_RAW, LABELS, REPORT_COL, SERIES_COL, STUDY_COL, check_label_order


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_RAW / "train.csv")
    series = pd.read_csv(DATA_RAW / "train_series.csv")
    check_label_order(train.columns)
    return train, series


def build_study_manifest(train: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Mot dong = mot study. Cot `provenance` phan biet nhan nguoi gan vs nhan may.

    `NaN` o cot nhan nghia la CHUA GAN, khong phai am tinh. Ep NaN -> 0 la loi nang nhat
    co the mac o buoc nay: no bien 4,349 ca chua biet thanh 4,349 ca "khong benh".
    """
    labelled = train[list(LABELS)].notna().all(axis=1)
    if not ((train[list(LABELS)].notna().sum(axis=1) % 12 == 0).all()):
        raise AssertionError("Co study gan nhan do dang - gia dinh 'du 12 hoac trong sach' sai")

    per_study = (
        series.groupby(STUDY_COL)
        .agg(
            n_series=(SERIES_COL, "size"),
            n_sagittal=("Anatomical_Plane", lambda s: int((s == "Sagittal").sum())),
            n_coronal=("Anatomical_Plane", lambda s: int((s == "Coronal").sum())),
            n_axial=("Anatomical_Plane", lambda s: int((s == "Axial").sum())),
            n_fluid_sensitive=("Fluid_Sensitive", "sum"),
        )
        .reset_index()
    )

    man = train[[STUDY_COL, REPORT_COL, *LABELS]].merge(per_study, on=STUDY_COL, how="left")
    man["provenance"] = pd.Series("unlabelled", index=man.index).mask(labelled, "human")
    man["n_positive"] = man[list(LABELS)].sum(axis=1, skipna=False)
    man["report_chars"] = man[REPORT_COL].astype(str).str.len()
    man["report_words"] = man[REPORT_COL].astype(str).str.split().str.len()
    return man


def gold_subset(manifest: pd.DataFrame) -> pd.DataFrame:
    """58 study co nhan nguoi gan - tap tham chieu duy nhat de cham bo rut nhan yeu."""
    return manifest[manifest["provenance"] == "human"].reset_index(drop=True)


def complete_local_studies() -> set[str]:
    """UID (day du) cua cac study da tai ve DU file theo `local_images.csv`.

    Can thiet vi viec tai duoc dung giua chung: study cuoi cung luon thieu lat. Train tren
    mot series thieu 10 lat khong nem loi nao - no chi lam ket qua te di mot cach kho hieu.
    """
    import pandas as pd

    from .config import DATA_MANIFEST, STUDY_COL

    csv = DATA_MANIFEST / "local_images.csv"
    if not csv.exists():
        return set()
    df = pd.read_csv(csv)
    df["have"] = [Path(p).exists() for p in df["local_path"]]
    per_study = df.groupby(STUDY_COL)["have"].agg(["sum", "size"])
    return set(per_study[per_study["sum"] == per_study["size"]].index)
