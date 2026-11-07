"""Doc du doan out-of-fold ma KHONG luu nhan gold vao file.

File OOF (`reports/day13_epoch_*.npz`) duoc commit len repo cong khai. Nhan bac si cua 58
ca gold la du lieu cuoc thi, khong duoc phat tan lai - nen file chi chua du doan, so fold
va MOT ma bam cua thu tu ca. Luc cham, nhan duoc dung lai tu `data/raw/train.csv` o may,
theo dung phep chia fold da dung luc train, roi doi chieu ma bam: lech thu tu mot ca thi
dung lai, khong cham nham.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import LABELS, N_FOLDS, SEED, STUDY_COL
from .manifest import build_study_manifest, gold_subset, load_raw
from .splits import assign_folds


def ma_thu_tu(uid) -> str:
    """Ma bam cua thu tu ca - doi mot vi tri la doi ma."""
    return hashlib.md5("\n".join(map(str, uid)).encode("utf-8")).hexdigest()


def gold_theo_fold() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, fold, uid) cua 58 ca gold, dung thu tu ma 48_do_theo_epoch.py da train."""
    train, series = load_raw()
    gold = gold_subset(build_study_manifest(train, series)).reset_index(drop=True)
    folded = assign_folds(gold, n_splits=N_FOLDS, seed=SEED)
    return (folded[list(LABELS)].to_numpy(dtype=float), folded["fold"].to_numpy(),
            folded[STUDY_COL].to_numpy(dtype=str))


def nap_oof(path: Path, uid: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Tra ve `per_ep` (epoch, ca, nhan) sau khi kiem thu tu ca va fold khop."""
    with np.load(path) as z:
        if str(z["uid_md5"]) != ma_thu_tu(uid):
            raise SystemExit(f"{path.name}: thu tu ca khac voi phep chia hien tai - khong cham duoc")
        if not np.array_equal(z["fold"], fold):
            raise SystemExit(f"{path.name}: so fold khac")
        return z["per_ep"]


def khoa_thu_tu(y: np.ndarray) -> dict:
    """Thay mang nhan `y` bang ma bam thu tu ca + fold truoc khi ghi file OOF cong khai.

    Kiem `y` trung voi `gold_theo_fold()` - tuc file ghi dung thu tu ma luc cham se dung lai.
    Lech thi dung, khong ghi mot file ma sau nay khong ai cham dung duoc."""
    yg, fold, uid = gold_theo_fold()
    if y.shape != yg.shape or not np.array_equal(np.nan_to_num(y, nan=-1), np.nan_to_num(yg, nan=-1)):
        raise SystemExit("Thu tu ca khac gold_theo_fold() - khong ghi duoc file OOF khong nhan")
    return {"uid_md5": ma_thu_tu(uid), "fold": fold}


def nap_khong_nhan(path: Path, uid: np.ndarray) -> dict[str, np.ndarray]:
    """Cac mang du doan trong file OOF (khong nhan) sau khi kiem thu tu ca."""
    with np.load(path) as z:
        if str(z["uid_md5"]) != ma_thu_tu(uid):
            raise SystemExit(f"{path.name}: thu tu ca khac voi phep chia hien tai - khong cham duoc")
        return {k: z[k] for k in z.files if k not in ("uid_md5", "fold")}
