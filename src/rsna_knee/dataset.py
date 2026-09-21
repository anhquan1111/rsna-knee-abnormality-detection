"""`Dataset` tra ve MOT STUDY va `collate_fn` cho batch co so lat khac nhau (ngay 3).

Don vi cua `__getitem__` la mot study chu khong phai mot slice. Ly do: nhan nam o cap
study. Neu Dataset tra ve slice thi model se hoc "nhan cua ca ca" gan cho tung lat, va tap
validation se chua lat cua chinh nhung ca da co trong train - ro ri ngay o buoc dau.

Hau qua ky thuat: moi study co so lat khac nhau (19-36 lat trong tap da tai), nen
`default_collate` vo:
    RuntimeError: stack expects each tensor to be equal size,
                  but got [19, 224, 224] at entry 0 and [36, 224, 224] at entry 1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import LABELS, STUDY_COL, short_uid
from .dicom_io import load_series, normalize_series, resize_volume


class StudyDataset(Dataset):
    """Mot phan tu = mot study. Tra ve (pixels, labels, study_uid).

    `labels` co the chua NaN khi study chua duoc gan nhan - loss phai bo qua cac vi tri do,
    khong duoc coi la 0.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: str | Path,
        size: int = 224,
        cache_dir: str | Path | None = None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.size = size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def _series_dir(self, study_uid: str) -> Path:
        # Tren dia, thu muc mang UID rut gon - xem `config.short_uid` (MAX_PATH tren Windows).
        study_dir = self.image_root / short_uid(study_uid)
        if not study_dir.is_dir():
            raise FileNotFoundError(f"Chua tai anh cho study {study_uid} ({study_dir})")
        subdirs = sorted(d for d in study_dir.iterdir() if d.is_dir())
        if not subdirs:
            raise FileNotFoundError(f"Khong co series nao trong {study_dir}")
        return subdirs[0]

    def available(self) -> list[str]:
        """Cac study da co anh tren dia - dung de loc manifest truoc khi train."""
        return [uid for uid in self.manifest[STUDY_COL].astype(str)
                if (self.image_root / short_uid(uid)).is_dir()]

    def _load_pixels(self, study_uid: str) -> np.ndarray:
        # Cache la file .npy da chuan hoa + resize. Khoa cache PHAI chua moi tham so anh
        # huong toi noi dung, neu khong doi size ma van doc lai cache cu la sai am tham.
        if self.cache_dir:
            cached = self.cache_dir / f"{study_uid}_{self.size}.npy"
            if cached.exists():
                return np.load(cached)

        vol = load_series(self._series_dir(study_uid))
        arr = resize_volume(normalize_series(vol.pixels), self.size)

        if self.cache_dir:
            np.save(self.cache_dir / f"{study_uid}_{self.size}.npy", arr)
        return arr

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        study_uid = str(row[STUDY_COL])
        arr = self._load_pixels(study_uid)
        y = row[list(LABELS)].to_numpy(dtype=np.float32)
        return torch.from_numpy(arr), torch.from_numpy(y), study_uid


def collate_pad(batch):
    """Pad ve so lat lon nhat trong batch va tra ve mask.

    `mask[i, k] = True` nghia la lat k cua study i la lat THAT. Quen mask thi buoc pooling
    se tinh trung binh ca phan pad bang 0, lam loang dac trung cua nhung study it lat -
    sai lech nay khong bao loi va rat kho phat hien.
    """
    xs, ys, ids = zip(*batch)
    k_max = max(x.shape[0] for x in xs)
    padded = torch.stack([
        torch.cat([x, x.new_zeros(k_max - x.shape[0], *x.shape[1:])]) for x in xs
    ])
    mask = torch.stack([
        torch.arange(k_max) < x.shape[0] for x in xs
    ])
    return padded, mask, torch.stack(list(ys)), list(ids)


class FeatureDataset(Dataset):
    """Dung sau khi da trich feature: mot study = mot ma tran (n_slice, d).

    Tach rieng khoi `StudyDataset` vi vong train head chay hang tram epoch - doc lai DICOM
    moi epoch la lang phi. Feature cua ca dataset du nho de nam het trong RAM.
    """

    def __init__(self, features: dict[str, np.ndarray], manifest: pd.DataFrame):
        self.features = features
        self.manifest = manifest[manifest[STUDY_COL].isin(features)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        uid = str(row[STUDY_COL])
        x = torch.from_numpy(self.features[uid].astype(np.float32))
        y = torch.from_numpy(row[list(LABELS)].to_numpy(dtype=np.float32))
        return x, y, uid
