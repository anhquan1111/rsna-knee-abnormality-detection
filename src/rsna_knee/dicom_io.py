"""Doc DICOM va chuan hoa cuong do MRI (ngay 3).

Hai quyet dinh quan trong nhat trong file nay:

1. **Quet metadata phai dung `stop_before_pixels=True`.** Header doc duoc ma khong can giai
   nen anh. Quet ca nghin file de dung manifest ma giai nen tung file la lang phi hang chuc
   lan thoi gian.

2. **Chuan hoa percentile theo SERIES, khong theo slice va khong chia hang so.**
   Cuong do MRI khong tuyet doi: cung mot mo, hai may khac nhau cho hai dai gia tri khac
   nhau. Chia `/255` giu nguyen do lech do. Chuan hoa tung slice thi xoa mat tuong phan
   GIUA cac lat - ma tuong phan do chinh la thu bac si dung de doc benh.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_META_TAGS = (
    "StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber", "Manufacturer",
    "ManufacturerModelName", "Rows", "Columns", "PixelRepresentation",
    "MagneticFieldStrength", "SliceThickness", "PixelSpacing",
)


@dataclass
class SeriesVolume:
    """Mot series da duoc doc va sap xep theo thu tu lat cat."""

    study_uid: str
    series_uid: str
    pixels: np.ndarray          # (N, H, W)
    instance_numbers: list[int]
    transfer_syntax: str


def read_header(path: str | Path) -> dict:
    """Doc metadata mot file, khong giai nen pixel.

    Luon dung `getattr(..., default)`: du lieu da khu dinh danh thuong XOA HAN tag chu
    khong de trong, nen truy cap thang `ds.Manufacturer` se nem AttributeError giua vong
    quet hang nghin file.
    """
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    out = {tag: getattr(ds, tag, None) for tag in _META_TAGS}
    out["path"] = str(path)
    out["transfer_syntax"] = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
    out["transfer_syntax_name"] = getattr(
        getattr(ds.file_meta, "TransferSyntaxUID", None), "name", "unknown")
    return out


def scan_series_dir(series_dir: str | Path) -> list[dict]:
    """Quet toan bo header cua mot thu muc series."""
    return [read_header(p) for p in sorted(Path(series_dir).glob("*.dcm"))]


def load_series(series_dir: str | Path) -> SeriesVolume:
    """Doc ca series thanh mot khoi (N, H, W), sap theo `InstanceNumber`.

    Sap xep la bat buoc: thu tu file tren dia KHONG phai thu tu giai phau. Ten file la UID
    ngau nhien, nen `sorted(glob)` cho ra mot chong lat bi xao tron.
    """
    import pydicom

    paths = sorted(Path(series_dir).glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"Khong co .dcm trong {series_dir}")

    slices = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p))
        except pydicom.errors.InvalidDicomError as exc:
            # Bay that gap khi tai du lieu: file tai do dang (0 byte hoac cut giua chung) nem
            # dung loi nay. Thong bao goc chi noi "thieu DICM prefix" va goi y `force=True` -
            # lam theo goi y do la doc mot file rong thanh anh rac ma khong bao gi.
            raise ValueError(
                f"File DICOM hong hoac tai do dang: {p} ({p.stat().st_size} byte). "
                f"Kiem lai bang scripts/05_verify_images.py roi tai lai file nay. "
                f"KHONG dung force=True de bo qua."
            ) from exc
        inst = int(getattr(ds, "InstanceNumber", -1))
        slices.append((inst, ds))
    slices.sort(key=lambda t: t[0])

    first = slices[0][1]
    shapes = {(int(s.Rows), int(s.Columns)) for _, s in slices}
    if len(shapes) > 1:
        raise ValueError(f"Series co nhieu kich thuoc khac nhau: {shapes}")

    arr = np.stack([s.pixel_array for _, s in slices])
    return SeriesVolume(
        study_uid=str(first.StudyInstanceUID),
        series_uid=str(first.SeriesInstanceUID),
        pixels=arr,
        instance_numbers=[i for i, _ in slices],
        transfer_syntax=str(first.file_meta.TransferSyntaxUID),
    )


def normalize_series(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Chuan hoa percentile tren CA series -> float32 trong [0, 1].

    `+ 1e-8` khong phai trang tri: series toan mot gia tri (anh hong, anh den) cho
    `p_high == p_low` -> chia cho 0 -> ca mang thanh inf/nan va lan sang loss.
    """
    # `np.percentile` tra ve scalar float64. Tru mot float64 vao mang float32 se NANG
    # ca ket qua len float64 - cache to gap doi ma khong ai thay, vi ham van chay dung.
    # Ep tung he so ve float32 truoc khi tinh.
    p_low, p_high = (np.float32(v) for v in np.percentile(arr, (low, high)))
    out = np.clip(arr, p_low, p_high).astype(np.float32)
    return (out - p_low) / (p_high - p_low + np.float32(1e-8))


def normalize_per_slice(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Cach SAI, giu lai de do bang so o tai lieu ngay 3. Khong dung trong pipeline that."""
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        p_low, p_high = np.percentile(arr[i], (low, high))
        out[i] = (np.clip(arr[i], p_low, p_high) - p_low) / (p_high - p_low + 1e-8)
    return out


def resize_volume(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize (N, H, W) -> (N, size, size) bang PIL, sau khi da chuan hoa ve [0, 1].

    Thu tu quan trong: chuan hoa TRUOC roi resize. Lam nguoc lai thi phep noi suy tao ra
    gia tri moi o bien, keo percentile lech di.
    """
    from PIL import Image

    out = np.empty((arr.shape[0], size, size), dtype=np.float32)
    for i in range(arr.shape[0]):
        img = Image.fromarray(arr[i])
        out[i] = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32)
    return out
