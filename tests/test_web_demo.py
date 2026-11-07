"""Web demo (src/rsna_knee/demo.py): ca tai len KHONG co train_series.csv, nen mat phang va
series nhay dich phai tu doan tu header DICOM. Doan sai thi model van ra 12 so - chi la
so cua mat phang khac. Test dung DICOM gia, khong can anh cuoc thi hay model."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from rsna_knee import demo  # noqa: E402

HUONG = {"Sagittal": [0, 1, 0, 0, 0, -1], "Coronal": [1, 0, 0, 0, 0, -1], "Axial": [1, 0, 0, 0, 1, 0]}


def _lat(series_uid: str, mo_ta: str, huong: list[int], so: int) -> bytes:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID, ds.SeriesDescription = series_uid, mo_ta
    ds.ImageOrientationPatient, ds.InstanceNumber = huong, so
    ds.Rows = ds.Columns = 8
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation, ds.SamplesPerPixel = 15, 0, 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = (np.arange(64, dtype=np.uint16).reshape(8, 8) * so).tobytes()
    b = io.BytesIO()
    ds.save_as(b, enforce_file_format=True)
    return b.getvalue()


def _zip(series: list[tuple[str, str]]) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, (pl, mo_ta) in enumerate(series):
            uid = generate_uid()
            for i in range(4):
                z.writestr(f"s{k}/{i}.dcm", _lat(uid, mo_ta, HUONG[pl], i + 1))
    return b.getvalue()


def test_doan_mat_phang_tu_huong_anh_va_uu_tien_nhay_dich():
    z = _zip([("Sagittal", "SAG T1 TSE"), ("Sagittal", "SAG PD FS"), ("Coronal", "COR T1"), ("Axial", "AX T2 FS")])
    theo_mat, tam, canh_bao = demo.tach_series_tai_len([("ca.zip", z)])
    try:
        assert sorted(theo_mat) == ["Axial", "Coronal", "Sagittal"]
        assert [d.mo_ta for d in theo_mat["Sagittal"]] == ["SAG PD FS", "SAG T1 TSE"], "nhay dich truoc, T1 sau"
        assert canh_bao == []
        x, n = demo._nap_81().doc_series(theo_mat["Axial"][0])     # file tach ra van doc duoc
        assert x.shape == (24, 256, 256) and n == 4
    finally:
        demo.don(tam)


def test_thieu_mat_phang_thi_canh_bao_chu_khong_im_lang():
    theo_mat, tam, canh_bao = demo.tach_series_tai_len([("ca.zip", _zip([("Sagittal", "SAG PD FS")]))])
    try:
        assert list(theo_mat) == ["Sagittal"]
        assert any("Coronal" in c for c in canh_bao) and any("Axial" in c for c in canh_bao)
    finally:
        demo.don(tam)
