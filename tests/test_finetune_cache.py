"""Kiem cache anh cua ngay 9 - dac biet la duong GHI TIEP khi phien chay bi dut.

Vi sao dang co test rieng: cache nay mat ~35 phut de xay tren Kaggle. Neu doan ghi tiep
sai mot buoc `seek` thi du lieu lech offset - va no KHONG nem loi nao ca, chi la moi ca
tu do tro ve sau doc ra anh cua ca khac. Model van train, loss van giam, diem van ra -
chi la sai. Day dung la kieu hong am tham ma ca repo nay duoc viet de chan.

Khong can DICOM: thay `doc_mot_ca` bang mot ham sinh anh gia co the nhan dang duoc.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "60_kaggle_finetune.py"


def _load(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("ft_cache_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ft_cache_mod"] = mod
    spec.loader.exec_module(mod)
    mod.CACHE = tmp_path
    mod.OUT = tmp_path
    return mod


# Moi ca sinh ra mot khoi anh co gia tri hang = so thu tu cua no, nen doc ra la biet
# ngay co bi lech offset hay khong.
def _gia(n_lat: dict):
    def doc(root, rows):
        uid = rows.iloc[0]["StudyInstanceUID"]
        k = n_lat[uid]
        return np.full((k, 224, 224), float(uid[1:]), dtype=np.float32)
    return doc


def _chon(uids):
    return {u: pd.DataFrame([{"StudyInstanceUID": u, "SeriesInstanceUID": "s"}])
            for u in uids}


@pytest.fixture
def mod(tmp_path):
    return _load(tmp_path)


def test_cache_ghi_dung_noi_dung(mod, tmp_path):
    n_lat = {"u1": 3, "u2": 5, "u3": 2}
    mod.doc_mot_ca = _gia(n_lat)
    dat, idx = mod.xay_cache(None, _chon(["u1", "u2", "u3"]), "t")

    assert idx["xong"] and idx["n_study"] == 3 and idx["n_slice"] == 10
    mm = np.memmap(dat, dtype=np.float16, mode="r").reshape(-1, 224, 224)
    for uid, k in n_lat.items():
        start, n = idx["offsets"][uid]
        assert n == k
        assert np.all(np.asarray(mm[start:start + n]) == float(uid[1:])), \
            f"{uid} doc ra sai noi dung -> offset lech"


def test_lan_hai_khong_doc_lai(mod, tmp_path):
    n_lat = {"u1": 3, "u2": 5}
    mod.doc_mot_ca = _gia(n_lat)
    mod.xay_cache(None, _chon(["u1", "u2"]), "t")

    def no_goi(root, rows):
        raise AssertionError("da xong roi ma van doc lai DICOM")

    mod.doc_mot_ca = no_goi
    _, idx = mod.xay_cache(None, _chon(["u1", "u2"]), "t")
    assert idx["xong"] and idx["n_study"] == 2


def test_ghi_tiep_sau_khi_dut_giua_chung(mod, tmp_path):
    """Duong quan trong nhat: chi muc noi 'chua xong', phai ghi tiep dung offset."""
    n_lat = {"u1": 3, "u2": 5, "u3": 2, "u4": 4}
    mod.doc_mot_ca = _gia(n_lat)

    # Lan 1: chi co u1, u2 (gia lap phien chay dut sau 2 ca)
    dat, _ = mod.xay_cache(None, _chon(["u1", "u2"]), "t")
    idx_path = tmp_path / "t_index.json"
    d = json.loads(idx_path.read_text())
    d["xong"] = False
    idx_path.write_text(json.dumps(d))

    # Gia lap rac cua lan ghi dut: noi them byte thua vao cuoi file
    with open(dat, "ab") as f:
        f.write(b"\x00" * 1234)

    # Lan 2: day du 4 ca -> phai giu u1,u2 va ghi tiep u3,u4
    _, idx = mod.xay_cache(None, _chon(["u1", "u2", "u3", "u4"]), "t")

    assert idx["xong"] and idx["n_study"] == 4
    assert idx["n_slice"] == sum(n_lat.values())
    mm = np.memmap(dat, dtype=np.float16, mode="r").reshape(-1, 224, 224)
    assert mm.shape[0] == sum(n_lat.values()), "rac cua lan dut chua bi cat bo"
    for uid, k in n_lat.items():
        start, n = idx["offsets"][uid]
        assert n == k
        assert np.all(np.asarray(mm[start:start + n]) == float(uid[1:])), \
            f"{uid} lech sau khi ghi tiep"


def test_ca_hong_khong_lam_lech_offset(mod, tmp_path):
    """Ca doc khong duoc phai bi BO HAN, khong duoc chiem cho trong file."""
    n_lat = {"u1": 3, "u3": 2}

    def doc(root, rows):
        uid = rows.iloc[0]["StudyInstanceUID"]
        if uid == "u2":
            raise FileNotFoundError("gia vo hong")
        return np.full((n_lat[uid], 224, 224), float(uid[1:]), dtype=np.float32)

    mod.doc_mot_ca = doc
    dat, idx = mod.xay_cache(None, _chon(["u1", "u2", "u3"]), "t")

    assert idx["n_study"] == 2 and idx["n_hong"] == 1
    assert "u2" not in idx["offsets"]
    mm = np.memmap(dat, dtype=np.float16, mode="r").reshape(-1, 224, 224)
    for uid, k in n_lat.items():
        start, n = idx["offsets"][uid]
        assert np.all(np.asarray(mm[start:start + n]) == float(uid[1:]))


def test_doi_kich_thuoc_thi_lam_lai(mod, tmp_path):
    """Doi SIZE ma dung lai cache cu se doc ra anh sai hinh dang - phai lam lai tu dau."""
    mod.doc_mot_ca = _gia({"u1": 3})
    mod.xay_cache(None, _chon(["u1"]), "t")

    d = json.loads((tmp_path / "t_index.json").read_text())
    d["size"] = 196            # gia vo cache cu duoc xay o kich thuoc khac
    (tmp_path / "t_index.json").write_text(json.dumps(d))

    goi = []

    def doc(root, rows):
        goi.append(rows.iloc[0]["StudyInstanceUID"])
        return np.full((3, 224, 224), 1.0, dtype=np.float32)

    mod.doc_mot_ca = doc
    mod.xay_cache(None, _chon(["u1"]), "t")
    assert goi == ["u1"], "cache kich thuoc khac le ra phai bi xay lai"
