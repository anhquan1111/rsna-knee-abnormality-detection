"""Test cho bo rut nhan tu bao cao.

Cac cau duoi day deu LAY TU DU LIEU THAT (train.csv cua RSNA), khong phai cau tu bia -
neu bia cau thi test chi xac nhan regex khop chinh no.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import LABELS, REPORT_COL, STUDY_COL
from rsna_knee.reports.evaluate import per_label_scores
from rsna_knee.reports.extract import (
    ExtractConfig,
    explain,
    extract_frame,
    extract_report,
    label_sentence,
    split_sentences,
)

CFG = ExtractConfig()


# ---------------------------------------------------------------- phu dinh
@pytest.mark.parametrize("cau, nhan", [
    ("No fracture is seen.", "Fracture"),
    ("Medial meniscus is not torn.", "Medial Meniscus"),
    ("Menisco medial de morfologia y senal conservada, sin signos de rotura.", "Medial Meniscus"),
    ("Degenerative signal throughout the lateral meniscus without surfacing tear.",
     "Lateral Meniscus"),
])
def test_phu_dinh_cho_ra_am_tinh(cau, nhan):
    assert label_sentence(cau, nhan, CFG) == 0


@pytest.mark.parametrize("cau, nhan", [
    ("ACL is intact.", "ACL"),
    ("The MCL is intact.", "MCL"),
    ("Anterior cruciate ligament (ACL): Normal with both bundles intact.", "ACL"),
])
def test_khang_dinh_binh_thuong_cho_ra_am_tinh(cau, nhan):
    """Khac phu dinh: khong co tu phu dinh nao, nhung van la bang chung am tinh."""
    assert label_sentence(cau, nhan, CFG) == 0


@pytest.mark.parametrize("cau, nhan", [
    ("Horizontal tear at anterior horn of the lateral meniscus is noted.", "Lateral Meniscus"),
    ("Extensive complete tearing of the body, posterior horn medial meniscus.",
     "Medial Meniscus"),
    ("Moderate joint effusion, distended suprapatellar bursa.", "Effusion"),
    ("Chronic reactive synovitis, suprapatellar bursitis", "Synovitis"),
    ("Leve derrame articular.", "Effusion"),
])
def test_duong_tinh_that_duoc_nhan_ra(cau, nhan):
    assert label_sentence(cau, nhan, CFG) == 1


def test_cau_khong_nhac_toi_thi_tra_ve_none():
    """Khong ket luan duoc KHAC voi am tinh - day la bản chat cua ba trang thai."""
    assert label_sentence("Alignment is anatomic.", "Fracture", CFG) is None
    assert label_sentence("MRI of the right knee was performed.", "ACL", CFG) is None


def test_cua_so_phu_dinh_khong_lat_nham_ve_sau_cua_cau():
    """Bay da do o ngay 6: quet ca cau thi `No` dau cau lat nham ket luan o ve cuoi."""
    doan = ("No fracture is seen. ACL is intact. "
            "Horizontal tear at anterior horn of the lateral meniscus is noted.")
    out = extract_report(doan, CFG)
    assert out["Fracture"] == 0.0
    assert out["ACL"] == 0.0
    assert out["Lateral Meniscus"] == 1.0, "ve cuoi bi lat nham boi 'No' o dau doan"


def test_mot_cau_duong_tinh_thang_moi_cau_am_tinh():
    """Bao cao mo ta roi moi ket luan o phan Impression -> duong tinh phai thang."""
    doan = "Medial meniscus appears normal. Impression: complete tear of the medial meniscus."
    assert extract_report(doan, CFG)["Medial Meniscus"] == 1.0


# ---------------------------------------------------------------- ba trang thai
def test_khong_ket_luan_duoc_tra_ve_nan_chu_khong_phai_0():
    out = extract_report("Exam Type: MRI KNEE RIGHT WO CONTRAST", CFG)
    assert np.isnan(out["Fracture"])
    assert all(np.isnan(v) for v in out.values())


def test_cau_hinh_ep_ve_0_doi_han_dau_ra():
    doan = "Exam Type: MRI KNEE RIGHT WO CONTRAST"
    ep0 = extract_report(doan, ExtractConfig(unknown_as_negative=True))
    assert all(v == 0.0 for v in ep0.values())
    assert not any(np.isnan(v) for v in ep0.values())


def test_tat_xu_ly_phu_dinh_lam_tang_duong_tinh_gia():
    """Bang chung so cua ngay 6, o dang test: tat phu dinh thi cau loai tru thanh duong tinh."""
    doan = "No fracture is seen. Medial meniscus is not torn."
    co = extract_report(doan, ExtractConfig())
    khong = extract_report(doan, ExtractConfig(negation_window=0, use_normal_cues=False))
    assert co["Fracture"] == 0.0 and co["Medial Meniscus"] == 0.0
    assert khong["Fracture"] == 1.0 and khong["Medial Meniscus"] == 1.0


# ---------------------------------------------------------------- hai kieu nhan
def test_nhan_kieu_concept_cong_finding_can_ca_hai():
    """Nhac ten day chang ma khong co dau hieu benh ly thi KHONG phai duong tinh."""
    assert label_sentence("The ACL and PCL are visualized.", "ACL", CFG) is None
    assert label_sentence("Complete tear of the ACL.", "ACL", CFG) == 1


def test_nhan_kieu_finding_la_chinh_no():
    """Effusion / Fracture: ban than tu da la bat thuong, chi can khong bi phu dinh."""
    assert label_sentence("Small joint effusion.", "Effusion", CFG) == 1
    assert label_sentence("No joint effusion.", "Effusion", CFG) == 0


def test_da_ngon_ngu():
    assert label_sentence("Ruptura completa del ligamento cruzado anterior.", "ACL", CFG) == 1
    assert label_sentence("Ruptur des vorderen Kreuzbandes.", "ACL", CFG) == 1
    assert label_sentence("On capraz bag yirtigi mevcuttur.", "ACL", CFG) == 1


# ---------------------------------------------------------------- tien ich
def test_split_sentences_tach_duoc_bullet_va_xuong_dong():
    doan = "> Dong mot\n> Dong hai. Cau ba; cau bon"
    assert split_sentences(doan) == ["Dong mot", "Dong hai.", "Cau ba;", "cau bon"]


def test_explain_chi_ra_dung_cau_da_quyet_dinh():
    doan = "ACL is intact. Complete tear of the medial meniscus."
    hits = explain(doan, "Medial Meniscus", CFG)
    assert len(hits) == 1
    assert hits[0][0] == 1 and "medial meniscus" in hits[0][1].lower()


def test_extract_frame_giu_dung_thu_tu_va_cot():
    df = pd.DataFrame({
        STUDY_COL: ["a", "b"],
        REPORT_COL: ["No fracture is seen.", "Acute fracture of the patella."],
    })
    out = extract_frame(df, CFG)
    assert list(out[STUDY_COL]) == ["a", "b"]
    assert list(out.columns)[1:13] == list(LABELS), "thu tu 12 nhan phai giu nguyen"
    assert out.loc[0, "Fracture"] == 0.0 and out.loc[1, "Fracture"] == 1.0
    assert (out["provenance"] == "machine").all()


def test_cham_diem_phan_biet_recall_da_tra_loi_va_recall_ke_ca_bo_sot():
    """Bay cua ngay 7: recall 1.0 tren ca da tra loi co the che mat rat nhieu ca bo sot."""
    gold = pd.DataFrame({STUDY_COL: list("abcd")})
    pred = pd.DataFrame({STUDY_COL: list("abcd")})
    for lab in LABELS:
        gold[lab] = [1.0, 1.0, 1.0, 0.0]
        pred[lab] = [1.0, np.nan, np.nan, 0.0]     # dung 1, bo qua 2, dung 1

    row = per_label_scores(gold, pred).iloc[0]
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall_known"] == pytest.approx(1.0), "nhin rat dep"
    assert row["recall_all"] == pytest.approx(1 / 3), "thuc te chi bat duoc 1/3"
    assert row["coverage"] == pytest.approx(0.5)
    assert row["unknown_but_positive"] == 2


def test_cham_diem_chan_khi_lech_thu_tu_study():
    gold = pd.DataFrame({STUDY_COL: ["a", "b"]})
    pred = pd.DataFrame({STUDY_COL: ["b", "a"]})
    for lab in LABELS:
        gold[lab] = [1.0, 0.0]
        pred[lab] = [1.0, 0.0]
    with pytest.raises(ValueError):
        per_label_scores(gold, pred)
