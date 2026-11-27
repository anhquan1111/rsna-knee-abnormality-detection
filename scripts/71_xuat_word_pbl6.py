"""Xuat bao cao Word PBL6 theo khuon cua truong (lay file bao cao PBL5 lam goc: style, bia, logo).

    uv run --extra bao-cao python scripts/71_xuat_word_pbl6.py          # -> reports/pbl6/Bao_Cao_PBL6.docx
    uv run --extra bao-cao python scripts/71_xuat_word_pbl6.py --pdf    # them PDF qua Word tren Windows

Thong tin sinh vien, GVHD, ten hoc phan, duong dan file mau va cac cau trich bao cao vi du (Bang 2.1:
[cau, nghia, nhan|null]) nam o reports/pbl6/thong_tin.json.
Thu muc reports/pbl6/ KHONG len git (repo cong khai: bao cao co ten, MSSV va anh MRI cua cuoc thi).

So lieu doc tu reports/slide/slide.html (khoi `const D = {...}` do scripts/70 tinh tu reports/*.json)
va reports/day15_*.json, khong go tay; vai so do trong nhat ky (thoi gian chay, ty le chon series) nam
o SO_NHAT_KY kem nguon. Hinh ve bang matplotlib; anh web demo chup bang Chrome headless khi web dang
chay o 127.0.0.1:8790 (khong chay thi dung anh cu neu co).

Trich dan kieu IEEE: viet {r:khoa} trong bai, so [n] tu danh theo thu tu xuat hien lan dau, danh muc
cuoi bai chi gom muc da trich. Hinh/bang viet {h:khoa}/{b:khoa}, so tu danh theo chuong.

--pdf: chep docx sang thu muc tam cua Windows, Word (an, khong hien cua so) cap nhat muc luc, luu lai,
xuat PDF va dem trang, roi chep ca hai ve reports/pbl6/.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

REPO = Path(__file__).resolve().parents[1]
RA = REPO / "reports" / "pbl6"
HINH = RA / "hinh"
THONG_TIN = RA / "thong_tin.json"
DOCX = RA / "Bao_Cao_PBL6.docx"
CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
WEB = "http://127.0.0.1:8790/?mau=mau1"

# So do trong nhat ky va README (khong co trong file ket qua JSON)
SO_NHAT_KY = {
    "dung_luong_gb": 569,          # README.vi.md: toan bo DICOM cua cuoc thi
    "chon_series": "96,9",         # % ca co series nhay dich o ca 3 mat phang (cache notebook 81)
    "n_val_llm": 212,              # ca nhan LLM giu lai lam val trong notebook 81 (5%)
    "dp_giay": "10,7", "mot_gpu_giay": "0,47",   # kernel chan doan nn.DataParallel (ngay 14)
    "gio_kaggle": "3,1",           # lan chay 2 model song song
    "hang_may": "82,8",            # dac trung DINOv2 doan dung hang may (ngay 10)
    "auc_journey_1mp": 0.5256,     # day8_experiment.csv: DINOv2, 58 ca, 1 mat phang
    "n_test": 116,                 # uv run pytest --co
    "web_giay": "0,2",             # thoi gian suy luan mot ca tren GPU (web demo)
    "n_file_tai": "1.500",         # README.vi.md: tai thu ~1.500 file .dcm + >4.000 request liet ke ...
    "n_lenh_liet_ke": "4.000",
    "giay_cho_429": "179.280",     # ... thi bi HTTP 429, retry-after 179280 s (~50 gio)
    "chi_dung_tap": "96,9",        # = chon_series (cache notebook 81)
}

TEN_VI = {
    "ACL": "Dây chằng chéo trước (ACL)", "MCL": "Dây chằng bên trong (MCL)",
    "Medial Meniscus": "Sụn chêm trong", "Lateral Meniscus": "Sụn chêm ngoài",
    "Medial OA": "Thoái hoá khoang trong", "Lateral OA": "Thoái hoá khoang ngoài",
    "PF OA": "Thoái hoá bánh chè – đùi", "Effusion": "Tràn dịch khớp",
    "Synovitis": "Viêm màng hoạt dịch", "Baker's": "Nang Baker",
    "Contusion": "Dập xương", "Fracture": "Gãy xương",
}

TAI_LIEU = {
    # Chu nhan: bao cao khong gioi thieu day la cuoc thi, chi mo ta du lieu -> khong ghi trang cuoc thi
    "rsna": 'Radiological Society of North America (RSNA), "RSNA Knee Abnormality Detection," bộ dữ liệu '
            "MRI khớp gối kèm báo cáo chẩn đoán, 2026.",
    "mrnet": 'N. Bien et al., "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: '
             'Development and retrospective validation of MRNet," PLOS Medicine, vol. 15, no. 11, '
             "Art. no. e1002699, 2018.",
    "snorkel": 'A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Ré, "Snorkel: Rapid '
               'training data creation with weak supervision," Proc. VLDB Endowment, vol. 11, no. 3, '
               "pp. 269–282, 2017.",
    "llmlabels": 'stevenleehans, "RSNA Knee LLM Report Labels," Kaggle dataset. [Online]. Available: '
                 "https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels. "
                 "[Accessed: Nov. 2026].",
    "gemini": 'Gemini Team, Google, "Gemini: A family of highly capable multimodal models," '
              "arXiv:2312.11805, 2023.",
    "dinov2": 'M. Oquab et al., "DINOv2: Learning robust visual features without supervision," '
              "Transactions on Machine Learning Research, 2024.",
    "ilse": 'M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance '
            'learning," in Proc. 35th Int. Conf. Machine Learning (ICML), 2018, pp. 2127–2136.',
    "effnet": 'M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural '
              'networks," in Proc. 36th Int. Conf. Machine Learning (ICML), 2019, pp. 6105–6114.',
    "effv2": 'M. Tan and Q. V. Le, "EfficientNetV2: Smaller models and faster training," in Proc. '
             "38th Int. Conf. Machine Learning (ICML), 2021, pp. 10096–10106.",
    "timm": 'R. Wightman, "PyTorch Image Models," GitHub repository, 2019. [Online]. Available: '
            "https://github.com/huggingface/pytorch-image-models",
    "hanley": 'J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver '
              'operating characteristic (ROC) curve," Radiology, vol. 143, no. 1, pp. 29–36, 1982.',
    "efron": "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: "
             "Chapman & Hall, 1993.",
    "pytorch": 'A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning '
               'library," in Advances in Neural Information Processing Systems 32 (NeurIPS), 2019, '
               "pp. 8024–8035.",
    "pydicom": 'D. Mason, "SU-E-T-33: Pydicom: An open source DICOM library," Medical Physics, '
               "vol. 38, no. 6, p. 3493, 2011.",
    "sklearn": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine '
               "Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "adamw": 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. Int. '
             "Conf. Learning Representations (ICLR), 2019.",
    "fastapi": 'S. Ramírez, "FastAPI," 2018. [Online]. Available: https://fastapi.tiangolo.com',
    # 6 muc duoi da doi chieu tac gia, noi dang, trang voi trang xuat ban/PubMed/arXiv (25/11/2026)
    "kneemri": 'I. Štajduhar, M. Mamula, D. Miletić, and G. Ünal, "Semi-automated detection of anterior '
               'cruciate ligament injury from MRI," Computer Methods and Programs in Biomedicine, vol. 140, '
               "pp. 151–164, 2017.",
    "elnet": 'C.-H. Tsai, N. Kiryati, E. Konen, I. Eshed, and A. Mayer, "Knee injury detection using MRI with '
             'efficiently-layered network (ELNet)," in Proc. 3rd Conf. Medical Imaging with Deep Learning '
             "(MIDL), PMLR 121, 2020, pp. 784–794.",
    "chestxray8": 'X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-ray8: Hospital-scale '
                  "chest X-ray database and benchmarks on weakly-supervised classification and localization of "
                  'common thorax diseases," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), '
                  "2017, pp. 2097–2106.",
    "chexpert": 'J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert '
                'comparison," in Proc. AAAI Conf. Artificial Intelligence, vol. 33, 2019, pp. 590–597.',
    "chexbert": 'A. Smit, S. Jain, P. Rajpurkar, A. Pareek, A. Y. Ng, and M. P. Lungren, "CheXbert: Combining '
                "automatic labelers and expert annotations for accurate radiology report labeling using BERT,\" "
                "in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 1500–1519.",
    "gpt4bc": 'L. C. Adams et al., "Leveraging GPT-4 for post hoc transformation of free-text radiology reports '
              'into structured reporting: A multilingual feasibility study," Radiology, vol. 307, no. 4, '
              "Art. no. e230725, 2023.",
}


# ============================================================ so lieu
def nap_so_lieu() -> dict:
    html = (REPO / "reports" / "slide" / "slide.html").read_text(encoding="utf-8")
    m = re.search(r"const D = (\{.*?\});\n", html, re.S)
    if not m:
        raise SystemExit("Khong thay so lieu trong reports/slide/slide.html - chay scripts/70 --slide truoc")
    D = json.loads(m.group(1))
    D["llm"] = json.loads((REPO / "reports" / "day15_nhan_llm.json").read_text(encoding="utf-8"))
    D["cuoi_day"] = json.loads((REPO / "reports" / "day15_model_cuoi.json").read_text(encoding="utf-8"))
    return D


def f(x: float, n: int = 3) -> str:
    """So thap phan kieu Viet: 0,896."""
    return f"{x:.{n}f}".replace(".", ",")


def ci(c, n: int = 3) -> str:
    return f"[{f(c[0], n)}; {f(c[1], n)}]"


def ci_dau(c, n: int = 3) -> str:
    g = lambda x: ("+" if x >= 0 else "−") + f(abs(x), n)
    return f"[{g(c[0])}; {g(c[1])}]"


def pt(x: float, n: int = 1) -> str:
    return f(100 * x, n) + "%"


# ============================================================ hinh
def chuan_bi_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for ten in ("times.ttf", "timesbd.ttf", "timesi.ttf", "arial.ttf", "arialbd.ttf"):
        p = Path("/mnt/c/Windows/Fonts") / ten
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight"})
    return plt


XAM, XANH, CAM, LUC = "#7f8c99", "#2b6cd4", "#e0662b", "#1f9d6a"


def ve_hop(ax, x, y, w, h, chu, mau="#eef3fb", vien=XANH, dam=False, co=10):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=mau, ec=vien, lw=1.2))
    ax.text(x + w / 2, y + h / 2, chu, ha="center", va="center", fontsize=co,
            fontweight="bold" if dam else "normal", linespacing=1.3)


def mui_ten(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1, shrinkA=0, shrinkB=0))


def hinh_tong_quan(plt, D, ra: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.75))
    ax.set_xlim(0, 10), ax.set_ylim(0, 5.6), ax.axis("off")
    n_khong = D["du_lieu"]["n_study"] - D["du_lieu"]["n_gold"]
    so = lambda n: f"{n:,}".replace(",", ".")
    # nhanh nhan
    ax.text(0.1, 5.35, "Nhánh nhãn", fontsize=9, fontweight="bold", color="#555")
    ve_hop(ax, 0.1, 4.2, 2.6, 0.95, f"Báo cáo chẩn đoán\n{so(n_khong)} ca, 9 ngôn ngữ", co=8)
    ve_hop(ax, 3.1, 4.2, 2.6, 0.95, "Từ điển luật → LLM\nđọc báo cáo", co=8)
    ve_hop(ax, 6.1, 4.2, 3.8, 0.95, "Nhãn mềm 0–1 cho 12 bệnh\n(nhãn yếu để huấn luyện)", co=8)
    mui_ten(ax, 2.7, 4.67, 3.1, 4.67), mui_ten(ax, 5.7, 4.67, 6.1, 4.67)
    # nhanh anh
    ax.text(0.1, 3.75, "Nhánh ảnh", fontsize=9, fontweight="bold", color="#555")
    ve_hop(ax, 0.1, 2.45, 2.6, 1.1, f"DICOM 3 mặt phẳng\n{so(D['du_lieu']['n_series'])} series", co=8)
    ve_hop(ax, 3.1, 2.45, 2.6, 1.1, "Chọn series nhạy dịch,\nchuẩn hoá, 24 lát\n× 256 px", co=8)
    mui_ten(ax, 2.7, 3.0, 3.1, 3.0)
    ve_hop(ax, 6.1, 3.05, 3.8, 0.6, f"GĐ 1: DINOv2 đóng băng → {f(D['seed']['gop'])}",
           mau="#f1f2f4", vien=XAM, co=8)
    ve_hop(ax, 6.1, 2.25, 3.8, 0.6, "GĐ 2: CNN 2.5D, 6 model", mau="#e8f6ef", vien=LUC, co=8)
    mui_ten(ax, 5.7, 3.0, 6.1, 3.35), mui_ten(ax, 5.7, 3.0, 6.1, 2.55)
    mui_ten(ax, 8.0, 4.2, 8.0, 3.65)
    # gop va cham
    ve_hop(ax, 6.1, 1.05, 3.8, 0.8, f"Gộp theo hạng → 12 xác suất\nmacro AUC {f(D['cuoi']['nhom']['gop_6_cnn']['auc'])}",
           mau="#e8f6ef", vien=LUC, dam=True, co=8)
    mui_ten(ax, 8.0, 2.25, 8.0, 1.85)
    ve_hop(ax, 0.1, 1.05, 5.6, 0.8, f"{D['du_lieu']['n_gold']} ca bác sĩ gán nhãn: GĐ 2 không dùng để huấn luyện,\n"
           "chỉ dùng để chấm (macro AUC, KTC bootstrap 95%)", mau="#fdf1ea", vien=CAM, co=8)
    mui_ten(ax, 5.7, 1.45, 6.1, 1.45)
    ve_hop(ax, 6.1, 0.1, 3.8, 0.6, "Web demo cục bộ (127.0.0.1)", mau="#f1f2f4", vien=XAM, co=8)
    mui_ten(ax, 8.0, 1.05, 8.0, 0.7)
    fig.savefig(ra), plt.close(fig)


def hinh_kien_truc(plt, ra: Path):
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.set_xlim(0, 10), ax.set_ylim(0, 3.2), ax.axis("off")
    hop = [
        (0.0, "Một ca MRI:\n3 mặt phẳng\n× 24 lát\n× 256 px"),
        (2.0, "Cửa sổ\n3 lát liền kề\n= ảnh 3 kênh\n(2.5D)"),
        (4.0, "EfficientNet\n(ImageNet)\n→ 1 vector\nmỗi cửa sổ"),
        (6.0, "Attention\nriêng từng\nbệnh trên mọi\ncửa sổ (MIL)"),
        (8.0, "12 logit\n→ sigmoid\n→ 12 xác suất"),
    ]
    for x, chu in hop:
        ve_hop(ax, x + 0.05, 0.85, 1.75, 2.1, chu, co=8)
    for x, _ in hop[:-1]:
        mui_ten(ax, x + 1.8, 1.9, x + 2.05, 1.9)
    ax.text(5, 0.3, "Huấn luyện: lấy ngẫu nhiên 8 cửa sổ mỗi mặt phẳng. Suy luận: mỗi lát làm tâm một cửa sổ.",
            ha="center", fontsize=8, style="italic", color="#444")
    fig.savefig(ra), plt.close(fig)


def hinh_hanh_trinh(plt, D, ra: Path):
    nn, ce, cn, cf = D["nguon_nhan"], D["chon_epoch"], D["cnn"], D["cuoi"]["nhom"]["gop_6_cnn"]
    buoc = [
        ("Đoán\nmò", 0.5, None, XAM), ("DINOv2\n58 ca,\n1 mặt\nphẳng", SO_NHAT_KY["auc_journey_1mp"], None, XAM),
        ("+ 3 mặt\nphẳng", nn["A"], None, XAM), ("+ nhãn\ntừ điển", nn["B"], None, XAM),
        ("+ nhãn\nLLM", nn["D"], None, XAM), ("+ atten-\ntion\ntừng\nbệnh", D["thanh_phan"][3]["cv_cu"], None, XAM),
        ("5 head\n(đo cũ)", ce["cv_cu_bao_cao"], None, CAM),
        ("5 head\n× 3 seed\n(đo lại)", D["seed"]["gop"], D["seed"]["ci_gop"], XANH),
        ("CNN\n2.5D,\n2 model", cn["auc"], cn["ci"], LUC), ("Cuối:\n6 CNN", cf["auc"], cf["ci"], LUC),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.95))
    for i, (ten, v, c, mau) in enumerate(buoc):
        ax.bar(i, v, color=mau, width=0.66)
        if c:
            ax.errorbar(i, v, yerr=[[v - c[0]], [c[1] - v]], color="#222", capsize=3, lw=0.9)
        ax.text(i, (c[1] if c else v) + 0.008, f(v), ha="center", fontsize=8)
    ax.set_xticks(range(len(buoc)), [b[0] for b in buoc], fontsize=8)
    ax.set_ylim(0.45, 0.96), ax.set_ylabel("Macro AUC (58 ca bác sĩ)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(lambda x, _: f(x, 2))
    fig.savefig(ra), plt.close(fig)


def hinh_epoch(plt, D, ra: Path):
    fig, ax = plt.subplots(figsize=(6.2, 2.75))
    ten = {"efficientnet_b3": "EfficientNet-B3", "tf_efficientnetv2_b2": "EfficientNetV2-B2"}
    for mh, mau in zip(D["cnn"]["mo_hinh"], (XANH, LUC)):
        x = range(1, len(mh["gold"]) + 1)
        ax.plot(x, mh["gold"], color=mau, lw=1.6, label=f"{ten[mh['backbone']]}: 58 ca bác sĩ")
        ax.plot(x, mh["val_llm"], color=mau, lw=1.3, ls="--",
                label=f"{ten[mh['backbone']]}: {SO_NHAT_KY['n_val_llm']} ca nhãn LLM")
    ax.axvline(12, color="#999", lw=0.8, ls=":")
    ax.text(11.85, 0.70, "epoch chốt (cuối)", ha="right", fontsize=8, color="#555")
    ax.set_xlabel("Epoch", fontsize=9), ax.set_ylabel("Macro AUC (trọng số EMA)", fontsize=9)
    ax.set_xticks(range(1, 13)), ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(lambda x, _: f(x, 2))
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(ra), plt.close(fig)


def hinh_tung_benh(plt, D, ra: Path):
    dino = {r["nhan"]: r["dino"] for r in D["cnn"]["tung_nhan"]}
    cuoi = D["cuoi"]["tung_benh"]
    thu_tu = sorted(cuoi, key=lambda k: cuoi[k] - dino[k])
    fig, ax = plt.subplots(figsize=(6.2, 3.25))
    for i, k in enumerate(thu_tu):
        ax.plot([dino[k], cuoi[k]], [i, i], color="#bbb", lw=2, zorder=1)
        ax.scatter(dino[k], i, color=XAM, s=26, zorder=2, label="DINOv2 đóng băng (GĐ 1)" if i == 0 else None)
        ax.scatter(cuoi[k], i, color=LUC, s=26, zorder=2, label="Model cuối, gộp 6 CNN" if i == 0 else None)
        d = cuoi[k] - dino[k]
        ax.text(1.01, i, ("+" if d >= 0 else "−") + f(abs(d)), va="center", fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(thu_tu)), [TEN_VI[k] for k in thu_tu], fontsize=8.5)
    ax.set_xlim(0.6, 1.0), ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUC trên 58 ca bác sĩ", fontsize=9), ax.tick_params(axis="x", labelsize=8)
    ax.xaxis.set_major_formatter(lambda x, _: f(x, 1))
    ax.legend(fontsize=8.5, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
    fig.savefig(ra), plt.close(fig)


# ------------------------------------------------------------ chuong du lieu
def thong_ke_du_lieu() -> dict:
    """So lieu kham pha du lieu, tinh lai tu cac bang o data/ (khong go tay).

    competition_files.csv la danh sach file lay qua API truoc khi tai khoan bi chan (khong du het
    24.371 series), nen so lat moi series tinh tren phan da liet ke va ghi ro co mau."""
    import pandas as pd
    dl = REPO / "data"
    L = list(TEN_VI)
    s = pd.read_csv(dl / "raw" / "train_series.csv")
    hd = pd.read_csv(dl / "manifest" / "dicom_headers.csv")
    ds = pd.read_csv(dl / "manifest" / "competition_files.csv")
    ds = ds[ds.name.str.startswith("train_series/")].name.str.split("/", expand=True)
    lat = ds.groupby(2).size()
    sm = pd.read_csv(dl / "interim" / "study_manifest.csv")
    gold = sm[sm.provenance == "human"]
    llm = pd.read_csv(dl / "external" / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv")
    vd = pd.read_csv(dl / "raw" / "train.csv").iloc[0]   # bao cao vi du: dong dau cua train.csv
    llm_vd = llm.set_index("StudyInstanceUID").loc[vd.StudyInstanceUID]
    llm = llm[~llm.StudyInstanceUID.isin(gold.StudyInstanceUID)]
    tt = s.groupby("StudyInstanceUID").size()
    tu = hd.MagneticFieldStrength.round(1)
    return {
        "series_moi_ca": tt.value_counts().sort_index().to_dict(), "series_mode_ti_le": float((tt == 5).mean()),
        "series_min": int(tt.min()), "series_max": int(tt.max()),
        "mat_phang": {pl: {"tong": int((s.Anatomical_Plane == pl).sum()),
                           "nhay": int(((s.Anatomical_Plane == pl) & (s.Fluid_Sensitive == 1)).sum())}
                      for pl in ("Sagittal", "Coronal", "Axial")},
        "nhay_dich": float(s.Fluid_Sensitive.mean()),
        "hang": pd.read_csv(dl / "manifest" / "study_manufacturer.csv").manufacturer_norm.value_counts().to_dict(),
        "n_header": len(hd), "tu_truong": tu.value_counts().to_dict(),
        "rows": hd.Rows.to_numpy(), "day_lat": hd.SliceThickness.to_numpy(),
        "lat": lat.to_numpy(), "n_series_liet_ke": len(lat),
        "so_tu": sm.report_words.to_numpy(),
        "ty_le_gold": {k: float(gold[k].mean()) for k in L},
        "ty_le_llm": {k: float((llm[k] > 0.5).mean()) for k in L}, "n_llm": len(llm),
        "duong_moi_ca_gold": float(gold[L].sum(axis=1).mean()),
        "gold_khong_benh": int((gold[L].sum(axis=1) == 0).sum()),
        "vi_du": {"bao_cao": vd.Report, "llm": {k: float(llm_vd[k]) for k in L},
                  "co_nhan_bac_si": bool(vd[L].notna().any())},
    }


def nap_ca_mau(k: int = 7):
    """Mot ca gold dong goi cho web demo (khong ca nao vao train CNN): lat goc va khoi da chuan hoa.

    Doc dung duong cua web demo (src/rsna_knee/demo.py) va ham doc_series cua notebook train, nen
    hinh minh hoa la dung du lieu mo hinh nhin thay."""
    sys.path.insert(0, str(REPO / "src"))
    import numpy as np
    import pydicom

    from rsna_knee import demo
    z = REPO / "data" / "web_demo" / "ca_mau" / f"ca_mau_{k}.zip"
    series, tam, _ = demo.tach_series_tai_len([(z.name, z.read_bytes())])
    nb = demo._nap_81()
    ra = {}
    try:
        for pl in ("Sagittal", "Coronal", "Axial"):
            d = series[pl][0]
            lat = sorted(((int(getattr(x, "InstanceNumber", 0) or 0), x) for x in
                          (pydicom.dcmread(str(p)) for p in d.glob("*.dcm"))), key=lambda t: t[0])
            giua = lat[len(lat) // 2][1]
            khoi, n = nb.doc_series(d)
            ra[pl] = {"goc": giua.pixel_array.astype(np.float32), "khoi": khoi, "n": n,
                      "co": giua.pixel_array.shape,
                      "tat_ca": np.concatenate([x.pixel_array.ravel() for _, x in lat]).astype(np.float32)}
    finally:
        demo.don(tam)
    return ra


def hinh_quy_trinh(plt, ra: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.05))
    ax.set_xlim(0, 10), ax.set_ylim(0, 4.5), ax.axis("off")
    buoc = [("Thu thập dữ liệu", "4.407 ca, 569 GB;\nđọc header DICOM", "Mục 2.1–2.2"),
            ("Phân tích dữ liệu", "ảnh, báo cáo, nhãn;\nrút ra hướng xử lý", "Mục 2.3–2.6"),
            ("Tiền xử lý ảnh", "chọn series nhạy dịch,\nchuẩn hoá 3 × 24 × 256²", "Mục 4.2"),
            ("Rút nhãn yếu", "từ điển luật, LLM\ncho 4.349 báo cáo", "Mục 3.2, 4.4"),
            ("Giai đoạn 1", "DINOv2 đóng băng\n+ head attention", "Mục 3.3"),
            ("Đánh giá, đo lại", "macro AUC, bootstrap;\nsửa chọn epoch", "Mục 5.2"),
            ("Giai đoạn 2", "CNN 2.5D huấn luyện\ntoàn mạng, 6 mô hình", "Mục 3.4, 4.3"),
            ("Gộp và triển khai", "gộp theo hạng,\nweb demo cục bộ", "Mục 4.5, 5.5")]
    w, h, xs = 2.2, 1.75, [0.05, 2.6, 5.15, 7.7]
    vi_tri = [(xs[i], 2.6) for i in range(4)] + [(xs[3 - i], 0.2) for i in range(4)]
    for i, ((ten, ct, muc), (x, y)) in enumerate(zip(buoc, vi_tri)):
        mau, vien = ("#e8f6ef", LUC) if i >= 6 else ("#eef3fb", XANH)
        ve_hop(ax, x, y, w, h, "", mau=mau, vien=vien)
        ax.text(x + w / 2, y + h - 0.3, f"{i + 1}. {ten}", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.08, ct, ha="center", va="center", fontsize=8, linespacing=1.25)
        ax.text(x + w / 2, y + 0.2, muc, ha="center", va="center", fontsize=7.5, style="italic", color="#555")
    for i in range(3):
        mui_ten(ax, xs[i] + w, 3.47, xs[i + 1], 3.47)
        mui_ten(ax, xs[3 - i], 1.07, xs[2 - i] + w, 1.07)
    mui_ten(ax, xs[3] + w / 2, 2.6, xs[3] + w / 2, 1.95)
    fig.savefig(ra), plt.close(fig)


def hinh_cau_truc(plt, D, ra: Path):
    du = D["du_lieu"]
    so = lambda n: f"{n:,}".replace(",", ".")
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.set_xlim(0, 10), ax.set_ylim(0, 4.3), ax.axis("off")
    ve_hop(ax, 0.05, 1.55, 1.9, 1.2, "Bộ dữ liệu\nMRI khớp gối", mau="#f1f2f4", vien=XAM, dam=True, co=9)
    tep = [(3.25, "train.csv", f"{so(du['n_study'])} dòng, mỗi dòng một ca:\nmã ca, báo cáo chẩn đoán,\n12 cột nhãn (chỉ {du['n_gold']} ca có giá trị)"),
           (1.75, "train_series.csv", f"{so(du['n_series'])} dòng, mỗi dòng một series:\nmã ca, mã series, mặt phẳng,\ncờ nhạy dịch, cờ xoá mỡ"),
           (0.25, "train_series/", "<mã ca>/<mã series>/<lát>.dcm\nmỗi lát cắt là một file DICOM\n(ảnh + header thông số máy)")]
    for y, ten, ct in tep:
        ve_hop(ax, 2.55, y, 1.85, 0.85, ten, dam=True, co=9)
        ve_hop(ax, 4.55, y - 0.1, 5.4, 1.05, ct, mau="#ffffff", vien="#c5cdd6", co=8)
        mui_ten(ax, 1.95, 2.15, 2.55, y + 0.42)
        ax.plot([4.4, 4.55], [y + 0.42, y + 0.42], color="#999", lw=0.8)
    fig.savefig(ra), plt.close(fig)


def hinh_eda_anh(plt, T, ra: Path):
    import numpy as np
    fig, axs = plt.subplots(2, 3, figsize=(6.4, 3.95))
    for a in axs.flat:
        a.tick_params(labelsize=7.5)
    so = lambda n: f"{n:,}".replace(",", ".")
    # (a) so series moi ca
    a = axs[0, 0]
    k, v = zip(*sorted(T["series_moi_ca"].items()))
    a.bar(k, v, color=XANH, width=0.75)
    a.set_title("(a) Số series mỗi ca", fontsize=8.5), a.set_xticks(range(3, 15, 2))
    a.set_ylabel("Số ca", fontsize=8)
    a.yaxis.set_major_formatter(lambda x, _: so(int(x)))
    # (b) mat phang x nhay dich
    a = axs[0, 1]
    mp = T["mat_phang"]
    ten = list(mp)
    nh = [mp[t]["nhay"] for t in ten]
    kh = [mp[t]["tong"] - mp[t]["nhay"] for t in ten]
    a.bar(ten, nh, color=XANH, width=0.6, label="nhạy dịch")
    a.bar(ten, kh, bottom=nh, color="#c5d6ef", width=0.6, label="khác")
    a.set_title("(b) Series theo mặt phẳng", fontsize=8.5)
    a.set_ylim(0, 12500)
    a.legend(fontsize=7, frameon=False, loc="upper right", ncol=2, handlelength=1, columnspacing=0.8)
    a.yaxis.set_major_formatter(lambda x, _: so(int(x)))
    # (c) so lat moi series
    a = axs[0, 2]
    lat = np.clip(T["lat"], 0, 80)
    a.hist(lat, bins=range(10, 82, 3), color=XANH)
    a.axvline(np.median(T["lat"]), color=CAM, lw=1.2, ls="--")
    a.set_title("(c) Số lát mỗi series", fontsize=8.5)
    a.text(0.97, 0.92, f"trung vị {int(np.median(T['lat']))}", transform=a.transAxes, ha="right", fontsize=7.5, color=CAM)
    a.yaxis.set_major_formatter(lambda x, _: so(int(x)))
    # (d) hang may
    a = axs[1, 0]
    h = sorted(T["hang"].items(), key=lambda t: t[1])
    tong = sum(v for _, v in h)
    a.barh([t.title() if t != "GE" else "GE" for t, _ in h], [v for _, v in h], color=LUC, height=0.6)
    for i, (_, v) in enumerate(h):
        a.text(v + tong * 0.01, i, f(100 * v / tong, 1) + "%", va="center", fontsize=7)
    a.set_xlim(0, max(v for _, v in h) * 1.35)
    a.set_title("(d) Hãng máy (theo ca)", fontsize=8.5)
    a.xaxis.set_major_formatter(lambda x, _: so(int(x)))
    # (e) tu truong
    a = axs[1, 1]
    tt = sorted(T["tu_truong"].items())
    a.bar([f(t, 1) + " T" for t, _ in tt], [v for _, v in tt], color=LUC, width=0.6)
    for i, (_, v) in enumerate(tt):
        a.text(i, v + 80, so(v), ha="center", fontsize=7)
    a.set_title("(e) Từ trường (series đã chọn)", fontsize=8.5)
    a.yaxis.set_major_formatter(lambda x, _: so(int(x)))
    # (f) kich thuoc ma tran
    a = axs[1, 2]
    a.hist(T["rows"], bins=range(160, 1300, 64), color=LUC)
    a.set_title("(f) Kích thước ảnh (điểm ảnh)", fontsize=8.5)
    a.yaxis.set_major_formatter(lambda x, _: so(int(x)))
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    fig.savefig(ra), plt.close(fig)


def hinh_eda_bao_cao(plt, D, T, ra: Path):
    import numpy as np
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1.1, 1]})
    nn = sorted(D["du_lieu"]["ngon_ngu"], key=lambda r: r["n"])
    a.barh([r["ten"] for r in nn], [r["n"] for r in nn], color=XANH, height=0.65)
    for i, r in enumerate(nn):
        a.text(r["n"] + 20, i, f"{r['n']:,}".replace(",", "."), va="center", fontsize=7)
    a.set_xlim(0, 2100), a.set_title("(a) Ngôn ngữ của báo cáo (số ca)", fontsize=8.5)
    a.tick_params(labelsize=7.5)
    a.xaxis.set_major_formatter(lambda x, _: f"{int(x):,}".replace(",", "."))
    b.hist(T["so_tu"], bins=range(0, 700, 25), color=XANH)
    b.axvline(np.median(T["so_tu"]), color=CAM, lw=1.2, ls="--")
    b.text(np.median(T["so_tu"]) + 15, b.get_ylim()[1] * 0.88, f"trung vị {int(np.median(T['so_tu']))} từ",
           fontsize=7.5, color=CAM)
    b.set_title("(b) Độ dài báo cáo (số từ)", fontsize=8.5), b.set_ylabel("Số ca", fontsize=8)
    b.tick_params(labelsize=7.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(ra), plt.close(fig)


def hinh_ty_le_nhan(plt, T, ra: Path):
    import numpy as np
    L = list(TEN_VI)
    thu_tu = sorted(L, key=lambda k: T["ty_le_gold"][k])
    y = np.arange(len(L))
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.barh(y + 0.19, [100 * T["ty_le_gold"][k] for k in thu_tu], height=0.38, color=CAM,
            label="58 ca có nhãn bác sĩ")
    ax.barh(y - 0.19, [100 * T["ty_le_llm"][k] for k in thu_tu], height=0.38, color=XANH,
            label=f"{T['n_llm']:,}".replace(",", ".") + " ca nhãn LLM (xác suất > 0,5)")
    ax.set_yticks(y, [TEN_VI[k] for k in thu_tu], fontsize=8.5)
    ax.set_xlabel("Tỷ lệ ca dương tính (%)", fontsize=9), ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(ra), plt.close(fig)


def hinh_mri(plt, M, ra: Path):
    import numpy as np
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.45))
    ten = {"Sagittal": "Sagittal (đứng dọc)", "Coronal": "Coronal (đứng ngang)",
           "Axial": "Axial (nằm ngang)"}
    for a, pl in zip(axs, ("Sagittal", "Coronal", "Axial")):
        g = M[pl]["goc"]
        lo, hi = np.percentile(M[pl]["tat_ca"], (1, 99))
        a.imshow(np.clip((g - lo) / (hi - lo), 0, 1), cmap="gray")
        a.set_title(ten[pl], fontsize=8.5)
        a.text(0.5, -0.07, f"{M[pl]['n']} lát gốc · {M[pl]['co'][0]} × {M[pl]['co'][1]} điểm ảnh",
               transform=a.transAxes, ha="center", va="top", fontsize=7.5)
        a.axis("off")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(ra), plt.close(fig)


def hinh_tien_xu_ly(plt, M, ra: Path):
    import numpy as np
    m = M["Sagittal"]
    g, khoi = m["goc"], m["khoi"]
    lo, hi = np.percentile(m["tat_ca"], (1, 99))
    fig = plt.figure(figsize=(6.4, 3.05))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 0.78], hspace=0.42, wspace=0.28)
    a = fig.add_subplot(gs[0, 0])
    a.imshow(g, cmap="gray"), a.axis("off")
    a.set_title(f"(a) Lát gốc {g.shape[0]}×{g.shape[1]},\nhiển thị min–max", fontsize=8)
    a = fig.add_subplot(gs[0, 1])
    v = m["tat_ca"]
    a.hist(v[v > 0], bins=80, color=XAM, log=True)
    for x in (lo, hi):
        a.axvline(x, color=CAM, lw=1.1, ls="--")
    a.set_title("(b) Cường độ cả series,\ncắt ở phân vị 1 và 99", fontsize=8)
    a.tick_params(labelsize=6.5), a.set_yticks([]), a.minorticks_off()
    a.spines["left"].set_visible(False)
    a = fig.add_subplot(gs[0, 2])
    k = 12
    a.imshow(khoi[k], cmap="gray", vmin=0, vmax=255), a.axis("off")
    a.set_title("(c) Sau chuẩn hoá,\n256×256, uint8", fontsize=8)
    a = fig.add_subplot(gs[0, 3])
    a.imshow(np.stack([khoi[k - 1], khoi[k], khoi[k + 1]], -1)), a.axis("off")
    a.set_title("(d) Cửa sổ 2.5D: lát\nk−1, k, k+1 → R, G, B", fontsize=8)
    con = gs[1, :].subgridspec(1, 8, wspace=0.06)
    for j, i in enumerate(range(2, 24, 3)):
        a = fig.add_subplot(con[0, j])
        a.imshow(khoi[i], cmap="gray", vmin=0, vmax=255), a.axis("off")
        a.set_title(f"lát {i + 1}", fontsize=7)
    fig.text(0.5, 0.0, f"(e) 24 lát lấy cách đều theo chiều sâu từ {m['n']} lát gốc (hiện 8 lát, cách nhau 3)",
             ha="center", fontsize=8)
    fig.savefig(ra), plt.close(fig)


def hinh_roc(plt, D, ra: Path, k: float = 1.0, cham: bool = False, co=(6.4, 3.95), gon: bool = False) -> float:
    """ROC tung benh cua mo hinh cuoi (gop hang 6 CNN). Tra macro AUC de doi chieu voi file ket qua.

    cham=True: so thap phan dau cham. gon=True (ban slide): bo dong "x ca duong" va so truc o cac o trong,
    de chu con lai du to khi chieu."""
    fm = (lambda x, n=3: f"{x:.{n}f}") if cham else f
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score, roc_curve
    sys.path.insert(0, str(REPO / "src"))
    from rsna_knee.oof import gold_theo_fold, ma_thu_tu
    y, _, uid = gold_theo_fold()
    z = np.load(REPO / "reports" / "day15_model_cuoi_du_doan.npz")
    if str(z["uid_md5"]) != ma_thu_tu(uid):
        raise SystemExit("Thu tu ca trong day15_model_cuoi_du_doan.npz khac thu tu gold hien tai")
    P = np.mean([np.column_stack([rankdata(z[f"model_{s}"][:, j]) for j in range(12)]) for s in range(1, 7)], 0)
    fig, axs = plt.subplots(3, 4, figsize=co, sharex=True, sharey=True)
    auc = []
    for j, (ten, a) in enumerate(zip(TEN_VI, axs.flat)):
        fpr, tpr, _ = roc_curve(y[:, j], P[:, j])
        auc.append(roc_auc_score(y[:, j], P[:, j]))
        a.plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls=":")
        a.plot(fpr, tpr, color=LUC, lw=1.4)
        a.fill_between(fpr, tpr, step=None, color=LUC, alpha=0.08)
        a.set_title(TEN_VI[ten].split(" (")[0], fontsize=8 * k)
        a.text(0.96, 0.05, f"AUC {fm(auc[-1])}", ha="right", fontsize=7.5 * k, transform=a.transAxes)
        if not gon:
            a.text(0.96, 0.05 + 0.13 * k, f"{int(y[:, j].sum())} ca dương", ha="right", fontsize=7 * k, color="#555",
                   transform=a.transAxes)
        a.tick_params(labelsize=7 * k)
        a.set_xticks([0, 1] if gon else [0, 0.5, 1]), a.set_yticks([0, 1] if gon else [0, 0.5, 1])
        if gon:
            a.label_outer()
        a.xaxis.set_major_formatter(lambda x, _: fm(x, 1)), a.yaxis.set_major_formatter(lambda x, _: fm(x, 1))
    fig.supxlabel("Tỷ lệ dương tính giả (1 − độ đặc hiệu)", fontsize=8.5 * k)
    fig.supylabel("Độ nhạy", fontsize=8.5 * k)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(ra), plt.close(fig)
    return float(np.mean(auc))


def hinh_web_kien_truc(plt, ra: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.set_xlim(0, 10), ax.set_ylim(0, 4.6), ax.axis("off")
    for x, ten in ((0.1, "Lớp giao diện"), (3.5, "Lớp máy chủ"), (6.9, "Lớp suy luận")):
        ax.text(x, 4.4, ten, fontsize=9, fontweight="bold", color="#555")
    ve_hop(ax, 0.1, 1.6, 2.7, 2.6, "Trình duyệt\n(trang HTML tĩnh)\n\n• chọn ca mẫu,\n  tải zip DICOM\n• bảng 12 xác suất\n"
           "• khung xem lát\n  mô hình chú ý", co=7.5)
    ve_hop(ax, 3.5, 1.6, 2.7, 2.6, "FastAPI\n127.0.0.1:8790\n\nGET  /\nGET  /api/ca-mau\nPOST /api/du-doan\n\n"
           "giới hạn 400 MB", mau="#f1f2f4", vien=XAM, co=7.5)
    for i, chu in enumerate(("1. Gom file theo series,\nsuy mặt phẳng từ header",
                             "2. Đọc, chuẩn hoá: hàm tiền\nxử lý lúc huấn luyện",
                             "3. 5 CNN trên GPU, trung\nbình xác suất, lát chính")):
        ve_hop(ax, 6.9, 3.4 - i * 0.9, 3.0, 0.78, chu, mau="#e8f6ef", vien=LUC, co=7.5)
    for i in range(2):
        mui_ten(ax, 8.4, 3.4 - i * 0.9, 8.4, 3.18 - i * 0.9)
    ax.annotate("", xy=(3.5, 3.1), xytext=(2.8, 3.1), arrowprops={"arrowstyle": "-|>", "color": "#333", "lw": 1.1})
    ax.annotate("", xy=(2.8, 2.7), xytext=(3.5, 2.7), arrowprops={"arrowstyle": "-|>", "color": "#333", "lw": 1.1, "ls": "--"})
    ax.text(3.15, 3.22, "HTTP", ha="center", fontsize=7), ax.text(3.15, 2.45, "JSON", ha="center", fontsize=7)
    ax.annotate("", xy=(6.9, 2.9), xytext=(6.2, 2.9), arrowprops={"arrowstyle": "-|>", "color": "#333", "lw": 1.1})
    for x, chu in ((0.1, "Trọng số 5 CNN\n(trọng số EMA, epoch 12)"), (3.5, "Ca mẫu: zip DICOM\n+ nhãn bác sĩ"),
                   (6.9, "Thư mục tạm cho ca tải\nlên, xoá sau mỗi lượt")):
        ve_hop(ax, x, 0.15, 2.7 if x < 6 else 3.0, 0.85, chu, mau="#fdf1ea", vien=CAM, co=7.5)
    ax.text(5.0, 1.22, "Lưu trên máy (không gửi ra ngoài)", ha="center", fontsize=7.5, style="italic", color="#555")
    fig.savefig(ra), plt.close(fig)


def hinh_web_tuan_tu(plt, ra: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.set_xlim(0, 10), ax.set_ylim(0, 11.4), ax.axis("off")
    cot = {"nd": 0.8, "td": 2.7, "api": 4.6, "sl": 6.6, "gpu": 8.9}
    for k, ten in (("nd", "Người dùng"), ("td", "Trình duyệt"), ("api", "FastAPI"), ("sl", "Lớp suy luận"),
                   ("gpu", "5 CNN (GPU)")):
        ve_hop(ax, cot[k] - 0.8, 10.55, 1.6, 0.7, ten, co=7.5, dam=True)
        ax.plot([cot[k]] * 2, [0.2, 10.55], color="#aaa", lw=0.8, ls=":")
    tin = [
        ("nd", "td", "chọn ca mẫu / kéo thả zip", False), ("td", "api", "POST /api/du-doan", False),
        ("api", "sl", "gom series, suy mặt phẳng", False), ("sl", "sl", "chọn series nhạy dịch, chuẩn hoá → 3×24×256²", False),
        ("sl", "gpu", "72 cửa sổ 2.5D, chạy lần lượt 5 mô hình", False),
        ("gpu", "sl", "mỗi mô hình: 12 logit + trọng số chú ý", True),
        ("sl", "sl", "sigmoid, trung bình 5 mô hình; lát chính mỗi bệnh", False),
        ("sl", "api", "12 bệnh + 72 lát JPEG", True), ("api", "td", "JSON (xoá thư mục tạm)", True),
        ("td", "nd", "bảng 12 xác suất + lát chú ý", True),
    ]
    for i, (a, b, chu, ve) in enumerate(tin):
        y = 9.75 - i * 1.0
        if a == b:
            x = cot[a]
            ax.plot([x, x + 0.35, x + 0.35, x], [y + 0.12, y + 0.12, y - 0.28, y - 0.28], color="#333", lw=0.9)
            ax.text(x - 0.1, y - 0.08, chu, ha="right", va="center", fontsize=7,
                    bbox={"fc": "white", "ec": "none", "pad": 0.5})
            continue
        ax.annotate("", xy=(cot[b], y), xytext=(cot[a], y),
                    arrowprops={"arrowstyle": "-|>", "color": "#333", "lw": 0.9, "ls": "--" if ve else "-"})
        ax.text((cot[a] + cot[b]) / 2, y + 0.16, chu, ha="center", fontsize=7,
                bbox={"fc": "white", "ec": "none", "pad": 0.3})
    fig.savefig(ra), plt.close(fig)


def phan_tich_ca() -> dict:
    """Ket qua web demo tren 8 ca mau (data/web_demo/phan_tich_ca/ca_<k>.json, khong len git).

    Thieu file thi goi web dang chay (127.0.0.1:8790) bang curl, dung duong ca tai len."""
    import numpy as np
    import pandas as pd
    goc = REPO / "data" / "web_demo"
    ra = goc / "phan_tich_ca"
    ra.mkdir(parents=True, exist_ok=True)
    uid = dict(enumerate((goc / "ca_mau" / "danh_sach.txt").read_text().split(), 1))
    tr = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    ca, tap_trung = {}, {}
    for k in uid:
        f = ra / f"ca_{k}.json"
        if not f.exists():
            subprocess.run(["curl", "-s", "-m", "180", "-F", f"tep=@{goc / 'ca_mau' / f'ca_mau_{k}.zip'}",
                            "http://127.0.0.1:8790/api/du-doan", "-o", str(f)], check=True)
        d = json.loads(f.read_text(encoding="utf-8"))
        if "benh" not in d:
            raise SystemExit(f"Web demo tra loi cho ca {k}: {d}")
        g = tr.loc[uid[k]]
        for b in d["benh"]:
            b["bac_si"] = int(g[b["ma"]])
            a = np.array(b["chu_y"], float).ravel()
            a = a / a.sum()
            tap_trung.setdefault(b["ma"], []).append(float(np.sort(a)[-5:].sum()))
        ca[k] = d
    return {"ca": ca, "tap_trung": {m: float(np.mean(v)) for m, v in tap_trung.items()},
            "n_cua_so": 3 * 24}


CA_TOT, CA_KEM = (1, "Baker's"), (3, "Synovitis")   # ca mau 1 tren web; ca 3 khong len web (mo hinh bat 1/3)


def hinh_phan_tich_ca(plt, PT, ra: Path):
    import base64
    import io

    import numpy as np
    from PIL import Image
    fig = plt.figure(figsize=(6.4, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.1], hspace=0.55, wspace=0.18)
    mau_mp = (XANH, LUC, CAM)
    ymax = 0
    hang = []
    for r, (k, ma) in enumerate((CA_TOT, CA_KEM)):
        d = PT["ca"][k]
        b = next(x for x in d["benh"] if x["ma"] == ma)
        a = np.array(b["chu_y"], float)
        j, t = b["lat_chinh"]["mat_phang"], b["lat_chinh"]["lat"]
        anh = Image.open(io.BytesIO(base64.b64decode(d["mat_phang"][j]["lat"][t])))
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(np.asarray(anh), cmap="gray"), ax.axis("off")
        ax.set_title(f"{d['mat_phang'][j]['mat_phang']}, lát {t + 1}", fontsize=7.5)
        ax2 = fig.add_subplot(gs[r, 1])
        for jj in range(3):
            ax2.bar(np.arange(24) + 24 * jj, a[jj], color=mau_mp[jj], width=0.85,
                    label=d["mat_phang"][jj]["mat_phang"] if r == 0 else None)
        ax2.axhline(1 / PT["n_cua_so"], color="#555", lw=0.8, ls=":")
        ax2.set_xticks([12, 36, 60], ["Sagittal", "Coronal", "Axial"], fontsize=7.5)
        ax2.set_yticks(np.arange(0, 0.121, 0.02)), ax2.tick_params(axis="y", labelsize=7)
        ax2.yaxis.set_major_formatter(lambda x, _: f(x, 2))
        ymax = max(ymax, a.max())
        ten = "Ca A" if r == 0 else "Ca B"
        ax2.set_title(f"{ten} · {TEN_VI[ma]}: bác sĩ ghi {'có' if b['bac_si'] else 'không'}, mô hình "
                      f"{f(b['p'], 2)}", fontsize=8, loc="left")
        hang.append(ax2)
    for ax2 in hang:
        ax2.set_ylim(0, ymax * 1.12)
    hang[0].text(71, 1 / PT["n_cua_so"], "chia đều (1/72)", ha="right", va="bottom", fontsize=6.5, color="#555")
    fig.text(0.62, 0.0, "Trọng số chú ý của bệnh trên 72 cửa sổ (3 mặt phẳng × 24 lát)", ha="center", fontsize=8)
    fig.savefig(ra), plt.close(fig)


def ktc_tung_benh(D) -> dict:
    """AUC va KTC bootstrap 95% tung benh cua mo hinh cuoi, dung dung ham lay mau cua scripts/93."""
    import importlib.util

    import numpy as np
    sys.path.insert(0, str(REPO / "src"))
    from rsna_knee.oof import gold_theo_fold, ma_thu_tu
    sp = importlib.util.spec_from_file_location("cham93", REPO / "scripts" / "93_cham_model_cuoi.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    y, _, uid = gold_theo_fold()
    z = np.load(REPO / "reports" / "day15_model_cuoi_du_doan.npz")
    if str(z["uid_md5"]) != ma_thu_tu(uid):
        raise SystemExit("Thu tu ca trong day15_model_cuoi_du_doan.npz khac thu tu gold hien tai")
    g = m.gop([z[f"model_{s}"] for s in range(1, 7)])
    mau = list(m._mau(y, 2000, 0))
    ra = {}
    for j, ten in enumerate(m.L):
        auc = m._auc_nhanh(y[:, j], g[:, j])
        if abs(auc - D["cuoi"]["tung_benh"][ten]) > 5e-4:
            raise SystemExit(f"AUC {ten} tinh lai {auc:.4f} lech file ket qua")
        ra[ten] = [float(v) for v in np.percentile([m._auc_nhanh(y[i, j], g[i, j]) for i in mau], [2.5, 97.5])]
    return ra


def anh_slide(M, ra_thu_muc: Path) -> None:
    """Anh khong chu cho slide (chu dat bang HTML de cung font slide)."""
    import numpy as np
    from PIL import Image
    ra_thu_muc.mkdir(parents=True, exist_ok=True)
    for pl in ("Sagittal", "Coronal", "Axial"):
        g = M[pl]["goc"]
        lo, hi = np.percentile(M[pl]["tat_ca"], (1, 99))
        Image.fromarray((np.clip((g - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)).save(ra_thu_muc / f"mri_{pl}.png")
    m = M["Sagittal"]
    g = m["goc"]
    Image.fromarray((255 * (g - g.min()) / (g.max() - g.min())).astype(np.uint8)).save(ra_thu_muc / "txl_goc.png")
    Image.fromarray(m["khoi"][12]).save(ra_thu_muc / "txl_chuan.png")
    Image.fromarray(np.stack([m["khoi"][11], m["khoi"][12], m["khoi"][13]], -1)).save(ra_thu_muc / "txl_rgb.png")
    o = list(m["khoi"]) + [np.zeros_like(m["khoi"][0])]   # 24 lat + 1 o trong -> luoi vuong 5 x 5
    luoi = np.vstack([np.hstack(o[r * 5:(r + 1) * 5]) for r in range(5)])
    Image.fromarray(luoi).save(ra_thu_muc / "txl_24_lat.png")


def chup_web(ra: Path) -> None:
    """Chup web demo bang Chrome headless (khong hien cua so). Web khong chay thi giu anh cu."""
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8790/api/ca-mau", timeout=3)
    except OSError:
        if ra.exists():
            print("  web demo khong chay - dung anh chup cu", ra.name)
            return
        raise SystemExit("Web demo chua chay: uv run --extra demo python scripts/95_web_demo.py")
    doi = lambda p: subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True,
                                   check=True).stdout.strip()
    ho_so = Path(tempfile.mkdtemp(prefix="chrome_pbl6_"))
    tam = ra.with_suffix(".tam.png")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                    f"--user-data-dir={doi(ho_so)}", "--virtual-time-budget=60000", "--window-size=1600,1080",
                    "--force-device-scale-factor=2", f"--screenshot={doi(tam)}", WEB],
                   capture_output=True, timeout=240)
    shutil.rmtree(ho_so, ignore_errors=True)
    if not tam.exists():
        raise SystemExit("Chrome khong chup duoc web demo")
    tam.replace(ra)


def ve_hinh(D) -> dict[str, Path]:
    HINH.mkdir(parents=True, exist_ok=True)
    plt = chuan_bi_matplotlib()
    h = {k: HINH / f"{k}.png" for k in ("tong_quan", "kien_truc", "hanh_trinh", "epoch", "tung_benh", "web",
                                         "quy_trinh", "cau_truc", "eda_anh", "eda_bao_cao", "ty_le_nhan",
                                         "mri", "tien_xu_ly", "roc", "web_kien_truc", "web_tuan_tu",
                                         "phan_tich_ca")}
    import numpy as np
    (HINH / "slide").mkdir(parents=True, exist_ok=True)
    T = thong_ke_du_lieu()
    T["cau_vi_du"] = json.loads(THONG_TIN.read_text(encoding="utf-8"))["vi_du_bao_cao"]
    M = nap_ca_mau()
    chup_web(h["web"])
    hinh_tong_quan(plt, D, h["tong_quan"])
    hinh_kien_truc(plt, h["kien_truc"])
    hinh_hanh_trinh(plt, D, h["hanh_trinh"])
    hinh_epoch(plt, D, h["epoch"])
    hinh_tung_benh(plt, D, h["tung_benh"])
    hinh_quy_trinh(plt, h["quy_trinh"])
    hinh_cau_truc(plt, D, h["cau_truc"])
    hinh_eda_anh(plt, T, h["eda_anh"])
    hinh_eda_bao_cao(plt, D, T, h["eda_bao_cao"])
    hinh_ty_le_nhan(plt, T, h["ty_le_nhan"])
    hinh_mri(plt, M, h["mri"])
    hinh_tien_xu_ly(plt, M, h["tien_xu_ly"])
    hinh_web_kien_truc(plt, h["web_kien_truc"])
    hinh_web_tuan_tu(plt, h["web_tuan_tu"])
    T["pt"] = phan_tich_ca()
    hinh_phan_tich_ca(plt, T["pt"], h["phan_tich_ca"])
    T["ktc_benh"] = ktc_tung_benh(D)
    auc_roc = hinh_roc(plt, D, h["roc"])
    g6 = D["cuoi"]["nhom"]["gop_6_cnn"]["auc"]
    if abs(auc_roc - g6) > 5e-4:
        raise SystemExit(f"ROC tinh lai {auc_roc:.4f} lech file ket qua {g6:.4f}")
    # ban cho slide: chu Arial (gan font slide), co lon hon
    with plt.rc_context({"font.family": "Arial", "savefig.transparent": True}):
        hinh_roc(plt, D, HINH / "slide" / "roc.png", k=1.55, co=(8.6, 6.4), gon=True)
    anh_slide(M, HINH / "slide")
    T.update(n_lat_mau={pl: M[pl]["n"] for pl in M})
    (HINH / "slide" / "thong_ke.json").write_text(json.dumps({
        k: v for k, v in T.items() if k not in ("rows", "day_lat", "lat", "so_tu", "pt", "ktc_benh")} | {
        "lat_trung_vi": float(np.median(T["lat"])),
        "rows_min": int(T["rows"].min()), "rows_max": int(T["rows"].max()),
        "so_tu_trung_vi": float(np.median(T["so_tu"]))}, ensure_ascii=False, indent=1,
        default=lambda o: o.item()), encoding="utf-8")
    return h, T


# ============================================================ noi dung
def noi_dung(D, H, T) -> list:
    """Danh sach khoi: (loai, ...). Loai: h1 h2 h3 p ul hinh bang."""
    du = D["du_lieu"]
    n_khong = du["n_study"] - du["n_gold"]
    nhom = D["cuoi"]["nhom"]
    g6, g4, g2t, g2c = nhom["gop_6_cnn"], nhom["gop_4_cnn_cong_khai"], nhom["gop_2_cnn_tu_gan"], nhom["gop_2_cnn_cong_khai"]
    ss = D["cuoi"]["so_sanh"]
    llm = D["llm"]["nguon"]
    cn = D["cnn"]
    ce = D["chon_epoch"]
    nk = SO_NHAT_KY
    so = lambda n: f"{n:,}".replace(",", ".")
    ti_le_gold = f(100 * du["n_gold"] / du["n_study"], 2)
    duong = {r["nhan"]: r["n"] for r in du["nhan_duong"]}
    dino = {r["nhan"]: r["dino"] for r in cn["tung_nhan"]}
    cnn2 = {r["nhan"]: r["cnn"] for r in cn["tung_nhan"]}
    tm = {m["so"]: m for m in D["cuoi"]["tung_model"]}
    hk = D["llm"]["hai_bo_llm_ngoai_gold"]

    K = []
    h1 = lambda t: K.append(("h1", t))
    h2 = lambda t: K.append(("h2", t))
    h3 = lambda t: K.append(("h3", t))
    p = lambda t: K.append(("p", t))
    ul = lambda *t: K.append(("ul", list(t)))
    ct = lambda t, so: K.append(("ct", t, so))      # cong thuc dong rieng, so (chuong.thu tu) can phai

    # ------------------------------------------------ chuong 1
    h1("CHƯƠNG 1. GIỚI THIỆU")
    h2("1.1. Bối cảnh và bài toán")
    p("Chụp cộng hưởng từ (MRI) là phương tiện chính để đánh giá tổn thương mô mềm ở khớp gối như dây chằng, "
      "sụn chêm và sụn khớp. Một ca chụp gồm nhiều chuỗi ảnh (series) ở ba mặt phẳng Sagittal, Coronal và Axial, "
      "mỗi series vài chục lát cắt, nên việc đọc ảnh tốn thời gian của bác sĩ chẩn đoán hình ảnh. Các nghiên cứu "
      "như MRNet {r:mrnet} đã cho thấy mạng học sâu có thể phát hiện bất thường, rách dây chằng chéo trước và rách "
      "sụn chêm trên MRI gối, khi được huấn luyện trên 1.370 ca có nhãn đầy đủ.")
    p(f"Đồ án sử dụng bộ dữ liệu MRI khớp gối do Hội Điện quang Bắc Mỹ (Radiological Society of North America, "
      f"RSNA) công bố {{r:rsna}}. Đầu vào là một ca MRI gối, trung bình {f(du['series_moi_study'], 2)} series mỗi "
      f"ca; toàn bộ tập dữ liệu gồm {so(du['n_study'])} ca, {so(du['n_series'])} series, khoảng "
      f"{nk['dung_luong_gb']} GB file DICOM (chương 2). Đầu ra là 12 xác suất, mỗi xác suất ứng với một bất thường "
      "trong {b:benh} (phần sau gọi ngắn là \"bệnh\"). Đây là bài toán đa nhãn: một ca có thể mắc nhiều bệnh cùng "
      "lúc. Thước đo là trung bình AUC của 12 nhãn (macro AUC).")
    rows = [[TEN_VI[k].split(" (")[0], k, str(duong[k])] for k in TEN_VI]
    K.append(("bang", "benh", f"12 bất thường cần dự đoán và số ca dương tính trong {du['n_gold']} ca có nhãn bác sĩ",
              ["Bất thường", "Tên trong dữ liệu", "Số ca dương tính"], rows, [7.0, 4.5, 4.0]))
    h2("1.2. Các nghiên cứu liên quan")
    p("Hai bộ dữ liệu MRI gối công khai được dùng nhiều nhất là KneeMRI và MRNet. KneeMRI {r:kneemri} gồm 917 chuỗi "
      "ảnh Sagittal từ một bệnh viện ở Croatia, chỉ gán nhãn tổn thương dây chằng chéo trước; nhóm tác giả dùng đặc "
      "trưng HOG trên vùng quan tâm khoanh tay và máy vectơ hỗ trợ, đạt AUC 0,894 cho việc phát hiện tổn thương. "
      "MRNet {r:mrnet} gồm 1.370 ca có đủ ba mặt phẳng, nhãn cho ba câu hỏi (bất thường, rách dây chằng chéo trước, "
      "rách sụn chêm); mỗi mặt phẳng đi qua một CNN 2D tiền huấn luyện trên ImageNet, đặc trưng các lát được gộp "
      "bằng max pooling rồi kết hợp ba mặt phẳng bằng hồi quy logistic. ELNet {r:elnet} thiết kế một CNN nhỏ dưới "
      "1 MB, huấn luyện từ đầu trên cả hai bộ dữ liệu này, và cho thấy mô hình vẫn khoanh được vị trí tổn thương dù "
      "khi huấn luyện không có nhãn vị trí.")
    p("Cả ba công trình đều có nhãn đầy đủ cho mọi ca và chỉ dự đoán một đến ba nhãn. Bài toán của đồ án khác ở hai "
      "điểm: 12 nhãn cùng lúc, và chỉ 1,32% số ca có nhãn bác sĩ. Hướng gần nhất với điểm thứ hai đến từ ảnh X-quang "
      "ngực. ChestX-ray8 {r:chestxray8} rút 8 nhãn bệnh từ báo cáo chẩn đoán bằng xử lý ngôn ngữ tự nhiên cho hơn "
      "100.000 ảnh rồi huấn luyện mô hình trên các nhãn này; CheXpert {r:chexpert} dùng bộ gán nhãn dựa trên luật, có "
      "xử lý câu phủ định và câu không chắc chắn, cho 14 nhãn; CheXbert {r:chexbert} thay bộ luật bằng mô hình BERT "
      "học từ nhãn của bộ luật rồi tinh chỉnh trên một ít nhãn chuyên gia, và vượt bộ luật. Gần đây, mô hình ngôn "
      "ngữ lớn như GPT-4 được thử chuyển báo cáo tự do ở nhiều ngôn ngữ thành dữ liệu có cấu trúc {r:gpt4bc}. Đồ "
      "án đi theo đúng chuỗi này: từ điển luật trước (mục 3.2.1), LLM sau (mục 3.2.2), và áp dụng cho báo cáo MRI "
      "gối viết bằng 9 ngôn ngữ.")
    h2("1.3. Các vấn đề cần giải quyết")
    p("Bài toán có bốn khó khăn chính:")
    ul(f"**Nhãn rất hiếm.** Chỉ {du['n_gold']} trên {so(du['n_study'])} ca ({ti_le_gold}%) có nhãn do bác sĩ gán. "
       f"{so(n_khong)} ca còn lại không có nhãn bác sĩ, chỉ có báo cáo chẩn đoán viết tự do bằng 9 ngôn ngữ. Muốn "
       "dùng được phần dữ liệu này phải rút nhãn từ văn bản.",
       f"**Dữ liệu ảnh lớn và không đồng nhất.** Mỗi ca có số series, chuỗi xung, số lát và độ phân giải khác nhau, "
       f"đến từ máy của {len(T['hang'])} hãng với từ trường từ 1,0 đến 3,0 T (mục 2.3).",
       f"**Tập đánh giá rất nhỏ.** {du['n_gold']} ca có nhãn bác sĩ là thước đo duy nhất, nên khoảng tin cậy rộng và "
       "rất dễ vô tình chọn mô hình theo đúng tập dùng để chấm, làm điểm số lạc quan.",
       "**Tài nguyên tính toán giới hạn.** Nhóm chỉ có một laptop RTX 4060 8 GB và GPU đám mây miễn phí "
       "(2 × T4 16 GB, tối đa 12 giờ mỗi lần chạy).")
    h2("1.4. Mục tiêu và phạm vi")
    p("Mục tiêu của đồ án là xây dựng trọn vẹn quy trình từ file DICOM thô tới 12 xác suất bệnh, tự viết toàn bộ mã "
      "và chỉ dùng trọng số tiền huấn luyện công khai (ImageNet, DINOv2), không dùng mô hình người khác đã huấn luyện cho "
      "bài toán này; đạt macro AUC cao nhất có thể trên tập có nhãn bác sĩ với cách đo "
      "trung thực; và có một web demo để minh hoạ mô hình trên ca thật.")
    p(f"Phạm vi: hệ thống là sản phẩm học tập ở mức chứng minh ý tưởng, không phải công cụ chẩn đoán lâm sàng. Kết quả "
      f"được đánh giá trên {du['n_gold']} ca có nhãn bác sĩ của bộ dữ liệu. Web demo chỉ chạy trên máy cục bộ; dữ "
      "liệu gốc không được phát tán lại.")
    h2("1.5. Quy trình thực hiện")
    p("Đồ án đi theo tám bước, từ thu thập dữ liệu tới triển khai ({h:quytrinh}). Hai bước đầu (chương 2) tìm hiểu "
      "dữ liệu trước khi chọn mô hình: các đặc điểm đo được ở đây quyết định cách tiền xử lý, cách rút nhãn và cách "
      "đánh giá ở các chương sau. Bước 6 kiểm tra lại chính cách đo của nhóm và dẫn tới thay đổi lớn nhất về phương "
      "pháp (mục 5.2). Màu xanh lá đánh dấu hai bước của mô hình cuối.")
    K.append(("hinh", "quytrinh", H["quy_trinh"], 15.5, "Quy trình thực hiện đồ án từ dữ liệu thô tới web demo"))

    # ------------------------------------------------ chuong 2
    mp, hg, tt = T["mat_phang"], T["hang"], T["tu_truong"]
    tong_hang = sum(hg.values())
    import numpy as np
    lat, so_tu, kt = T["lat"], T["so_tu"], T["rows"]
    tu_tong = sum(tt.values())
    nn_ = {r["ma"]: r for r in du["ngon_ngu"]}
    vd = T["vi_du"]
    h1("CHƯƠNG 2. DỮ LIỆU VÀ PHÂN TÍCH DỮ LIỆU")
    h2("2.1. Nguồn và cấu trúc dữ liệu")
    p(f"Bộ dữ liệu gồm ảnh MRI khớp gối và báo cáo chẩn đoán đi kèm của {so(du['n_study'])} ca chụp. Dữ liệu đã được "
      "ẩn danh trước khi công bố: mã ca và mã series là chuỗi UID sinh lại, ngày tháng trong báo cáo được thay bằng "
      "chuỗi [DATE], và tên series của một số ca được thay bằng chuỗi giả. Dữ liệu gồm ba phần ({h:cautruc}):")
    K.append(("hinh", "cautruc", H["cau_truc"], 14.5, "Cấu trúc bộ dữ liệu"))
    ul(f"**train.csv:** mỗi ca một dòng, gồm mã ca, toàn văn báo cáo chẩn đoán và 12 cột nhãn. Chỉ {du['n_gold']} "
       f"ca có giá trị ở các cột nhãn (0 hoặc 1, do bác sĩ gán); {so(n_khong)} ca còn lại để trống.",
       "**train_series.csv:** mỗi series một dòng, cho biết series thuộc ca nào, chụp ở mặt phẳng nào, có thuộc "
       "nhóm chuỗi xung nhạy dịch (fluid-sensitive) hay không và có xoá mỡ hay không.",
       "**Thư mục ảnh:** tổ chức theo <mã ca>/<mã series>/<lát>.dcm. Mỗi file DICOM chứa một lát cắt và phần "
       "header ghi thông số chụp: hãng máy, từ trường, kích thước ảnh, độ dày lát và thứ tự lát.")
    p("Mỗi ca thường có series ở cả ba mặt phẳng ({h:mri}). Sagittal là mặt cắt đứng dọc, thấy rõ dây chằng chéo "
      "và sụn chêm; Coronal là mặt cắt đứng ngang, thấy dây chằng bên và hai khoang khớp trong, ngoài; Axial là mặt "
      "cắt nằm ngang, thấy rõ khớp bánh chè – đùi. Mỗi bệnh hiện rõ nhất trên một vài mặt phẳng, nên mô hình dùng "
      "cả ba.")
    K.append(("hinh", "mri", H["mri"], 15.5,
              "Ba mặt phẳng của một ca mẫu (lát giữa của series nhạy dịch, đã cắt cường độ theo phân vị 1–99)"))
    h2("2.2. Thu thập dữ liệu")
    p(f"Toàn bộ ảnh có dung lượng khoảng {nk['dung_luong_gb']} GB, vượt xa ổ cứng và đường truyền của nhóm. Nhóm thử "
      f"tải file DICOM về máy qua API của nền tảng lưu trữ: sau khoảng {nk['n_file_tai']} file và hơn "
      f"{nk['n_lenh_liet_ke']} lệnh liệt kê trong một buổi, tài khoản bị chặn với mã HTTP 429 và phải chờ "
      f"{nk['giay_cho_429']} giây (khoảng 50 giờ) mới được gọi tiếp. Danh sách file lấy được trước khi bị chặn phủ "
      f"{so(T['n_series_liet_ke'])} series và được dùng để thống kê số lát ở mục 2.3.")
    p(f"Vì vậy nhóm chuyển bước đọc ảnh lên notebook GPU đám mây, nơi bộ dữ liệu được gắn sẵn vào máy và không tốn "
      "lượt gọi API. Trên đó, mỗi ca được đọc, chọn series và chuẩn hoá một lần (mục 4.2); header của series được "
      f"chọn ở mỗi mặt phẳng được ghi thành bảng ({so(T['n_header'])} series, tức 3 series cho mỗi ca). Về máy "
      "chỉ tải kết quả: bảng header, đặc trưng, dự đoán và trọng số mô hình, tổng cộng vài trăm MB. Riêng web demo "
      "cần ảnh gốc, nên 8 ca có nhãn bác sĩ được đóng gói thành file zip DICOM (mỗi mặt phẳng một series) để chạy "
      "trên máy.")
    h2("2.3. Phân tích dữ liệu ảnh")
    p(f"{{h:edaanh}} tóm tắt dữ liệu ảnh. Mỗi ca có từ {T['series_min']} đến {T['series_max']} series, phổ biến "
      f"nhất là 5 series ({pt(T['series_mode_ti_le'])} số ca). Sagittal có nhiều series nhất "
      f"({so(mp['Sagittal']['tong'])}), Axial ít nhất ({so(mp['Axial']['tong'])}); {pt(T['nhay_dich'])} số series "
      f"thuộc nhóm nhạy dịch. Trên {so(T['n_series_liet_ke'])} series có trong danh sách file, số lát mỗi series có "
      f"trung vị {int(np.median(lat))} (khoảng tứ phân vị {int(np.percentile(lat, 25))}–{int(np.percentile(lat, 75))}), "
      f"nhưng có series tới {int(lat.max())} lát.")
    p(f"Header của {so(T['n_header'])} series được chọn cho thấy ảnh đến từ {len(hg)} hãng máy, nhiều nhất là Siemens "
      f"({pt(hg['SIEMENS'] / tong_hang)} số ca) và Philips ({pt(hg['PHILIPS'] / tong_hang)}). Phần lớn series chụp ở "
      f"từ trường 1,5 T ({pt(tt[1.5] / tu_tong)}) và 3,0 T ({pt(tt[3.0] / tu_tong)}). Kích thước ảnh từ "
      f"{int(kt.min())} đến {so(int(kt.max()))} điểm ảnh mỗi cạnh (trung vị {int(np.median(kt))}), độ dày lát từ "
      f"{f(T['day_lat'].min(), 1)} đến {f(T['day_lat'].max(), 0)} mm (trung vị {f(np.median(T['day_lat']), 0)} mm).")
    K.append(("hinh", "edaanh", H["eda_anh"], 15.5,
              "Đặc điểm dữ liệu ảnh: số series, mặt phẳng, số lát, hãng máy, từ trường và kích thước ảnh"))
    p("Dữ liệu không đồng nhất ở mọi chiều đo, nên không thể đưa thẳng vào mạng nơ-ron. Cần chọn một series cho mỗi "
      "mặt phẳng, đưa về cùng số lát và cùng kích thước, và chuẩn hoá cường độ vì mỗi máy cho thang giá trị điểm ảnh "
      f"khác nhau. Sự khác biệt giữa các hãng máy còn là một rủi ro: đặc trưng DINOv2 của giai đoạn 1 đủ để đoán đúng "
      f"hãng máy ở {nk['hang_may']}% số ca, tức mô hình có thể học lối tắt theo nơi chụp thay vì theo tổn thương.")
    h2("2.4. Phân tích báo cáo chẩn đoán")
    p(f"Báo cáo được viết tự do bằng 9 ngôn ngữ ({{h:edabaocao}}a). Tiếng Anh chiếm nhiều nhất "
      f"({so(nn_['en']['n'])} ca, {pt(nn_['en']['n'] / du['n_study'])}), tiếp theo là tiếng Tây Ban Nha, Thổ Nhĩ Kỳ "
      f"và Croatia. Độ dài trung vị là {int(np.median(so_tu))} từ, ngắn nhất {int(so_tu.min())} từ và dài nhất "
      f"{int(so_tu.max())} từ ({{h:edabaocao}}b). Cách viết cũng khác nhau giữa các nơi: có báo cáo chia rõ phần kỹ "
      "thuật, kết quả và kết luận; có báo cáo chỉ ghi câu hỏi lâm sàng và quy trình chụp mà không nêu kết luận nào.")
    K.append(("hinh", "edabaocao", H["eda_bao_cao"], 15.5, "Ngôn ngữ và độ dài của báo cáo chẩn đoán"))
    # cau trich tu bao cao (du lieu goc) nam o thong_tin.json - khong de trong repo cong khai
    cau = [tuple(c) for c in T["cau_vi_du"]]
    for c, _, _ in cau:
        if c.rstrip("…") not in vd["bao_cao"]:
            raise SystemExit(f"Bao cao vi du khong con cau: {c}")
    co = {k for _, _, k in cau if k}
    con_lai = [vd["llm"][k] for k in TEN_VI if k not in co]
    rows = [[c, ng, TEN_VI[k].split(" (")[0] if k else "Không thuộc 12 bệnh", f(vd["llm"][k]) if k else "–"]
            for c, ng, k in cau]
    rows.append(["(không nhắc tới)", "–", f"{len(con_lai)} bệnh còn lại", f"{f(min(con_lai))}–{f(max(con_lai))}"])
    p("{b:vidu} lấy một báo cáo tiếng Tây Ban Nha của một ca không có nhãn bác sĩ, đặt cạnh nhãn LLM của bộ nhãn "
      "công khai dùng để huấn luyện (mục 3.2.2). Báo cáo nêu ba bệnh thuộc danh sách và nhãn LLM cho cả ba xác suất "
      "trên 0,9; các bệnh không được nhắc tới nhận xác suất thấp. Một dấu hiệu khác (hoại tử vô mạch) không thuộc 12 "
      "bệnh nên bị bỏ qua. Như vậy báo cáo chứa đủ thông tin để rút nhãn, với điều kiện công cụ rút nhãn đọc "
      "được nhiều ngôn ngữ và hiểu câu phủ định.")
    if vd["co_nhan_bac_si"]:
        raise SystemExit("Bao cao vi du lai la ca co nhan bac si - doi vi du")
    K.append(("bang", "vidu", "Một báo cáo mẫu và nhãn LLM tương ứng",
              ["Câu trong báo cáo", "Nghĩa", "Bệnh tương ứng", "Nhãn LLM"], rows, [5.6, 4.4, 3.8, 2.2]))
    h2("2.5. Phân tích nhãn")
    p(f"Trong {du['n_gold']} ca có nhãn bác sĩ, mỗi ca trung bình có {f(T['duong_moi_ca_gold'], 1)} bệnh và "
      f"{'không ca nào' if T['gold_khong_benh'] == 0 else str(T['gold_khong_benh']) + ' ca'} hoàn toàn bình thường. "
      f"Tràn dịch khớp phổ biến nhất ({duong['Effusion']} ca), dây chằng bên trong hiếm nhất ({duong['MCL']} ca) "
      f"({{b:benh}}). Vì phần lớn ô (ca, bệnh) là âm tính, một mô hình luôn đoán \"không bệnh\" đã đúng "
      f"{pt(1 - sum(duong.values()) / (12 * du['n_gold']))} số ô, nên độ chính xác (accuracy) không phải thước đo "
      "phù hợp. Với dây chằng bên trong, chia 5 fold thì mỗi fold chỉ còn 1–2 ca dương tính.")
    p(f"{{h:tylenhan}} so tỷ lệ dương tính giữa {du['n_gold']} ca bác sĩ và {so(T['n_llm'])} ca có nhãn LLM. Hai tập "
      f"lệch nhau rõ: viêm màng hoạt dịch chiếm {pt(T['ty_le_gold']['Synovitis'])} ở tập bác sĩ nhưng chỉ "
      f"{pt(T['ty_le_llm']['Synovitis'])} ở nhãn LLM; gãy xương là {pt(T['ty_le_gold']['Fracture'])} so với "
      f"{pt(T['ty_le_llm']['Fracture'])}. Có hai cách giải thích: tập có nhãn bác sĩ được chọn ra từ những ca nhiều "
      "bệnh hơn mức thường gặp (không ca nào lành), và báo cáo có thể không nhắc tới các dấu hiệu nhẹ mà người gán "
      "nhãn vẫn đánh dấu. Nhóm chưa tách được phần nào do nguyên nhân nào. Dù do nguyên nhân nào, tập chấm cũng không "
      "đại diện cho toàn bộ dữ liệu, nên điểm AUC chỉ nói về chính tập này.")
    K.append(("hinh", "tylenhan", H["ty_le_nhan"], 15.0, "Tỷ lệ ca dương tính theo bệnh: nhãn bác sĩ và nhãn LLM"))
    h2("2.6. Nhận xét và định hướng xử lý")
    p("Các đặc điểm trên dẫn tới những quyết định thiết kế ở các chương sau:")
    ul("Ảnh không đồng nhất, nên mỗi mặt phẳng chọn một series nhạy dịch theo cờ trong train_series.csv (không theo "
       "tên series, vì tên có thể đã bị thay), lấy 24 lát cách đều, đưa về 256 × 256 và chuẩn hoá cường độ theo cả "
       "series (mục 4.2).",
       f"{f(100 * n_khong / du['n_study'], 1)}% số ca chỉ có báo cáo, nên nhãn được rút từ báo cáo; báo cáo đa "
       "ngôn ngữ và có câu phủ định, nên dùng LLM thay vì chỉ dùng từ điển (mục 3.2).",
       "Mỗi bệnh hiện trên mặt phẳng khác nhau và chỉ trên vài lát, nên mô hình dùng cả ba mặt phẳng và gom các lát "
       "bằng attention riêng cho từng bệnh (mục 3.3, 3.4).",
       "Tập chấm nhỏ và lệch, nên mọi kết quả đều kèm khoảng tin cậy bootstrap và mọi lựa chọn được chốt trước khi "
       "chấm (mục 3.5).",
       f"Dữ liệu nặng {nk['dung_luong_gb']} GB, nên ảnh được xử lý trên GPU đám mây và chỉ kết quả được tải về máy.")

    # ------------------------------------------------ chuong 3
    h1("CHƯƠNG 3. CƠ SỞ LÝ THUYẾT VÀ GIẢI PHÁP")
    h2("3.1. Tổng quan giải pháp")
    p("Giải pháp gồm hai nhánh gặp nhau ở bước huấn luyện ({h:tongquan}). Nhánh nhãn biến báo cáo chẩn đoán của "
      f"{so(n_khong)} ca thành nhãn mềm cho 12 bệnh. Nhánh ảnh đọc DICOM, chọn series và chuẩn hoá thành khối ảnh cố "
      f"định cho mỗi ca. Mô hình được phát triển qua hai giai đoạn. Ở giai đoạn 1, {du['n_gold']} ca có nhãn bác sĩ được "
      "chia 5 fold để kiểm định chéo; ở giai đoạn 2, chúng được giữ riêng hoàn toàn, chỉ dùng để chấm.")
    K.append(("hinh", "tongquan", H["tong_quan"], 13.5, "Sơ đồ tổng quan hệ thống"))
    h2("3.2. Nhãn yếu từ báo cáo chẩn đoán")
    p("Học với nhãn yếu (weak supervision) là dùng nhãn được sinh tự động, có nhiễu nhưng rẻ, thay cho nhãn do "
      "chuyên gia gán {r:snorkel}. Ở đây nguồn nhãn yếu là báo cáo chẩn đoán đi kèm mỗi ca. Nhóm thử hai cách rút nhãn.")
    h3("3.2.1. Từ điển luật")
    p("Cách đầu tiên là bộ từ điển tự viết: với mỗi bệnh, liệt kê từ khoá ở 9 ngôn ngữ, kèm luật phủ định (ví dụ "
      "\"không thấy rách\") và biến thể hình thái của từng ngôn ngữ. Mỗi ô (ca, bệnh) nhận giá trị 1, 0 hoặc \"chưa rõ\". "
      f"Trên {du['n_gold']} ca có nhãn bác sĩ, từ điển chỉ đưa ra kết luận cho {pt(D['llm']['do_phu_tu_dien_tren_gold'])} "
      f"số ô, và đạt AUC {f(llm['tu_dien']['macro_auc'])} khi so với bác sĩ. Dù vậy, chỉ thêm nhãn từ điển cho "
      f"{so(n_khong)} ca đã nâng điểm mô hình từ {f(D['nguon_nhan']['A'])} lên {f(D['nguon_nhan']['B'])}.")
    h3("3.2.2. Nhãn từ mô hình ngôn ngữ lớn")
    p("Cách thứ hai là để mô hình ngôn ngữ lớn (LLM) đọc báo cáo và cho xác suất từng bệnh. Mô hình chính được huấn "
      "luyện bằng một bộ nhãn LLM do cộng đồng công bố {r:llmlabels}: nhãn mềm trong khoảng 0–1, phủ 100% số ca, "
      f"đạt AUC {f(llm['cong_khai']['macro_auc'])} so với bác sĩ. Vì bộ nhãn này không ghi rõ dùng LLM nào và prompt "
      "nào, nhóm tự gán lại nhãn cho cả 4.407 báo cáo bằng Gemini Flash {r:gemini} với prompt tự viết để kiểm chứng "
      "(mục 4.4 và 5.4).")
    h2("3.3. Giai đoạn 1: DINOv2 đóng băng")
    p("DINOv2 {r:dinov2} là mô hình Vision Transformer học tự giám sát trên khoảng 142 triệu ảnh tự nhiên, cho đặc "
      "trưng dùng được cho nhiều tác vụ mà không cần huấn luyện lại. Ở giai đoạn 1, DINOv2 được giữ đóng băng: mỗi lát "
      "MRI đi qua mạng một lần để thành một vector đặc trưng (384 chiều với ViT-S/14). Phần huấn luyện chỉ là các "
      "\"head\" nhỏ phía sau, nên toàn bộ vòng thí nghiệm chạy được trên laptop 8 GB.")
    p("Mỗi ca là một túi (bag) gồm nhiều lát, và bệnh thường chỉ thấy trên vài lát. Đây là bài toán học đa thể hiện "
      "(multiple instance learning, MIL). Head dùng gom chú ý (attention pooling) {r:ilse}: mạng học một trọng số "
      "cho từng lát, rồi lấy tổng có trọng số của các vector để ra vector đại diện cho cả ca. Nhóm dùng attention "
      "riêng cho từng bệnh, để mỗi bệnh tự chọn lát của mình. Năm head được huấn luyện: ba head chỉ nhìn một mặt "
      "phẳng, hai head nhìn cả ba mặt phẳng với hai cỡ backbone. Kết quả của năm head được gộp theo hạng (mục 3.5.3).")
    h2("3.4. Giai đoạn 2: CNN 2.5D với attention riêng từng bệnh")
    p("Đặc trưng của DINOv2 học từ ảnh thường nên không nhạy với tổn thương nhỏ trên MRI; giai đoạn 1 dừng ở mức "
      f"AUC khoảng {f(D['seed']['gop'], 2)}. Giai đoạn 2 huấn luyện toàn bộ mạng tích chập (CNN) trên chính ảnh MRI "
      "({h:kientruc}). Backbone là EfficientNet-B3 {r:effnet} và EfficientNetV2-B2 {r:effv2}, khởi tạo từ trọng số "
      "ImageNet qua thư viện timm {r:timm}.")
    K.append(("hinh", "kientruc", H["kien_truc"], 14.5, "Kiến trúc CNN 2.5D với attention riêng từng bệnh"))
    p("Ý tưởng 2.5D: ghép ba lát liền kề thành ba kênh màu của một ảnh, nên CNN hai chiều vẫn thấy được một phần "
      "chiều sâu mà chi phí chỉ như xử lý ảnh 2D. Gọi một ca có K cửa sổ (khi huấn luyện K = 3 × 8 = 24, khi suy luận "
      "mỗi lát làm tâm một cửa sổ nên K = 3 × 24 = 72). Cửa sổ thứ k là ảnh ba kênh xₖ ∈ ℝ³ˣ²⁵⁶ˣ²⁵⁶ thuộc mặt phẳng "
      "p(k). Backbone E biến nó thành một vector, cộng thêm vector nhúng học được của mặt phẳng rồi chuẩn hoá:")
    ct("fₖ = LayerNorm( E(xₖ) + e_p(k) ),   fₖ ∈ ℝᵈ", "3.1")
    p("với d = 1.536 (EfficientNet-B3) hoặc 1.408 (EfficientNetV2-B2). Một mạng nhỏ hai lớp cho điểm chú ý của mọi "
      "bệnh cùng lúc, rồi softmax được lấy theo chiều cửa sổ, riêng cho từng bệnh l:")
    ct("sₖ = U · tanh( W fₖ + b ) + c,   sₖ ∈ ℝ¹²,   W ∈ ℝ²⁵⁶ˣᵈ", "3.2")
    ct("aₖ,ₗ = exp(sₖ,ₗ) / Σⱼ exp(sⱼ,ₗ)", "3.3")
    ct("gₗ = Σₖ aₖ,ₗ · fₖ,    zₗ = wₗ · gₗ + bₗ,    pₗ = σ(zₗ)", "3.4")
    p("Cửa sổ của mặt phẳng bị thiếu nhận điểm −∞ trước softmax nên có trọng số 0. Điểm khác với attention pooling "
      "gốc {r:ilse} là mỗi bệnh có một cột trọng số aₖ,ₗ và một vector gₗ riêng; logit zₗ của bệnh l chỉ phụ thuộc "
      "gₗ, nên lát quan trọng của bệnh phổ biến không lấn lát quan trọng của bệnh hiếm. Hàm mất mát là binary "
      "cross-entropy trên nhãn mềm yₗ ∈ [0, 1] (nhãn LLM chia 100), lấy trung bình trên 12 bệnh:")
    ct("ℒ = −(1/12) Σₗ [ yₗ log pₗ + (1 − yₗ) log(1 − pₗ) ]", "3.5")
    p("Trọng số aₖ,ₗ còn cho biết cửa sổ nào được dùng để kết luận bệnh nào; web demo và mục 5.6 dùng nó để chỉ ra "
      "lát cắt mô hình chú ý.")
    h2("3.5. Đánh giá và gộp mô hình")
    h3("3.5.1. Macro AUC")
    p("AUC của một bệnh là xác suất một ca dương tính ngẫu nhiên được mô hình xếp điểm cao hơn một ca âm tính ngẫu "
      "nhiên {r:hanley}; 0,5 là đoán mò, 1,0 là xếp hạng hoàn hảo. Macro AUC là trung bình AUC của 12 bệnh. AUC không "
      "phụ thuộc ngưỡng và không đổi khi tỷ lệ bệnh thay đổi, trong khi độ chính xác (accuracy) của một mô hình "
      "luôn đoán \"không bệnh\" đã khá cao chỉ vì đa số ô là âm tính.")
    h3("3.5.2. Khoảng tin cậy bootstrap")
    p(f"Với {du['n_gold']} ca, một con số AUC đơn lẻ dao động nhiều. Mọi kết quả đều kèm khoảng tin cậy 95% bằng "
      "bootstrap {r:efron}: lấy mẫu lại có hoàn lại 58 ca (1.000 lần ở giai đoạn 1, 2.000 lần ở giai đoạn 2), tính "
      "macro AUC mỗi lần, lấy phân vị 2,5% và 97,5%; mẫu nào làm một bệnh mất hết ca dương hoặc ca âm thì bị lấy lại. "
      "Khi so hai mô hình, dùng bootstrap bắt cặp: cùng một mẫu ca cho cả hai, tính hiệu số. Nếu khoảng tin cậy của "
      "hiệu số không chứa 0 thì chênh lệch được coi là có thật.")
    h3("3.5.3. Gộp theo hạng")
    p("Các mô hình huấn luyện riêng nên thang xác suất không cùng gốc; lấy trung bình xác suất sẽ để mô hình \"tự tin\" "
      "hơn lấn át. Vì AUC chỉ quan tâm thứ hạng, nhóm gộp theo hạng: với từng bệnh, xếp hạng các ca theo xác suất của "
      "từng mô hình rồi lấy trung bình hạng, trọng số bằng nhau. Cách này cần cả một tập ca nên chỉ dùng khi chấm; web "
      "demo nhận từng ca nên lấy trung bình xác suất (mục 4.5).")
    h3("3.5.4. Chốt trước khi chấm")
    p("Giai đoạn 1 cho thấy (mục 5.2) con số sẽ lạc quan nếu các lựa chọn được quyết định "
      "dựa trên điểm của chính tập chấm. Vì vậy ở giai đoạn 2, số epoch, cách gộp và danh sách mô hình cuối đều được ghi vào "
      "file và commit lên GitHub trước khi có kết quả; điểm được báo cáo đúng như số đo, kể cả khi thấp hơn mong đợi.")

    # ------------------------------------------------ chuong 4
    h1("CHƯƠNG 4. XÂY DỰNG VÀ TRIỂN KHAI HỆ THỐNG")
    h2("4.1. Môi trường và công cụ")
    p("Hệ thống viết bằng Python 3.12, quản lý thư viện bằng uv. Mô hình dùng PyTorch {r:pytorch} và timm; đọc DICOM "
      "bằng pydicom {r:pydicom}; tính AUC bằng scikit-learn {r:sklearn}; web demo dùng FastAPI {r:fastapi}. Phần giai "
      "đoạn 1 và đánh giá chạy trên laptop RTX 4060 8 GB; phần trích đặc trưng và huấn luyện CNN chạy trên notebook "
      "GPU đám mây miễn phí với 2 GPU T4. Mã nguồn chia thành các script đánh số theo thứ tự chạy, mọi số liệu trong báo cáo được "
      "tính lại từ file kết quả.")
    h2("4.2. Tiền xử lý dữ liệu")
    p("Một ca đi qua bảy bước dưới đây để từ file DICOM thô thành khối ảnh đầu vào của mô hình ({h:tienxuly}). Toàn bộ "
      "bước này chạy một lần trên GPU đám mây và lưu bộ nhớ đệm; web demo gọi lại đúng hàm đọc ảnh lúc huấn luyện "
      "để tránh lệch xử lý giữa hai nơi.")
    ul(f"**Lập bảng ca và chia tập.** Bảng ca gồm {so(du['n_study'])} ca và {so(du['n_series'])} series. Tập được chia "
       "theo ca, không theo series: khi thử chia theo series, 100% ca kiểm tra có series nằm bên tập huấn luyện.",
       "**Chặn rò rỉ qua nhãn yếu.** Một ca ngoài tập chấm có báo cáo trùng từng ký tự với một ca có nhãn bác sĩ, nên "
       "nhãn yếu của nó mang luôn đáp án. Ca này bị loại khỏi tập huấn luyện ở cả hai giai đoạn.",
       f"**Chọn series.** Mỗi mặt phẳng chọn một series nhạy dịch theo cờ trong train_series.csv (ưu tiên chuỗi xung "
       f"PD hoặc T2 xoá mỡ), vì dịch và phù tuỷ xương hiện rõ trên các chuỗi này; {nk['chon_series']}% số ca có đủ "
       "series phù hợp ở cả ba mặt phẳng. Nếu series được chọn đọc lỗi, chương trình lùi sang series kế tiếp của "
       "cùng mặt phẳng.",
       "**Đọc và sắp lát.** Các file DICOM của series được đọc bằng pydicom và sắp theo số thứ tự lát "
       "(InstanceNumber) ghi trong header, không theo tên file.",
       "**Chuẩn hoá cường độ.** Cường độ được cắt theo phân vị 1 và 99 tính trên toàn bộ điểm ảnh của cả series rồi "
       "đưa về khoảng 0–1 ({h:tienxuly}b). Tính theo cả series chứ không theo từng lát để giữ độ sáng tương đối giữa "
       "các lát; tính trước khi đổi kích thước vì phép nội suy tạo ra giá trị mới ở biên và làm lệch phân vị.",
       "**Lấy mẫu lát và đổi kích thước.** Lấy 24 lát cách đều theo chiều sâu ({h:tienxuly}e), đổi mỗi lát về "
       "256 × 256 bằng nội suy song tuyến, lưu dạng uint8. Mỗi ca thành một khối 3 × 24 × 256 × 256 kèm cờ cho "
       "biết mặt phẳng nào có mặt.",
       "**Cửa sổ 2.5D.** Khi đưa vào CNN, ba lát liền kề được ghép thành ba kênh màu của một ảnh "
       "({h:tienxuly}d); phần màu trên ảnh ghép là chỗ ba lát khác nhau, tức thông tin về chiều sâu.")
    K.append(("hinh", "tienxuly", H["tien_xu_ly"], 15.5, "Các bước tiền xử lý trên series Sagittal của ca mẫu"))
    h2("4.3. Huấn luyện CNN 2.5D trên GPU đám mây")
    p(f"Mỗi mô hình được huấn luyện trên nhãn LLM của {so(cn['mo_hinh'][0]['n_train'])} ca; {nk['n_val_llm']} ca nhãn "
      "LLM khác được giữ lại để theo dõi, không dùng ca nào có nhãn bác sĩ. Cấu hình chính nằm trong {b:cauhinh}.")
    K.append(("bang", "cauhinh", "Cấu hình huấn luyện CNN 2.5D", ["Tham số", "Giá trị"], [
        ["Backbone", "EfficientNet-B3, EfficientNetV2-B2 (pretrained ImageNet)"],
        ["Số tham số", "11,1 triệu (B3), 9,1 triệu (V2-B2); phần nhúng mặt phẳng, attention và 12 đầu ra chiếm khoảng 4%"],
        ["Đầu vào mỗi ca", "3 mặt phẳng × 24 lát × 256 × 256; cửa sổ 3 lát"],
        ["Cửa sổ mỗi mặt phẳng khi huấn luyện", "8, lấy ngẫu nhiên trải đều theo chiều sâu"],
        ["Batch", "4 ca, cộng dồn gradient 2 bước (hiệu dụng 8 ca)"],
        ["Bộ tối ưu", "AdamW {r:adamw}; lr 2·10⁻⁴ (backbone), 10⁻³ (head); weight decay 10⁻²"],
        ["Lịch learning rate", "Khởi động 0,5 epoch, sau đó cosine"],
        ["Số epoch", "12, cố định từ trước"],
        ["Trọng số dùng để chấm", "EMA (hệ số 0,999) ở epoch cuối"],
        ["Độ chính xác số", "Tự động fp16 (autocast) + GradScaler"],
    ], [6.0, 10.0]))
    p("Danh sách 6 mô hình của bản cuối được chốt trước ({b:sauModel}). Hai lần chạy EfficientNet-B3 gặp lỗi kỹ "
      "thuật (dự đoán ra NaN sau epoch 1, nhiều khả năng do tràn số fp16) và được chạy lại đúng cấu hình cũ; sự cố "
      "được ghi vào file chốt trước khi xem điểm của mô hình bị lỗi.")
    ten_mh = lambda s: s.replace("tf_efficientnetv2_b2", "EfficientNetV2-B2").replace("efficientnet_b3", "EfficientNet-B3")
    rows = []
    for m in D["cuoi_day"]["danh_sach"]:
        bb, seed = m["mo_hinh"].rsplit("_s", 1)
        rows.append([str(m["so"]), ten_mh(bb), seed, "LLM công khai" if m["nhan"] == "cong_khai" else "LLM tự gán",
                     f(tm[m["so"]]["auc"])])
    K.append(("bang", "sauModel", "Sáu mô hình CNN 2.5D trong bản cuối (mỗi mô hình lấy trọng số EMA ở epoch 12)",
              ["#", "Backbone", "Seed", "Nhãn huấn luyện", "AUC riêng"], rows, [1.0, 5.0, 2.0, 4.5, 3.0]))
    p(f"Ở hai lần chạy đầu, quá trình huấn luyện rất chậm (khoảng 10 giây mỗi bước) và bị hệ thống buộc dừng (Killed). "
      f"Nhóm "
      f"viết một notebook chẩn đoán riêng và đo được rằng nn.DataParallel trên 2 GPU T4 mất {nk['dp_giay']} giây mỗi "
      f"bước, so với {nk['mot_gpu_giay']} giây khi chạy một GPU. Bản cuối cho mỗi GPU huấn luyện một mô hình độc lập "
      f"trong tiến trình riêng; hai mô hình xong trong {nk['gio_kaggle']} giờ.")
    h2("4.4. Tự gán nhãn bằng Gemini Flash")
    p("Để có bộ nhãn của riêng mình và kiểm tra bộ nhãn công khai, nhóm viết một prompt đặt "
      "LLM vào vai bác sĩ chẩn đoán hình ảnh cơ xương khớp, định nghĩa từng bệnh và quy định thang điểm 0–100: dấu "
      "hiệu được khẳng định cho khoảng 95, được khẳng định là không có hoặc bình thường khoảng 3, không được nhắc tới "
      "khoảng 8, viết nước đôi (\"có thể\", \"không loại trừ\") khoảng 50, \"nhiều khả năng\" khoảng 75. Một số "
      "trường hợp ranh giới có điểm riêng, ví dụ thoái hoá dạng nhầy của dây chằng mà không rách khoảng 30, tín hiệu "
      "thoái hoá trong sụn chêm mà không rách khoảng 25. Prompt cũng dặn chỉ chấm theo chữ trong báo cáo, không suy "
      "từ việc các bệnh hay đi kèm nhau.")
    p("LLM phải trả về đúng một đối tượng JSON, mỗi báo cáo 12 số nguyên theo thứ tự cố định; điểm chia 100 thành "
      "nhãn mềm. Báo cáo được gửi theo lô 25 báo cáo, chạy song song 4 lô. Lô nào trả sai định dạng (thiếu khoá, "
      "sai số cột, số ngoài 0–100) được gửi lại, tối đa 3 lần, và kết quả từng lô được lưu để chạy tiếp khi bị "
      "ngắt. LLM chỉ thấy văn bản báo cáo, không thấy mã ca hay nhãn bác sĩ; ca có nhãn bác sĩ nằm lẫn trong cùng lô "
      "với các ca khác, và prompt được chốt trước khi so với bác sĩ.")
    h2("4.5. Web demo")
    h3("4.5.1. Kiến trúc")
    p("Web demo gồm ba lớp ({h:webkt}). Lớp giao diện là một trang HTML tĩnh chạy trên trình duyệt. Lớp máy chủ là "
      "ứng dụng FastAPI {r:fastapi} chỉ nghe tại địa chỉ 127.0.0.1:8790, nên máy khác trong mạng không gọi được. Lớp "
      "suy luận nạp sẵn 5 mô hình CNN lên GPU khi khởi động và làm ba việc: gom file theo series và xác định mặt "
      "phẳng, đọc và chuẩn hoá ca, chạy mô hình rồi đóng gói kết quả. Bước đọc và chuẩn hoá gọi lại đúng hàm tiền xử "
      "lý của notebook huấn luyện (nạp trực tiếp từ file mã, không chép lại), để ảnh vào mô hình lúc demo giống hệt "
      "lúc huấn luyện.")
    K.append(("hinh", "webkt", H["web_kien_truc"], 14.0, "Kiến trúc ba lớp của web demo"))
    p("Ca tải lên không có bảng train_series.csv như lúc huấn luyện, nên lớp suy luận tự suy ra hai thông tin từ "
      "header DICOM. Mặt phẳng được tính từ trường ImageOrientationPatient: tích có hướng của hai vectơ hướng hàng và "
      "cột cho pháp tuyến của lát, trục có thành phần lớn nhất là mặt phẳng. Series nhạy dịch được đoán từ mô tả "
      "series (chứa T2, PD, STIR, FS, SPAIR… thì cộng điểm, chứa T1 thì trừ điểm) và xếp lên trước. Thiếu mặt phẳng "
      "nào thì trang hiện cảnh báo, mô hình vẫn chạy với các mặt phẳng còn lại. File tải lên nằm trong thư mục tạm "
      "và bị xoá ngay khi xử lý xong, kể cả khi có lỗi.")
    h3("4.5.2. Luồng một lượt dự đoán và API")
    p("{h:webtt} vẽ một lượt dự đoán. Máy chủ chạy lần lượt 5 mô hình trên cùng một khối ảnh, lấy sigmoid của logit "
      "rồi lấy trung bình xác suất của 5 mô hình; trọng số chú ý cũng được lấy trung bình. Gộp theo hạng như khi chấm "
      "(mục 3.5.3) không dùng được ở đây vì chỉ có một ca. Kết quả gửi về gồm cả các lát sau chuẩn hoá (ảnh JPEG mã "
      f"hoá base64), nên người xem thấy đúng ảnh mô hình đã nhận. Trên RTX 4060, phần chạy 5 mô hình mất khoảng "
      f"{nk['web_giay']} giây mỗi ca. {{b:api}} liệt kê ba địa chỉ của máy chủ.")
    K.append(("hinh", "webtt", H["web_tuan_tu"], 15.0, "Trình tự xử lý một lượt dự đoán"))
    K.append(("bang", "api", "Các API của máy chủ web demo", ["Phương thức, đường dẫn", "Đầu vào", "Đầu ra"], [
        ["GET /", "Không", "Trang giao diện HTML"],
        ["GET /api/ca-mau", "Không", "Danh sách 6 ca mẫu: mã, tên, số bệnh và tên bệnh bác sĩ ghi nhận"],
        ["POST /api/du-doan", "Mã ca mẫu (?mau=), hoặc form nhiều file DICOM hay zip, tối đa 400 MB",
         ("JSON: 12 bệnh (xác suất trung bình, xác suất từng mô hình, nhãn bác sĩ nếu có, trọng số chú ý 3 × 24, lát "
          "chính), 3 × 24 lát JPEG, cảnh báo, thời gian suy luận. Lỗi đầu vào trả mã 400 kèm lời giải thích")],
    ], [3.8, 5.2, 7.0]))
    h3("4.5.3. Giao diện")
    p("Người dùng chọn một trong 6 ca mẫu có nhãn bác sĩ hoặc tải lên file DICOM nén zip ({h:web}). Ca mẫu không nằm "
      "trong tập huấn luyện và được chọn để minh hoạ (những ca mô hình xếp đúng phần lớn bệnh), nên không đại diện "
      "cho kết quả chung ở chương 5; mục 5.6 phân tích thêm một ca mô hình làm kém. Trang hiển thị 12 xác suất xếp "
      "giảm dần, so với nhãn bác sĩ nếu có (dấu ✓ đánh dấu bệnh bác sĩ ghi nhận mà mô hình xếp vào nhóm đầu), kèm "
      "khung xem lát cắt mà mô hình chú ý nhất cho bệnh đang chọn. Web chạy 5 thay vì 6 mô hình vì trọng số của mô "
      "hình #6 bị ghi đè trên máy chủ đám mây khi chạy lại #5; dự đoán của #6 đã được tải về trước nên vẫn được chấm.")
    K.append(("hinh", "web", H["web"], 16.0, "Giao diện web demo với một ca mẫu"))
    h2("4.6. Kiểm thử")
    p(f"Dự án có {nk['n_test']} kiểm thử tự động (pytest), chạy trong khoảng 20 giây và không cần ảnh thật. Kiểm "
      "thử tập trung vào loại lỗi \"âm thầm\": chương trình vẫn chạy và cho ra một con số hợp lý, chỉ có con số đó sai. "
      "Vì vậy phần lớn kiểm thử dựng sẵn một dữ liệu nhỏ mà đáp án biết trước (mảng numpy, file DICOM giả, câu báo cáo "
      "lấy từ dữ liệu thật) rồi kiểm đúng hành vi dễ hỏng. {b:kiemthu} chia các kiểm thử theo phần của hệ thống.")
    K.append(("bang", "kiemthu", "Kiểm thử tự động theo phần của hệ thống", ["Phần", "Số kiểm thử", "Lỗi âm thầm được chặn (ví dụ)"], [
        ["Chia tập, thứ tự ca", "10", "Ca lọt giữa tập huấn luyện và tập chấm; lệch thứ tự ca khiến dự đoán bị chấm bằng nhãn ca khác"],
        ["Đọc ảnh, tiền xử lý", "27", "Chọn nhầm series T1; cửa sổ 3 lát không liền kề; cắt phân vị sai; bộ nhớ đệm ghi tiếp lệch vị trí"],
        ["Rút nhãn từ báo cáo", "26", "Câu phủ định hoặc câu \"bình thường\" bị đọc thành dương tính; câu không nhắc tới bị coi là âm tính thay vì chưa rõ"],
        ["Mô hình", "34", "Pooling quên mặt nạ lát đệm; softmax ra NaN khi ca thiếu cả ba mặt phẳng; logit bệnh này dùng vector của bệnh khác"],
        ["Đánh giá, gộp", "17", "Bệnh chỉ có một lớp bị bỏ âm thầm khỏi trung bình; nhãn chưa gán bị tính là âm tính; gộp theo hạng lúc suy luận khác lúc chấm"],
        ["Web demo", "2", "Suy sai mặt phẳng hoặc series nhạy dịch từ header DICOM"],
        ["**Tổng**", f"**{nk['n_test']}**", ""],
    ], [3.6, 2.2, 10.2]))

    # ------------------------------------------------ chuong 5
    h1("CHƯƠNG 5. THỰC NGHIỆM VÀ KẾT QUẢ")
    h2("5.1. Thiết lập đánh giá")
    p(f"Mọi con số trong chương này là macro AUC trên {du['n_gold']} ca có nhãn bác sĩ, kèm khoảng tin cậy bootstrap "
      "95% trong ngoặc vuông. Ở giai đoạn 1, các ca này được chia 5 fold theo ca và điểm là AUC ngoài fold; ở giai "
      "đoạn 2, chúng không bao giờ vào huấn luyện nên điểm là điểm ngoài mẫu trực tiếp.")
    h2("5.2. Hành trình cải tiến")
    p("{h:hanhtrinh} tóm tắt các bước. Mỗi bước chỉ đổi một thứ. Dùng cả ba mặt phẳng, thêm nhãn yếu cho "
      f"{so(n_khong)} ca, rồi thay từ điển bằng nhãn LLM đều tăng điểm rõ rệt. Bộ gộp 5 head từng đạt "
      f"{f(ce['cv_cu_bao_cao'])} theo cách đo cũ, nhưng khi kiểm tra lại cách đo, nhóm tìm ra hai lỗi: (1) lúc đo, mỗi fold chọn epoch "
      "có AUC cao nhất ngay trên các ca đang chấm; (2) lúc huấn luyện mô hình cuối, nhãn mềm khiến hàm AUC trả NaN nên "
      "chương trình âm thầm chuyển sang chọn theo loss và huấn luyện tới khoảng epoch 80, trong khi head đạt đỉnh "
      "quanh epoch 5 rồi giảm dần.")
    p(f"Cách sửa: chốt {ce['epoch_chot']} epoch, chọn bằng 10% ca nhãn LLM giữ riêng, không nhìn ca có nhãn bác sĩ, và "
      f"gộp 3 seed. Điểm đo lại đúng cách của giai đoạn 1 là {f(D['seed']['gop'])} {ci(D['seed']['ci_gop'])}. "
      f"Giai đoạn 2 tăng lên {f(cn['auc'])} với 2 CNN và {f(g6['auc'])} với bản cuối 6 CNN.")
    K.append(("hinh", "hanhtrinh", H["hanh_trinh"], 14.5,
              "Macro AUC qua từng bước (cột xám: đo theo cách cũ, chỉ để so các bước)"))
    h2("5.3. Giai đoạn 2 so với giai đoạn 1")
    mh = cn["mo_hinh"]
    K.append(("bang", "gd2", "So sánh giai đoạn 2 và giai đoạn 1", ["Mô hình", "Macro AUC", "KTC 95%"], [
        ["EfficientNet-B3, seed 42", f(mh[0]["auc"]), ci(mh[0]["ci"])],
        ["EfficientNetV2-B2, seed 7", f(mh[1]["auc"]), ci(mh[1]["ci"])],
        ["**CNN 2.5D, gộp 2 mô hình**", f"**{f(cn['auc'])}**", ci(cn["ci"])],
        ["DINOv2 đóng băng, 5 head × 3 seed", f(D["seed"]["gop"]), ci(D["seed"]["ci_gop"])],
        ["Gộp 50/50 CNN + DINOv2 (chốt trước)", f(cn["gop_50_50"]["auc"]), ci(cn["gop_50_50"]["ci"])],
    ], [8.0, 3.5, 4.5]))
    p(f"CNN hơn DINOv2 {f(cn['chenh']['chenh'])}, khoảng tin cậy bắt cặp {ci_dau(cn['chenh']['ci'])} không chứa 0, nên "
      f"đây là chênh lệch thật dù chỉ có {du['n_gold']} ca ({{b:gd2}}). Phép gộp 50/50 với DINOv2 cũng được chốt trước, "
      "nhưng lại kéo điểm xuống vì DINOv2 yếu hơn quá xa; kết quả này vẫn được báo cáo đúng như số đo.")
    p(f"{{h:epoch}} cho thấy vì sao không cần nhìn ca có nhãn bác sĩ để chọn epoch: AUC trên {nk['n_val_llm']} ca nhãn "
      "LLM giữ lại đi cùng chiều với AUC trên ca bác sĩ, cả hai phẳng từ khoảng epoch 6. Lấy epoch cuối như đã chốt "
      "chỉ thấp hơn đỉnh khoảng 0,01.")
    K.append(("hinh", "epoch", H["epoch"], 15.0, "AUC theo epoch: ca bác sĩ (nét liền, chỉ ghi lại) và ca nhãn LLM giữ lại (nét đứt)"))
    h2("5.4. So sánh ba nguồn nhãn")
    rows = []
    for k in ("cong_khai", "tu_gan", "tu_dien"):
        n = llm[k]
        rows.append([n["ten"].replace("gắn", "gán"), f(n["macro_auc"]), ci(n["ci"]), pt(n["nguong_05"]["khop"]),
                     pt(n["nguong_05"]["do_nhay"]), pt(n["nguong_05"]["do_dac_hieu"])])
    K.append(("bang", "nhan", f"Nhãn yếu so với nhãn bác sĩ trên {du['n_gold']} ca",
              ["Nguồn nhãn", "AUC", "KTC 95%", "Khớp ở ngưỡng 0,5", "Độ nhạy", "Độ đặc hiệu"], rows,
              [4.4, 1.7, 3.2, 2.4, 2.0, 2.3]))
    tg = llm["tu_gan"]["tung_benh"]; ck = llm["cong_khai"]["tung_benh"]
    p(f"Không nguồn nhãn nào đúng 100%: cả hai bộ LLM khớp bác sĩ khoảng 81% số ô ({{b:nhan}}). Nhãn tự gán kém bộ "
      f"công khai {f(-D['llm']['tu_gan_tru_cong_khai']['chenh'])}, khoảng tin cậy bắt cặp "
      f"{ci_dau(D['llm']['tu_gan_tru_cong_khai']['ci'])}, chủ yếu ở viêm màng hoạt dịch ({f(tg['Synovitis'])} so với "
      f"{f(ck['Synovitis'])}) và dập xương ({f(tg['Contusion'])} so với {f(ck['Contusion'])}); nhưng hơn từ điển "
      f"{f(D['llm']['tu_gan_tru_tu_dien']['chenh'])}. Trên {so(hk['n_ca'])} ca không có nhãn bác sĩ, hai bộ LLM khớp "
      f"nhau ở {pt(hk['khop_nguong_05'])} số ô. Dù vậy, bản gộp 2 CNN huấn luyện bằng nhãn tự gán lại "
      f"đạt {f(g2t['auc'])}, ngang bản gộp 2 CNN cùng cấu hình dùng nhãn công khai ({f(g2c['auc'])}; hiệu số "
      f"{ci_dau(ss['gop_2_cnn_tu_gan']['ci'])} chứa 0). Ít nhất trong thí nghiệm này, chất lượng nhãn đo trực tiếp "
      "chưa phản ánh hết chất lượng mô hình huấn luyện từ nó.")
    p(f"So riêng từng bệnh giữa hai bộ LLM trên {so(hk['n_ca'])} ca không có nhãn bác sĩ. Với 10 bệnh, thứ tự ca "
      "của hai bộ gần như trùng nhau (tương quan hạng Spearman từ 0,80 trở lên) và tỷ lệ dương tính chênh dưới 4 "
      f"điểm phần trăm, trừ tràn dịch khớp ({pt(hk['ty_le_duong_05']['tu_gan']['Effusion'])} so với "
      f"{pt(hk['ty_le_duong_05']['cong_khai']['Effusion'])}). Hai bệnh lệch rõ là gãy xương và nhất là viêm màng hoạt dịch (tương quan "
      f"{f(hk['tuong_quan_hang']['Synovitis'])}): bộ tự gán cho {pt(hk['ty_le_duong_05']['tu_gan']['Synovitis'])} số ca "
      f"là dương tính, bộ công khai chỉ {pt(hk['ty_le_duong_05']['cong_khai']['Synovitis'])}. Ranh giới của bệnh "
      "này không rõ: prompt của nhóm tính cả dày màng hoạt dịch và viêm mỡ Hoffa, còn bộ công khai không công bố định "
      "nghĩa, nên hai LLM có thể hiểu khác nhau. Đây cũng là bệnh mô hình làm kém nhất (mục 5.5, 5.6).")
    h2("5.5. Mô hình cuối")
    K.append(("bang", "cuoi", "Kết quả mô hình cuối và các nhóm báo cáo kèm (đều chốt trước)",
              ["Nhóm mô hình", "Macro AUC", "KTC 95%"], [
                  ["**Gộp 6 CNN (mô hình chính)**", f"**{f(g6['auc'])}**", ci(g6["ci"])],
                  ["Gộp 4 CNN nhãn công khai (#1–#4)", f(g4["auc"]), ci(g4["ci"])],
                  ["Gộp 2 CNN nhãn tự gán (#5–#6)", f(g2t["auc"]), ci(g2t["ci"])],
                  ["Gộp 2 CNN nhãn công khai (#1–#2, mốc cũ)", f(g2c["auc"]), ci(g2c["ci"])],
              ], [8.0, 3.5, 4.5]))
    p(f"Mô hình cuối gộp theo hạng 6 CNN đạt macro AUC {f(g6['auc'])} {ci(g6['ci'])} ({{b:cuoi}}), hơn mốc 2 CNN "
      f"{f(ss['gop_6_cnn']['chenh'])} với khoảng tin cậy bắt cặp {ci_dau(ss['gop_6_cnn']['ci'])}. Mức tăng nhỏ: "
      "khoảng tin cậy chỉ vừa không chứa 0. Không mô hình nào bị loại vì điểm thấp.")
    p("{b:tungbenh} và {h:tungbenh} so từng bệnh. Mức tăng lớn nhất so với giai đoạn 1 nằm ở các tổn thương nhỏ, khu "
      f"trú trên vài lát: dây chằng chéo trước ({f(dino['ACL'])} → {f(D['cuoi']['tung_benh']['ACL'])}), dập xương và "
      "dây chằng bên trong. Giả thuyết của nhóm là đặc trưng học từ ảnh thường không nhìn ra các tổn thương này, còn CNN "
      "huấn luyện trên MRI thì học được; giả thuyết này chưa được kiểm chứng riêng. Viêm màng hoạt dịch vẫn là bệnh "
      f"yếu nhất ({f(D['cuoi']['tung_benh']['Synovitis'])}).")
    K.append(("hinh", "tungbenh", H["tung_benh"], 15.0, "AUC từng bệnh: DINOv2 đóng băng so với mô hình cuối"))
    kb, tt_ = T["ktc_benh"], T["pt"]["tap_trung"]
    rows = [[TEN_VI[k], str(duong[k]), f(dino[k]), f(cnn2[k]), f(D["cuoi"]["tung_benh"][k]), ci(kb[k]), pt(tt_[k])]
            for k in TEN_VI]
    K.append(("bang", "tungbenh", ("AUC từng bệnh qua các giai đoạn, KTC 95% của mô hình cuối và mức tập trung chú ý "
              "(mục 5.6)"), ["Bệnh", "Số ca dương", "DINOv2 (GĐ 1)", "2 CNN", "6 CNN (cuối)", "KTC 95%", "Chú ý top 5"],
              rows, [4.0, 1.5, 1.9, 1.5, 1.9, 2.9, 2.0]))
    p(f"Khoảng tin cậy từng bệnh rộng hơn nhiều so với macro AUC, nhất là ở bệnh ít ca dương: dây chằng bên trong "
      f"chỉ có {duong['MCL']} ca dương nên khoảng là {ci(kb['MCL'])}. Dù vậy, cận dưới của dây chằng chéo trước "
      f"({f(kb['ACL'][0])}) và dây chằng bên trong ({f(kb['MCL'][0])}) vẫn cao hơn điểm giai đoạn 1 của hai bệnh này "
      f"({f(dino['ACL'])} và {f(dino['MCL'])}). Ngược lại, khoảng của viêm màng hoạt dịch {ci(kb['Synovitis'])} và "
      f"thoái hoá khoang ngoài {ci(kb['Lateral OA'])} rộng tới mức chưa nói được hai bệnh này đứng ở đâu.")
    p("{h:roc} vẽ đường cong ROC của mô hình cuối cho từng bệnh. Mỗi điểm trên đường cong ứng với một ngưỡng: trục "
      "ngang là tỷ lệ ca âm tính bị báo nhầm, trục dọc là tỷ lệ ca dương tính được phát hiện; diện tích dưới đường "
      "cong là AUC. Với thoái hoá khoang trong và nang Baker, mô hình phát hiện phần lớn ca dương tính trước khi báo "
      "nhầm ca nào. Đường cong của viêm màng hoạt dịch nằm gần đường chéo hơn hẳn, khớp với AUC thấp nhất. Các đường "
      "cong có dạng bậc thang vì mỗi bệnh chỉ có 9–35 ca dương tính.")
    K.append(("hinh", "roc", H["roc"], 15.5, "Đường cong ROC từng bệnh của mô hình cuối trên 58 ca có nhãn bác sĩ"))
    h2("5.6. Phân tích một số ca")
    pt_ = T["pt"]
    cb = pt_["ca"][CA_KEM[0]]["benh"]
    xep = sorted(cb, key=lambda b: -b["p"])
    co_benh = [b for b in xep if b["bac_si"]]
    tren = [b for b in xep if not b["bac_si"] and b["p"] > min(x["p"] for x in co_benh)]
    ten = lambda b: TEN_VI[b["ma"]].split(" (")[0].lower()
    p("Macro AUC cho biết mô hình xếp hạng tốt đến đâu, nhưng không cho biết mô hình nhìn vào đâu. Mục này dùng trọng "
      "số chú ý aₖ,ₗ (mục 3.4) trên 8 ca có nhãn bác sĩ đã đóng gói cho web demo, chạy đúng đường của ca tải lên. Tám "
      "ca này không vào huấn luyện; số ca quá ít để kết luận chung, nên các nhận xét dưới đây là quan sát, chưa phải "
      "kiểm định.")
    a_tot = next(b for b in pt_["ca"][CA_TOT[0]]["benh"] if b["ma"] == CA_TOT[1])
    b_kem = next(b for b in cb if b["ma"] == CA_KEM[1])
    p(f"{{h:phantich}} đặt cạnh nhau hai ca. Ca A là ca mẫu 1 trên web: bác sĩ ghi 9 bệnh và mô hình xếp cả 9 vào 9 vị "
      f"trí đầu. Với nang Baker, chú ý dồn vào hai cụm vài lát liền nhau, một ở Sagittal và một ở Coronal (trọng số "
      f"lớn nhất {f(a_tot['lat_chinh']['trong_so'])}, gấp khoảng {round(a_tot['lat_chinh']['trong_so'] * pt_['n_cua_so'])} "
      "lần mức chia đều), các lát còn lại gần như bằng 0. Ca B không được đưa lên web: bác sĩ ghi {len(co_benh)} bệnh "
      f"({', '.join(ten(b) for b in co_benh)}). Mô hình xếp {ten(xep[0])} đầu tiên ({f(xep[0]['p'], 2)}), nhưng đặt "
      f"{len(tren)} bệnh bác sĩ không ghi ({', '.join(ten(b) for b in tren)}) lên trên hai bệnh còn lại. Với viêm màng "
      f"hoạt dịch ({f(b_kem['p'], 2)}), chú ý gần như trải đều trên cả ba mặt phẳng, trọng số lớn nhất chỉ "
      f"{f(b_kem['lat_chinh']['trong_so'])}.")
    K.append(("hinh", "phantich", H["phan_tich_ca"], 14.5,
              "Lát được chú ý nhất và trọng số chú ý trên 72 cửa sổ: ca A (nang Baker) và ca B (viêm màng hoạt dịch)"))
    p(f"Để xem đây có phải hiện tượng chung không, cột cuối {{b:tungbenh}} đo mức tập trung của chú ý: tỷ trọng của 5 cửa sổ "
      f"được chú ý nhất trong tổng {pt_['n_cua_so']} cửa sổ, trung bình trên 8 ca (nếu chia đều thì là "
      f"{pt(5 / pt_['n_cua_so'])}). Các tổn thương khu trú như rách sụn chêm, dây chằng chéo trước, dập xương và nang "
      "Baker có chú ý tập trung nhất; các bệnh lan rộng như thoái hoá và viêm màng hoạt dịch có chú ý rải hơn. Rải "
      f"không có nghĩa là kém: thoái hoá khoang trong thuộc nhóm rải nhưng có AUC {f(D['cuoi']['tung_benh']['Medial OA'])}, "
      "có lẽ vì dấu hiệu thoái hoá thấy được trên nhiều lát. Viêm màng hoạt dịch vừa có chú ý rải nhất "
      f"({pt(tt_['Synovitis'])}) vừa có AUC thấp nhất, lại là bệnh mà hai bộ nhãn LLM bất đồng nhiều nhất (mục 5.4). "
      "Nhóm cho rằng điểm yếu ở bệnh này đến từ cả nhãn huấn luyện lẫn bản chất lan toả của tổn thương; tách riêng "
      "hai nguyên nhân cần nhãn bác sĩ cho nhiều ca hơn.")
    h2("5.7. Các lỗi âm thầm đã phát hiện")
    p("Nhiều lỗi trong quá trình phát triển không báo ngoại lệ mà chỉ làm sai con số. {b:loi} liệt kê các lỗi "
      "chính và cách nhóm chặn từng lỗi.")
    K.append(("bang", "loi", "Lỗi âm thầm và cách chặn", ["Lỗi", "Dấu hiệu đo được", "Cách chặn"], [
        ["Chia tập theo series", "100% ca kiểm tra có series bên tập huấn luyện", "Chia theo ca; có kiểm thử"],
        ["Báo cáo trùng lặp", "1 ca ngoài tập chấm trùng từng ký tự với 1 ca có nhãn bác sĩ", "Loại khỏi tập huấn luyện"],
        ["Bộ nhãn công khai chép nhãn bác sĩ", "AUC 1,000; giá trị trên tập chấm toàn 0/1 tuyệt đối",
         "Không dùng; chốt chặn tự dừng khi gặp dấu hiệu này"],
        ["Chọn epoch trên tập chấm", f"Điểm {f(ce['cv_cu_bao_cao'])} đo lại còn {f(D['seed']['gop'])}",
         "Chốt epoch trước, chọn bằng tập nhãn LLM riêng"],
        ["nn.DataParallel chậm", f"{nk['dp_giay']} so với {nk['mot_gpu_giay']} giây/bước", "Mỗi GPU một mô hình"],
    ], [4.2, 6.6, 5.2]))

    # ------------------------------------------------ chuong 6
    h1("CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
    h2("6.1. Kết quả đạt được")
    ul(f"Quy trình trọn vẹn từ DICOM thô tới 12 xác suất bệnh; mô hình cuối đạt macro AUC {f(g6['auc'])} "
       f"{ci(g6['ci'])} trên {du['n_gold']} ca có nhãn bác sĩ, so với {f(D['seed']['gop'])} của giai đoạn 1.",
       f"Dùng được {so(n_khong)} ca không có nhãn bác sĩ ({f(100 * n_khong / du['n_study'], 1)}% dữ liệu) nhờ nhãn LLM "
       "rút từ báo cáo 9 ngôn ngữ; nhãn tự gán bằng Gemini Flash cho mô hình ngang nhãn công khai.",
       "Cách đánh giá trung thực (chia theo ca, chặn rò rỉ, bootstrap, chốt trước khi chấm) giúp nhóm tự phát hiện và "
       "sửa một con số lạc quan của chính mình.",
       f"Web demo chạy cục bộ, khoảng {nk['web_giay']} giây mỗi ca; {nk['n_test']} kiểm thử tự động.")
    h2("6.2. Hạn chế")
    ul(f"Đây là đánh giá nội bộ: {du['n_gold']} ca có nhãn bác sĩ đã dùng để kiểm định chéo ở giai đoạn 1 và lệch so "
       "với toàn bộ dữ liệu (mục 2.5); khoảng tin cậy rộng khoảng ±0,03 nên chênh lệch cỡ 0,01 giữa các CNN chưa "
       "phân định được.",
       "Nhãn huấn luyện là nhãn LLM, chỉ khớp bác sĩ khoảng 81% số ô, và LLM có thể đã thấy các báo cáo này khi tiền "
       "huấn luyện; xác suất chưa được hiệu chỉnh (calibration) và chưa có đánh giá lâm sàng.",
       "Chưa đo ảnh hưởng của lối tắt theo hãng máy (mục 2.3) và của việc chỉ giữ 24 lát mỗi series: series tới 320 "
       "lát có thể mất đúng lát chứa tổn thương nhỏ.",
       f"Viêm màng hoạt dịch ({f(D['cuoi']['tung_benh']['Synovitis'])}) và thoái hoá khoang ngoài "
       f"({f(D['cuoi']['tung_benh']['Lateral OA'])}) vẫn còn yếu.")
    h2("6.3. Hướng phát triển")
    ul("Đánh giá trên một tập có nhãn bác sĩ lớn hơn và độc lập, ví dụ dữ liệu từ bệnh viện khác.",
       "Tăng độ phân giải và số lát cho tổn thương nhỏ; thử backbone 3D hoặc huấn luyện riêng cho từng mặt phẳng.",
       "Hiệu chỉnh xác suất, chọn ngưỡng theo từng bệnh; gộp nhiều LLM hoặc để bác sĩ duyệt những ô hai bộ nhãn bất đồng.")
    return K


# ============================================================ ghi docx
def dat_run(r, dam=False, nghieng=False, co=None):
    r.bold = dam or None
    r.italic = nghieng or None
    if co:
        r.font.size = Pt(co)


def viet_chu(par, chu: str, dam_ca=False, nghieng=False, co=None):
    """Viet chu vao doan, **...** la dam."""
    for i, phan in enumerate(re.split(r"\*\*", chu)):
        if phan:
            dat_run(par.add_run(phan), dam=dam_ca or i % 2 == 1, nghieng=nghieng, co=co)


def doan_than(par, before=0, after=120, thut=True):
    pf = par.paragraph_format
    pf.space_before, pf.space_after = Pt(before / 20), Pt(after / 20)
    if thut:
        pf.first_line_indent = Cm(1.27)


def ke_bang(tbl):
    tblPr = tbl._tbl.tblPr
    vien = OxmlElement("w:tblBorders")
    for c in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{c}")
        e.set(qn("w:val"), "single"), e.set(qn("w:sz"), "8"), e.set(qn("w:space"), "0"), e.set(qn("w:color"), "000000")
        vien.append(e)
    tblPr.append(vien)
    mar = OxmlElement("w:tblCellMar")
    for c, w in (("left", 100), ("right", 100), ("top", 30), ("bottom", 30)):
        e = OxmlElement(f"w:{c}")
        e.set(qn("w:w"), str(w)), e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def danh_so(K):
    """Gan so Hinh/Bang theo chuong: {khoa: 'Hình 2.1'}."""
    so, chuong, dem = {}, 0, {"hinh": 0, "bang": 0}
    for k in K:
        if k[0] == "h1":
            chuong += 1
            dem = {"hinh": 0, "bang": 0}
        elif k[0] in ("hinh", "bang"):
            dem[k[0]] += 1
            so[(k[0], k[1])] = f"{'Hình' if k[0] == 'hinh' else 'Bảng'} {chuong}.{dem[k[0]]}"
    return so


class TrichDan:
    def __init__(self):
        self.thu_tu: list[str] = []

    def thay(self, chu: str, so: dict) -> str:
        def r(m):
            k = m.group(1)
            if k not in TAI_LIEU:
                raise SystemExit(f"Tai lieu chua khai bao: {k}")
            if k not in self.thu_tu:
                self.thu_tu.append(k)
            return f"[{self.thu_tu.index(k) + 1}]"
        chu = re.sub(r"\{r:(\w+)\}", r, chu)
        chu = re.sub(r"\{h:(\w+)\}", lambda m: so[("hinh", m.group(1))], chu)
        chu = re.sub(r"\{b:(\w+)\}", lambda m: so[("bang", m.group(1))], chu)
        if re.search(r"\{[rhb]:", chu):
            raise SystemExit(f"Con the chua thay: {chu[:80]}")
        return chu


def ghi_than_bai(d, K, H):
    so, td = danh_so(K), TrichDan()
    for k in K:
        loai = k[0]
        if loai == "h1":
            par = d.add_paragraph(style="Heading 1")
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.page_break_before = True
            par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(0), Pt(12)
            dat_run(par.add_run(k[1]), co=17)
        elif loai in ("h2", "h3"):
            par = d.add_paragraph(k[1], style="Heading 2" if loai == "h2" else "Heading 3")
            par.paragraph_format.space_before = Pt(6)
            par.paragraph_format.space_after = Pt(3)
        elif loai == "p":
            par = d.add_paragraph()
            doan_than(par)
            viet_chu(par, td.thay(k[1], so))
        elif loai == "ul":
            for muc in k[1]:
                par = d.add_paragraph(style="List Paragraph")
                numPr = OxmlElement("w:numPr")
                ilvl, numId = OxmlElement("w:ilvl"), OxmlElement("w:numId")
                ilvl.set(qn("w:val"), "0"), numId.set(qn("w:val"), "14")
                numPr.append(ilvl), numPr.append(numId)
                par._p.get_or_add_pPr().append(numPr)
                doan_than(par, after=40, thut=False)
                viet_chu(par, td.thay(muc, so))
        elif loai == "ct":
            par = d.add_paragraph()
            doan_than(par, before=20, after=60, thut=False)
            ts = par.paragraph_format.tab_stops
            ts.add_tab_stop(Cm(8.0), WD_TAB_ALIGNMENT.CENTER), ts.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT)
            dat_run(par.add_run("\t" + k[1]), nghieng=True)
            dat_run(par.add_run(f"\t({k[2]})"))
        elif loai == "hinh":
            _, khoa, anh, rong, chu = k
            par = d.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.keep_with_next = True
            doan_than(par, before=60, after=0, thut=False)
            par.add_run().add_picture(str(anh), width=Cm(rong))
            cap = d.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doan_than(cap, after=160, thut=False)
            viet_chu(cap, f"{so[('hinh', khoa)]}. {td.thay(chu, so)}", nghieng=True)
        elif loai == "bang":
            _, khoa, chu, dau, dong, rong = k
            cap = d.add_paragraph()
            cap.paragraph_format.keep_with_next = True
            doan_than(cap, before=60, after=60, thut=False)
            viet_chu(cap, f"{so[('bang', khoa)]}: {td.thay(chu, so)}", dam_ca=True)
            t = d.add_table(rows=1 + len(dong), cols=len(dau))
            ke_bang(t)
            # cot chu (co o dai) can trai, cot so can giua
            trai = [j == 0 or any(len(h[j]) > 14 for h in dong) for j in range(len(dau))]
            dai = len(dong) > 10   # bang dai (co 13) cho ngat trang, lap dong tieu de; khong thi de trong nua trang
            for i, hang in enumerate([dau] + dong):
                trPr = t.rows[i]._tr.get_or_add_trPr()
                trPr.append(OxmlElement("w:cantSplit"))
                if i == 0 and dai:
                    trPr.append(OxmlElement("w:tblHeader"))
                for j, o in enumerate(hang):
                    c = t.cell(i, j)
                    c.width = Cm(rong[j])
                    par = c.paragraphs[0]
                    par.alignment = WD_ALIGN_PARAGRAPH.LEFT if trai[j] else WD_ALIGN_PARAGRAPH.CENTER
                    pf = par.paragraph_format
                    pf.space_before, pf.space_after, pf.line_spacing = Pt(1), Pt(1), 1.15
                    pf.keep_with_next = i < len(dong) and (not dai or i == 0 or i >= len(dong) - 2)   # bang dai: khong de 1-2 dong mo coi
                    viet_chu(par, td.thay(o, so), dam_ca=i == 0, co=13)
            sau = d.add_paragraph()
            doan_than(sau, after=0, thut=False)
            sau.paragraph_format.line_spacing = 0.6
    # danh muc tai lieu tham khao
    par = d.add_paragraph(style="Heading 1")
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.page_break_before = True
    par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(0), Pt(12)
    dat_run(par.add_run("DANH MỤC TÀI LIỆU THAM KHẢO"), co=17)
    for i, k in enumerate(td.thu_tu, 1):
        par = d.add_paragraph()
        pf = par.paragraph_format
        pf.left_indent, pf.first_line_indent = Cm(1.0), Cm(-1.0)
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(3), 1.15
        par.alignment = WD_ALIGN_PARAGRAPH.LEFT
        dat_run(par.add_run(f"[{i}]\t{TAI_LIEU[k]}"), co=12)
    thua = set(TAI_LIEU) - set(td.thu_tu)
    if thua:
        raise SystemExit(f"Tai lieu khai bao ma khong trich trong bai: {sorted(thua)}")
    return td.thu_tu


def dat_chu_doan(par, chu: str):
    """Thay chu cua doan, giu dinh dang run dau."""
    runs = par.runs
    runs[0].text = chu
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def sua_phan_dau(d, tt: dict, D):
    body = d.element.body
    els = list(body.iterchildren())
    P = lambda e: docx.text.paragraph.Paragraph(e, d)
    # bia
    dat_chu_doan(P(els[4]), tt["hoc_phan"])
    dat_chu_doan(P(els[7]), "PHÁT HIỆN BẤT THƯỜNG ĐẦU GỐI TRÊN ẢNH MRI")
    dat_chu_doan(P(els[8]), "VỚI NHÃN YẾU TỪ BÁO CÁO CHẨN ĐOÁN")
    gv = P(els[11]).runs
    gv[1].text = " " + tt["gvhd"]
    for r in gv[2:]:
        r._r.getparent().remove(r._r)
    sv = d.tables[1]
    for i, s in enumerate(tt["sinh_vien"], 1):
        for j, v in enumerate((s["ten"], s["mssv"], s["lop"])):
            dat_chu_doan(sv.cell(i, j).paragraphs[0], v)
    for hang in list(sv.rows)[1 + len(tt["sinh_vien"]):]:
        hang._tr.getparent().remove(hang._tr)
    dat_chu_doan(P(els[19]), tt["noi_thang"])
    P(els[19]).paragraph_format.space_before = Pt(54)   # thap xuong sat vien duoi nhu ban chu nhan chinh tay (27/11)

    # tom tat: 3 doan 21-23
    g6 = D["cuoi"]["nhom"]["gop_6_cnn"]
    du = D["du_lieu"]
    so = lambda n: f"{n:,}".replace(",", ".")
    tom_tat = [
        "Đồ án giải quyết bài toán phát hiện bất thường trên ảnh cộng hưởng từ (MRI) đầu gối, dùng bộ dữ liệu do "
        "Hội Điện quang Bắc Mỹ (RSNA) công bố: từ một ca chụp, dự đoán xác suất có mặt của 12 bất thường như rách dây chằng chéo trước, rách sụn "
        f"chêm, thoái hoá khớp và tràn dịch khớp. Khó khăn chính là chỉ {du['n_gold']} trên {so(du['n_study'])} ca "
        f"({f(100 * du['n_gold'] / du['n_study'], 2)}%) có nhãn do bác sĩ gán; các ca còn lại chỉ kèm một báo cáo "
        "chẩn đoán viết tự do bằng 9 ngôn ngữ.",
        "Nhóm xây dựng toàn bộ quy trình từ file DICOM thô: phân tích dữ liệu ảnh và báo cáo, rút nhãn yếu từ báo cáo bằng mô hình ngôn ngữ lớn (LLM) "
        "để dùng được phần dữ liệu chưa có nhãn, chuẩn hoá ảnh theo từng series, rồi huấn luyện qua hai giai đoạn. "
        "Giai đoạn 1 dùng DINOv2 đóng băng làm bộ trích đặc trưng và chỉ huấn luyện phần đầu phân loại. Giai đoạn 2 "
        "huấn luyện toàn bộ mạng CNN 2.5D (EfficientNet) trên chính ảnh MRI, gom các lát cắt bằng cơ chế chú ý riêng "
        "cho từng bệnh. Ở giai đoạn 2, các ca có nhãn bác sĩ không bao giờ được đưa vào huấn luyện mà chỉ dùng để "
        "chấm, và mọi lựa chọn của mô hình cuối được chốt trước khi xem điểm.",
        f"Mô hình cuối gộp theo hạng 6 CNN, đạt macro AUC {f(g6['auc'])} (khoảng tin cậy 95%: "
        f"{f(g6['ci'][0])}–{f(g6['ci'][1])}) trên {du['n_gold']} ca có nhãn bác sĩ, so với {f(D['seed']['gop'])} của "
        "giai đoạn 1. Nhóm cũng tự gán lại nhãn cho toàn bộ báo cáo bằng Gemini Flash để kiểm chứng bộ nhãn công khai, "
        "và xây dựng một web demo chạy cục bộ (5 mô hình CNN) trả về 12 xác suất kèm lát cắt mà mô hình chú ý. Kết quả là đánh giá "
        "nội bộ trên tập nhỏ, ở mức chứng minh ý tưởng, chưa dùng được cho lâm sàng.",
    ]
    for i, chu in enumerate(tom_tat):
        dat_chu_doan(P(els[21 + i]), chu)

    # bang phan cong
    pc = d.tables[2]
    mau_tieu_de = copy.deepcopy(pc.cell(1, 1).paragraphs[0]._p)
    mau_muc = copy.deepcopy(pc.cell(1, 1).paragraphs[1]._p)
    # chia ~60/40 theo dung mach trinh bay tren slide (chu nhan 27/11): moi nguoi lam phan minh noi
    viec = {
        "chinh": ("Nhãn LLM, giai đoạn 2 và web demo:", [
            "Tiền xử lý DICOM: chọn series, chuẩn hoá 24 lát; chia tập theo ca, chặn rò rỉ",
            "Rút nhãn LLM; tự gán nhãn bằng Gemini Flash",
            "Giai đoạn 2: xây dựng và huấn luyện CNN 2.5D trên GPU đám mây",
            "Đánh giá AUC, bootstrap; chốt và gộp mô hình cuối",
            "Xây dựng máy chủ suy luận cho web demo",
            "Soạn báo cáo và slide phần giai đoạn 2, web demo, kết luận",
        ]),
        "phu": ("Khảo sát, dữ liệu và giai đoạn 1:", [
            "Khảo sát bài toán và tài liệu liên quan; phân tích dữ liệu ảnh và báo cáo",
            "Rút nhãn yếu bằng từ điển luật; so sánh các nguồn nhãn với nhãn bác sĩ",
            "Giai đoạn 1: trích đặc trưng DINOv2, huấn luyện các head attention; phân tích lỗi và đo lại",
            "Thiết kế giao diện và kiểm thử web demo",
            "Soạn báo cáo và slide phần mở đầu, dữ liệu, giai đoạn 1",
        ]),
    }
    for i, s in enumerate(tt["sinh_vien"], 1):
        dat_chu_doan(pc.cell(i, 0).paragraphs[0], "- " + s["ten"])
        o = pc.cell(i, 1)
        for par in o.paragraphs:
            par._p.getparent().remove(par._p)
        tieu_de, muc = viec[s["vai"]]
        e = copy.deepcopy(mau_tieu_de)
        o._tc.append(e)
        dat_chu_doan(o.paragraphs[-1], tieu_de)
        for m in muc:
            e = copy.deepcopy(mau_muc)
            o._tc.append(e)
            dat_chu_doan(o.paragraphs[-1], "- " + m)
        dat_chu_doan(pc.cell(i, 2).paragraphs[0], "Đã hoàn thành")
    for hang in list(pc.rows)[1 + len(tt["sinh_vien"]):]:
        hang._tr.getparent().remove(hang._tr)
    for hang in pc.rows:   # mau gian dong 1,5 + cach doan lam bang tran trang
        for o in hang.cells:
            for par in o.paragraphs:
                pf = par.paragraph_format
                pf.space_before, pf.space_after, pf.line_spacing = Pt(2), Pt(2), 1.2
    # muc luc co 13, gian dong 1,5 nhu than bai (chu nhan 27/11); dai hon mot trang thi sang trang 2
    for ten in ("toc 1", "toc 2", "toc 3"):
        st = d.styles[ten]
        st.font.size = Pt(13)
        pf = st.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(0), 1.5
    # muc luc chi lay cap chuong va muc (Heading 1, 2); bo \u vi \u lay ca outline level 3
    for e in body.iter(qn("w:instrText")):
        if e.text and e.text.strip().startswith("TOC "):
            e.text = ' TOC \\h \\z \\t "Heading 1,1,Heading 2,2" '
    # doan ket thuc truong muc luc: thu nho de khong tran sang trang trang
    cuoi_ml = body.xpath("./w:sdt//w:p")[-1]
    sp = cuoi_ml.find(qn("w:pPr")).find(qn("w:spacing"))   # mau co san <w:spacing>, sua tai cho de giu thu tu
    for k, v in (("before", "0"), ("after", "0"), ("line", "20"), ("lineRule", "exact")):
        sp.set(qn(f"w:{k}"), v)

    # bo doan trong sau bang phan cong, muc luc sang trang moi
    els = list(body.iterchildren())
    i_ml = next(i for i, e in enumerate(els) if e.tag == qn("w:p") and P(e).text.strip() == "MỤC LỤC")
    i_pc = next(i for i, e in enumerate(els) if e is pc._tbl)
    for e in els[i_pc + 1:i_ml]:
        if e.tag == qn("w:p") and not P(e).text.strip() and not e.xpath(".//w:sectPr"):
            body.remove(e)
    for ten in ("TÓM TẮT ĐỒ ÁN", "BẢNG PHÂN CÔNG NHIỆM VỤ", "MỤC LỤC"):
        e = next(e for e in body.iterchildren() if e.tag == qn("w:p") and P(e).text.strip() == ten)
        P(e).paragraph_format.page_break_before = True

    # xoa than bai cu: tu "CHƯƠNG 1" toi truoc sectPr cuoi
    els = list(body.iterchildren())
    i_c1 = next(i for i, e in enumerate(els) if e.tag == qn("w:p") and P(e).text.strip().startswith("CHƯƠNG 1"))
    for e in els[i_c1:]:
        if e.tag != qn("w:sectPr"):
            body.remove(e)
    for e in list(body.iterchildren()):   # bookmark mo coi o cap than bai
        if e.tag in (qn("w:bookmarkStart"), qn("w:bookmarkEnd")):
            body.remove(e)
    # doan trong cuoi truoc chuong 1 (sau muc luc) bo luon
    cuoi = [e for e in body.iterchildren() if e.tag == qn("w:p")][-1]
    if not P(cuoi).text.strip() and not cuoi.xpath(".//w:sectPr"):
        body.remove(cuoi)


def xoa_anh_thua(d):
    """Anh cua than bai PBL5 da bo: xoa quan he khoi part de file khong mang anh cu."""
    dung = {e.get(qn("r:embed")) for e in d.element.body.iter(qn("a:blip"))}
    part = d.part
    for rid, rel in list(part.rels.items()):
        if rel.reltype.endswith("/image") and rid not in dung:
            del part.rels[rid]


def xuat_pdf() -> None:
    win_tam = subprocess.run(["/mnt/c/Windows/System32/cmd.exe", "/c", "echo %TEMP%"], capture_output=True, text=True,
                             cwd="/mnt/c").stdout.strip()
    tam = Path(subprocess.run(["wslpath", "-u", win_tam], capture_output=True, text=True).stdout.strip()) / "pbl6_bao_cao"
    tam.mkdir(exist_ok=True)
    shutil.copy(DOCX, tam / DOCX.name)
    w_docx, w_pdf = win_tam + "\\pbl6_bao_cao\\" + DOCX.name, win_tam + "\\pbl6_bao_cao\\" + DOCX.with_suffix(".pdf").name
    ps = f"""
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
$d = $w.Documents.Open('{w_docx}', $false, $false, $false)
foreach ($t in $d.TablesOfContents) {{ $t.Update() }}
$d.Repaginate()
foreach ($t in $d.TablesOfContents) {{ $t.Update() }}
$d.Save()
$d.ExportAsFixedFormat('{w_pdf}', 17)
Write-Output ("TRANG=" + $d.ComputeStatistics(2))
$d.Close($false)
if ($w.Documents.Count -eq 0) {{ $w.Quit() }}
"""
    kq = subprocess.run(["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                        timeout=300, cwd="/mnt/c")
    if kq.returncode != 0:
        raise SystemExit("Word loi:\n" + kq.stdout + kq.stderr)
    shutil.copy(tam / DOCX.name, DOCX)
    shutil.copy(tam / DOCX.with_suffix(".pdf").name, DOCX.with_suffix(".pdf"))
    shutil.rmtree(tam, ignore_errors=True)
    print(" ", kq.stdout.strip().splitlines()[-1], "->", DOCX.with_suffix(".pdf").relative_to(REPO))


def main() -> None:
    if not THONG_TIN.exists():
        raise SystemExit(f"Thieu {THONG_TIN.relative_to(REPO)} (mau_docx, hoc_phan, gvhd, noi_thang, sinh_vien, "
                         "vi_du_bao_cao)")
    tt = json.loads(THONG_TIN.read_text(encoding="utf-8"))
    D = nap_so_lieu()
    print("ve hinh ...")
    H, T = ve_hinh(D)
    d = docx.Document(tt["mau_docx"])
    sua_phan_dau(d, tt, D)
    tl = ghi_than_bai(d, noi_dung(D, H, T), H)
    xoa_anh_thua(d)
    d.save(DOCX)
    print(f"  {DOCX.relative_to(REPO)}  ({len(tl)} tai lieu tham khao)")
    if "--pdf" in sys.argv:
        xuat_pdf()


if __name__ == "__main__":
    main()
