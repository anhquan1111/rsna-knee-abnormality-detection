"""Xuat bao cao Word PBL6 theo khuon cua truong (lay file bao cao PBL5 lam goc: style, bia, logo).

    uv run --extra bao-cao python scripts/71_xuat_word_pbl6.py          # -> reports/pbl6/Bao_Cao_PBL6.docx
    uv run --extra bao-cao python scripts/71_xuat_word_pbl6.py --pdf    # them PDF qua Word tren Windows

Thong tin sinh vien, GVHD, ten hoc phan va duong dan file mau nam o reports/pbl6/thong_tin.json.
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
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

REPO = Path(__file__).resolve().parents[1]
RA = REPO / "reports" / "pbl6"
HINH = RA / "hinh"
THONG_TIN = RA / "thong_tin.json"
DOCX = RA / "Bao_Cao_PBL6.docx"
CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
WEB = "http://127.0.0.1:8790/?mau=mau7"

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
    "rsna": 'Radiological Society of North America, "RSNA Knee Abnormality Detection," Kaggle '
            "competition. [Online]. Available: https://www.kaggle.com/competitions/"
            "rsna-knee-abnormality-detection. [Accessed: Nov. 2026].",
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
    for ten in ("times.ttf", "timesbd.ttf", "timesi.ttf"):
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
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
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
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
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
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
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
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
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
                    f"--user-data-dir={doi(ho_so)}", "--virtual-time-budget=60000", "--window-size=1600,1000",
                    "--force-device-scale-factor=2", f"--screenshot={doi(tam)}", WEB],
                   capture_output=True, timeout=240)
    shutil.rmtree(ho_so, ignore_errors=True)
    if not tam.exists():
        raise SystemExit("Chrome khong chup duoc web demo")
    tam.replace(ra)


def ve_hinh(D) -> dict[str, Path]:
    HINH.mkdir(parents=True, exist_ok=True)
    plt = chuan_bi_matplotlib()
    h = {k: HINH / f"{k}.png" for k in ("tong_quan", "kien_truc", "hanh_trinh", "epoch", "tung_benh", "web")}
    hinh_tong_quan(plt, D, h["tong_quan"])
    hinh_kien_truc(plt, h["kien_truc"])
    hinh_hanh_trinh(plt, D, h["hanh_trinh"])
    hinh_epoch(plt, D, h["epoch"])
    hinh_tung_benh(plt, D, h["tung_benh"])
    chup_web(h["web"])
    return h


# ============================================================ noi dung
def noi_dung(D, H) -> list:
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

    # ------------------------------------------------ chuong 1
    h1("CHƯƠNG 1. GIỚI THIỆU")
    h2("1.1. Bối cảnh và bài toán")
    p("Chụp cộng hưởng từ (MRI) là phương tiện chính để đánh giá tổn thương mô mềm ở khớp gối như dây chằng, "
      "sụn chêm và sụn khớp. Một ca chụp gồm nhiều chuỗi ảnh (series) ở ba mặt phẳng Sagittal, Coronal và Axial, "
      "mỗi series vài chục lát cắt, nên việc đọc ảnh tốn thời gian của bác sĩ chẩn đoán hình ảnh. Các nghiên cứu "
      "như MRNet {r:mrnet} đã cho thấy mạng học sâu có thể phát hiện bất thường, rách dây chằng chéo trước và rách "
      "sụn chêm trên MRI gối, khi được huấn luyện trên 1.370 ca có nhãn đầy đủ.")
    p(f"Đồ án sử dụng dữ liệu của cuộc thi RSNA Knee Abnormality Detection trên Kaggle {{r:rsna}}. Đầu vào là một "
      f"ca MRI gối, trung bình {f(du['series_moi_study'], 2)} series mỗi ca; toàn bộ tập dữ liệu gồm "
      f"{so(du['n_study'])} ca, {so(du['n_series'])} series, khoảng {nk['dung_luong_gb']} GB file DICOM. Đầu ra là "
      "12 xác suất, mỗi xác suất ứng với một bất thường trong {b:benh} (phần sau gọi ngán là \"bệnh\"). Đây là bài toán đa nhãn: một ca có thể mắc "
      "nhiều bệnh cùng lúc. Thước đo là trung bình AUC của 12 nhãn (macro AUC).")
    rows = [[TEN_VI[k].split(" (")[0], k, str(duong[k])] for k in TEN_VI]
    K.append(("bang", "benh", f"12 bất thường cần dự đoán và số ca dương tính trong {du['n_gold']} ca có nhãn bác sĩ",
              ["Bất thường", "Tên trong dữ liệu", "Số ca dương tính"], rows, [7.0, 4.5, 4.0]))
    h2("1.2. Các vấn đề cần giải quyết")
    p("Bài toán có bốn khó khăn chính, quyết định cách thiết kế toàn bộ hệ thống:")
    ul(f"**Nhãn rất hiếm.** Chỉ {du['n_gold']} trên {so(du['n_study'])} ca ({ti_le_gold}%) có nhãn do bác sĩ gán. "
       f"{so(n_khong)} ca còn lại chỉ kèm một báo cáo chẩn đoán viết tự do bằng 9 ngôn ngữ (tiếng Anh, Tây Ban Nha, "
       "Thổ Nhĩ Kỳ, Croatia, Hy Lạp, Đức, Bulgaria, Hà Lan, Pháp). Muốn dùng được phần dữ liệu này phải rút nhãn từ văn bản.",
       "**Dữ liệu ảnh lớn và không đồng nhất.** Mỗi ca có số series, chuỗi xung, số lát và độ phân giải khác nhau, "
       "đến từ máy của nhiều hãng; một số file nén JPEG Lossless hoặc JPEG 2000.",
       f"**Tập đánh giá rất nhỏ.** {du['n_gold']} ca có nhãn bác sĩ là thước đo duy nhất, nên khoảng tin cậy rộng và "
       "rất dễ vô tình chọn mô hình theo đúng tập dùng để chấm, làm điểm số lạc quan.",
       "**Tài nguyên tính toán giới hạn.** Nhóm chỉ có một laptop RTX 4060 8 GB và GPU miễn phí của Kaggle "
       "(2 × T4 16 GB, tối đa 12 giờ mỗi lần chạy).")
    h2("1.3. Mục tiêu và phạm vi")
    p("Mục tiêu của đồ án là xây dựng trọn vẹn quy trình từ file DICOM thô tới 12 xác suất bệnh, tự viết toàn bộ, "
      "không dùng lại checkpoint của người khác; đạt macro AUC cao nhất có thể trên tập có nhãn bác sĩ với cách đo "
      "trung thực; và có một web demo để minh hoạ mô hình trên ca thật.")
    p(f"Phạm vi: hệ thống là sản phẩm học tập ở mức chứng minh ý tưởng, không phải công cụ chẩn đoán lâm sàng. Kết quả "
      f"được đánh giá trên {du['n_gold']} ca có nhãn bác sĩ của cuộc thi. Web demo chỉ chạy trên máy cục bộ; dữ liệu "
      "của cuộc thi không được phát tán lại.")

    # ------------------------------------------------ chuong 2
    h1("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ GIẢI PHÁP")
    h2("2.1. Tổng quan giải pháp")
    p("Giải pháp gồm hai nhánh gặp nhau ở bước huấn luyện ({h:tongquan}). Nhánh nhãn biến báo cáo chẩn đoán của "
      f"{so(n_khong)} ca thành nhãn mềm cho 12 bệnh. Nhánh ảnh đọc DICOM, chọn series và chuẩn hoá thành khối ảnh cố "
      f"định cho mỗi ca. Mô hình được phát triển qua hai giai đoạn. Ở giai đoạn 1, {du['n_gold']} ca có nhãn bác sĩ được "
      "chia 5 fold để kiểm định chéo; ở giai đoạn 2, chúng được giữ riêng hoàn toàn, chỉ dùng để chấm.")
    K.append(("hinh", "tongquan", H["tong_quan"], 15.5, "Sơ đồ tổng quan hệ thống"))
    h2("2.2. Nhãn yếu từ báo cáo chẩn đoán")
    p("Học với nhãn yếu (weak supervision) là dùng nhãn được sinh tự động, có nhiễu nhưng rẻ, thay cho nhãn do "
      "chuyên gia gán {r:snorkel}. Ở đây nguồn nhãn yếu là báo cáo chẩn đoán đi kèm mỗi ca. Nhóm thử hai cách rút nhãn.")
    h3("2.2.1. Từ điển luật")
    p("Cách đầu tiên là bộ từ điển tự viết: với mỗi bệnh, liệt kê từ khoá ở 9 ngôn ngữ, kèm luật phủ định (ví dụ "
      "\"không thấy rách\") và biến thể hình thái của từng ngôn ngữ. Mỗi ô (ca, bệnh) nhận giá trị 1, 0 hoặc \"chưa rõ\". "
      f"Trên {du['n_gold']} ca có nhãn bác sĩ, từ điển chỉ đưa ra kết luận cho {pt(D['llm']['do_phu_tu_dien_tren_gold'])} "
      f"số ô, và đạt AUC {f(llm['tu_dien']['macro_auc'])} khi so với bác sĩ. Dù vậy, chỉ thêm nhãn từ điển cho "
      f"{so(n_khong)} ca đã nâng điểm mô hình từ {f(D['nguon_nhan']['A'])} lên {f(D['nguon_nhan']['B'])}.")
    h3("2.2.2. Nhãn từ mô hình ngôn ngữ lớn")
    p("Cách thứ hai là để mô hình ngôn ngữ lớn (LLM) đọc báo cáo và cho xác suất từng bệnh. Mô hình chính được huấn "
      "luyện bằng một bộ nhãn LLM công khai trên Kaggle {r:llmlabels}: nhãn mềm trong khoảng 0–1, phủ 100% số ca, "
      f"đạt AUC {f(llm['cong_khai']['macro_auc'])} so với bác sĩ. Vì bộ nhãn này không ghi rõ dùng LLM nào và prompt "
      "nào, nhóm tự gán lại nhãn cho cả 4.407 báo cáo bằng Gemini Flash {r:gemini} với prompt tự viết để kiểm chứng "
      "(mục 3.4 và 4.4).")
    h2("2.3. Giai đoạn 1: DINOv2 đóng băng")
    p("DINOv2 {r:dinov2} là mô hình Vision Transformer học tự giám sát trên khoảng 142 triệu ảnh tự nhiên, cho đặc "
      "trưng dùng được cho nhiều tác vụ mà không cần huấn luyện lại. Ở giai đoạn 1, DINOv2 được giữ đóng băng: mỗi lát "
      "MRI đi qua mạng một lần để thành một vector đặc trưng (384 chiều với ViT-S/14). Phần huấn luyện chỉ là các "
      "\"head\" nhỏ phía sau, nên toàn bộ vòng thí nghiệm chạy được trên laptop 8 GB.")
    p("Mỗi ca là một túi (bag) gồm nhiều lát, và bệnh thường chỉ thấy trên vài lát. Đây là bài toán học đa thể hiện "
      "(multiple instance learning, MIL). Head dùng gom chú ý (attention pooling) {r:ilse}: mạng học một trọng số "
      "cho từng lát, rồi lấy tổng có trọng số của các vector để ra vector đại diện cho cả ca. Nhóm dùng attention "
      "riêng cho từng bệnh, để mỗi bệnh tự chọn lát của mình. Năm head được huấn luyện: ba head chỉ nhìn một mặt "
      "phẳng, hai head nhìn cả ba mặt phẳng với hai cỡ backbone. Kết quả của năm head được gộp theo hạng (mục 2.5.3).")
    h2("2.4. Giai đoạn 2: CNN 2.5D với attention riêng từng bệnh")
    p("Đặc trưng của DINOv2 học từ ảnh thường nên không nhạy với tổn thương nhỏ trên MRI; giai đoạn 1 dừng ở mức "
      f"AUC khoảng {f(D['seed']['gop'], 2)}. Giai đoạn 2 huấn luyện toàn bộ mạng tích chập (CNN) trên chính ảnh MRI "
      "({h:kientruc}). Backbone là EfficientNet-B3 {r:effnet} và EfficientNetV2-B2 {r:effv2}, khởi tạo từ trọng số "
      "ImageNet qua thư viện timm {r:timm}.")
    K.append(("hinh", "kientruc", H["kien_truc"], 15.5, "Kiến trúc CNN 2.5D với attention riêng từng bệnh"))
    p("Ý tưởng 2.5D: ghép ba lát liền kề thành ba kênh màu của một ảnh, nên CNN hai chiều vẫn thấy được một phần "
      "chiều sâu mà chi phí chỉ như xử lý ảnh 2D. Mỗi cửa sổ cho một vector, cộng thêm vector nhúng của mặt phẳng "
      "chứa nó. Với mỗi bệnh l, mạng tính điểm chú ý cho mọi cửa sổ k của ca, chuẩn hoá bằng softmax thành trọng số "
      "a(k, l), rồi gom thành vector g(l) = Σ a(k, l)·f(k). Logit của bệnh l chỉ phụ thuộc vào g(l). Hàm mất mát là "
      "binary cross-entropy trên nhãn mềm. Trọng số attention còn cho biết lát nào được dùng để kết luận bệnh nào; "
      "web demo dùng thông tin này để hiển thị lát cắt.")
    h2("2.5. Đánh giá và gộp mô hình")
    h3("2.5.1. Macro AUC")
    p("AUC của một bệnh là xác suất một ca dương tính ngẫu nhiên được mô hình xếp điểm cao hơn một ca âm tính ngẫu "
      "nhiên {r:hanley}; 0,5 là đoán mò, 1,0 là xếp hạng hoàn hảo. Macro AUC là trung bình AUC của 12 bệnh. AUC không "
      "phụ thuộc ngưỡng và không bị đánh lừa bởi tỷ lệ bệnh lệch, trong khi độ chính xác (accuracy) của một mô hình "
      "luôn đoán \"không bệnh\" đã khá cao chỉ vì đa số ô là âm tính.")
    h3("2.5.2. Khoảng tin cậy bootstrap")
    p(f"Với {du['n_gold']} ca, một con số AUC đơn lẻ dao động nhiều. Mọi kết quả đều kèm khoảng tin cậy 95% bằng "
      "bootstrap {r:efron}: lấy mẫu lại có hoàn lại 58 ca 1.000 lần, tính macro AUC mỗi lần, lấy phân vị 2,5% và 97,5%. "
      "Khi so hai mô hình, dùng bootstrap bắt cặp: cùng một mẫu ca cho cả hai, tính hiệu số. Nếu khoảng tin cậy của "
      "hiệu số không chứa 0 thì chênh lệch được coi là có thật.")
    h3("2.5.3. Gộp theo hạng")
    p("Các mô hình huấn luyện riêng nên thang xác suất không cùng gốc; lấy trung bình xác suất sẽ để mô hình \"tự tin\" "
      "hơn lấn át. Vì AUC chỉ quan tâm thứ hạng, nhóm gộp theo hạng: với từng bệnh, xếp hạng các ca theo xác suất của "
      "từng mô hình rồi lấy trung bình hạng, trọng số bằng nhau.")
    h3("2.5.4. Chốt trước khi chấm")
    p("Bài học quan trọng nhất của giai đoạn 1 (mục 4.2) là con số sẽ lạc quan nếu các lựa chọn được quyết định "
      "dựa trên điểm của chính tập chấm. Vì vậy ở giai đoạn 2, số epoch, cách gộp và danh sách mô hình cuối đều được ghi vào "
      "file và commit lên GitHub trước khi có kết quả; điểm báo cáo nguyên như đo được, kể cả khi thấp hơn mong đợi.")

    # ------------------------------------------------ chuong 3
    h1("CHƯƠNG 3. XÂY DỰNG VÀ TRIỂN KHAI HỆ THỐNG")
    h2("3.1. Môi trường và công cụ")
    p("Hệ thống viết bằng Python 3.12, quản lý thư viện bằng uv. Mô hình dùng PyTorch {r:pytorch} và timm; đọc DICOM "
      "bằng pydicom {r:pydicom}; tính AUC bằng scikit-learn {r:sklearn}; web demo dùng FastAPI {r:fastapi}. Phần giai "
      "đoạn 1 và đánh giá chạy trên laptop RTX 4060 8 GB; phần trích đặc trưng và huấn luyện CNN chạy trên Kaggle "
      "Notebook với 2 GPU T4. Mã nguồn chia thành các script đánh số theo thứ tự chạy, mọi số liệu trong báo cáo được "
      "tính lại từ file kết quả.")
    h2("3.2. Xử lý dữ liệu")
    ul(f"**Lập bảng ca và chia tập.** Bảng ca gồm {so(du['n_study'])} ca và {so(du['n_series'])} series. Tập được chia "
       "theo ca, không theo series: khi thử chia theo series, 100% ca kiểm tra có series nằm bên tập huấn luyện.",
       "**Chặn rò rỉ qua nhãn yếu.** Một ca ngoài tập chấm có báo cáo trùng từng ký tự với một ca có nhãn bác sĩ, nên "
       "nhãn yếu của nó mang luôn đáp án. Ca này bị loại khỏi tập huấn luyện ở cả hai giai đoạn.",
       f"**Chọn series.** Mỗi mặt phẳng chọn một series nhạy dịch (ưu tiên chuỗi xung PD hoặc T2 xoá mỡ), vì dịch và "
       f"phù tuỷ xương hiện rõ trên các chuỗi này; {nk['chon_series']}% số ca có đủ series phù hợp ở cả ba mặt phẳng.",
       "**Chuẩn hoá.** Cường độ được cắt theo phân vị 1–99 tính trên cả series (không theo từng lát, để giữ độ sáng "
       "tương đối giữa các lát), rồi lấy mẫu lại thành 24 lát × 256 × 256 điểm ảnh và lưu bộ nhớ đệm dạng uint8.")
    h2("3.3. Huấn luyện CNN 2.5D trên Kaggle")
    p(f"Mỗi mô hình được huấn luyện trên nhãn LLM của {so(cn['mo_hinh'][0]['n_train'])} ca; {nk['n_val_llm']} ca nhãn "
      "LLM khác được giữ lại để theo dõi, không dùng ca nào có nhãn bác sĩ. Cấu hình chính nằm trong {b:cauhinh}.")
    K.append(("bang", "cauhinh", "Cấu hình huấn luyện CNN 2.5D", ["Tham số", "Giá trị"], [
        ["Backbone", "EfficientNet-B3, EfficientNetV2-B2 (pretrained ImageNet)"],
        ["Đầu vào mỗi ca", "3 mặt phẳng × 24 lát × 256 × 256; cửa sổ 3 lát"],
        ["Cửa sổ mỗi mặt phẳng khi huấn luyện", "8, lấy ngẫu nhiên trải đều theo chiều sâu"],
        ["Batch", "4 ca, cộng dồn gradient 2 bước (hiệu dụng 8 ca)"],
        ["Bộ tối ưu", "AdamW {r:adamw}; lr 2·10⁻⁴ (backbone), 10⁻³ (head); weight decay 10⁻²"],
        ["Lịch learning rate", "Khởi động 0,5 epoch, sau đó cosine"],
        ["Số epoch", "12, cố định từ trước"],
        ["Trọng số dùng để chấm", "EMA (hệ số 0,999) ở epoch cuối"],
        ["Độ chính xác số", "Tự động fp16 (autocast) + GradScaler"],
    ], [6.0, 10.0]))
    p(f"Ở hai lần chạy đầu, quá trình huấn luyện rất chậm (khoảng 10 giây mỗi bước) và bị hệ thống buộc dừng (Killed). "
      f"Thay vì đoán nguyên nhân, nhóm "
      f"viết một notebook chẩn đoán riêng và đo được rằng nn.DataParallel trên 2 GPU T4 mất {nk['dp_giay']} giây mỗi "
      f"bước, so với {nk['mot_gpu_giay']} giây khi chạy một GPU. Bản cuối cho mỗi GPU huấn luyện một mô hình độc lập "
      f"trong tiến trình riêng; hai mô hình xong trong {nk['gio_kaggle']} giờ.")
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
    h2("3.4. Tự gán nhãn bằng Gemini Flash")
    p("Để có bộ nhãn của riêng mình và kiểm tra bộ nhãn công khai, nhóm viết prompt mô tả định nghĩa từng bệnh "
      "(ví dụ: thoái hoá dạng nhầy của dây chằng mà không rách thì cho khoảng 30 điểm), yêu cầu LLM chấm 0–100 cho "
      "12 bệnh của mỗi báo cáo. Báo cáo được gửi theo lô 25 báo cáo, chạy song song 4 lô, có lưu kết quả từng lô để chạy tiếp "
      "khi bị ngắt. LLM chỉ thấy văn bản báo cáo, không thấy mã ca hay nhãn bác sĩ, và prompt được chốt trước khi so "
      "với bác sĩ.")
    h2("3.5. Web demo")
    p(f"Web demo gồm một máy chủ FastAPI chạy tại 127.0.0.1:8790 và một trang HTML ({{h:web}}). Người dùng chọn một ca "
      "mẫu có nhãn bác sĩ (không nằm trong tập huấn luyện) hoặc tải lên file DICOM nén zip. Máy chủ đọc và chuẩn hoá ca "
      f"theo đúng quy trình lúc huấn luyện, chạy 5 mô hình CNN rồi gộp theo hạng, mất khoảng {nk['web_giay']} giây mỗi "
      "ca trên GPU. Trang hiển thị 12 xác suất xếp giảm dần, so với nhãn bác sĩ nếu có, kèm khung xem lát cắt MRI mà "
      "mô hình chú ý. Web chạy 5 thay vì 6 mô hình vì trọng số của mô hình #6 bị ghi đè trên Kaggle khi chạy lại #5; "
      "dự đoán của #6 đã được tải về trước nên vẫn được chấm.")
    K.append(("hinh", "web", H["web"], 15.5, "Giao diện web demo với một ca mẫu"))
    h2("3.6. Kiểm thử")
    p(f"Dự án có {nk['n_test']} kiểm thử tự động (pytest), chạy trong khoảng 20 giây và không cần ảnh thật. Kiểm "
      "thử tập trung vào loại lỗi \"âm thầm\": chương trình vẫn chạy và cho ra một con số hợp lý, chỉ có con số đó sai. "
      "Ví dụ: pooling quên mặt nạ cho lát đệm, softmax ra NaN khi ca thiếu cả ba mặt phẳng, cửa sổ 3 lát không liền "
      "kề, chia tập để lọt ca giữa hai phía, và hàm chọn epoch âm thầm chuyển sang chọn theo loss.")

    # ------------------------------------------------ chuong 4
    h1("CHƯƠNG 4. THỰC NGHIỆM VÀ KẾT QUẢ")
    h2("4.1. Thiết lập đánh giá")
    p(f"Mọi con số trong chương này là macro AUC trên {du['n_gold']} ca có nhãn bác sĩ, kèm khoảng tin cậy bootstrap "
      "95% trong ngoặc vuông. Ở giai đoạn 1, các ca này được chia 5 fold theo ca và điểm là AUC ngoài fold; ở giai "
      "đoạn 2, chúng không bao giờ vào huấn luyện nên điểm là điểm ngoài mẫu trực tiếp.")
    h2("4.2. Hành trình cải tiến")
    p("{h:hanhtrinh} tóm tắt các bước. Mỗi bước chỉ đổi một thứ. Dùng cả ba mặt phẳng, thêm nhãn yếu cho "
      f"{so(n_khong)} ca, rồi thay từ điển bằng nhãn LLM đều tăng điểm rõ rệt. Bộ gộp 5 head từng đạt "
      f"{f(ce['cv_cu_bao_cao'])} theo cách đo cũ, nhưng khi kiểm tra lại cách đo, nhóm tìm ra hai lỗi: (1) lúc đo, mỗi fold chọn epoch "
      "có AUC cao nhất ngay trên các ca đang chấm; (2) lúc huấn luyện mô hình cuối, nhãn mềm khiến hàm AUC trả NaN nên "
      "chương trình âm thầm chuyển sang chọn theo loss và huấn luyện tới khoảng epoch 80, trong khi head đạt đỉnh "
      "quanh epoch 5 rồi giảm dần.")
    p(f"Cách sửa: chốt {ce['epoch_chot']} epoch, chọn bằng 10% ca nhãn LLM giữ riêng, không nhìn ca có nhãn bác sĩ, và "
      f"gộp 3 seed. Điểm đo lại đúng cách của giai đoạn 1 là {f(D['seed']['gop'])} {ci(D['seed']['ci_gop'])}. "
      f"Giai đoạn 2 tăng lên {f(cn['auc'])} với 2 CNN và {f(g6['auc'])} với bản cuối 6 CNN.")
    K.append(("hinh", "hanhtrinh", H["hanh_trinh"], 14.0,
              "Macro AUC qua từng bước (cột xám: đo theo cách cũ, chỉ để so các bước)"))
    h2("4.3. Giai đoạn 2 so với giai đoạn 1")
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
    h2("4.4. So sánh ba nguồn nhãn")
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
      f"nhau ở {pt(hk['khop_nguong_05'])} số ô. Điều đáng chú ý là bản gộp 2 CNN huấn luyện bằng nhãn tự gán lại "
      f"đạt {f(g2t['auc'])}, ngang bản gộp 2 CNN cùng cấu hình dùng nhãn công khai ({f(g2c['auc'])}; hiệu số "
      f"{ci_dau(ss['gop_2_cnn_tu_gan']['ci'])} chứa 0). Ít nhất trong thí nghiệm này, chất lượng nhãn đo trực tiếp "
      "chưa phản ánh hết chất lượng mô hình huấn luyện từ nó.")
    h2("4.5. Mô hình cuối")
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
    rows = [[TEN_VI[k], str(duong[k]), f(dino[k]), f(cnn2[k]), f(D["cuoi"]["tung_benh"][k])] for k in TEN_VI]
    K.append(("bang", "tungbenh", "AUC từng bệnh qua các giai đoạn",
              ["Bệnh", "Số ca dương", "DINOv2 (GĐ 1)", "2 CNN", "6 CNN (cuối)"], rows, [6.0, 2.4, 2.6, 2.2, 2.8]))
    K.append(("hinh", "tungbenh", H["tung_benh"], 15.0, "AUC từng bệnh: DINOv2 đóng băng so với mô hình cuối"))
    h2("4.6. Các lỗi âm thầm đã phát hiện")
    p("Phần lớn thời gian phát triển dành cho các lỗi không báo ngoại lệ mà làm sai con số. {b:loi} liệt kê các lỗi "
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

    # ------------------------------------------------ chuong 5
    h1("CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
    h2("5.1. Kết quả đạt được")
    ul(f"Xây dựng trọn vẹn quy trình từ DICOM thô tới 12 xác suất bệnh; mô hình cuối đạt macro AUC {f(g6['auc'])} "
       f"{ci(g6['ci'])} trên {du['n_gold']} ca có nhãn bác sĩ, so với {f(D['seed']['gop'])} của giai đoạn 1.",
       f"Tận dụng được {so(n_khong)} ca không có nhãn bác sĩ ({f(100 * n_khong / du['n_study'], 1)}% dữ liệu) nhờ nhãn "
       "yếu rút từ báo cáo 9 ngôn ngữ; tự gán lại nhãn bằng Gemini Flash và cho thấy nhãn này dùng được để huấn luyện.",
       "Thiết lập cách đánh giá trung thực: chia theo ca, chặn rò rỉ, khoảng tin cậy bootstrap, chốt mô hình trước khi "
       "chấm; tự phát hiện và sửa một con số lạc quan của chính nhóm.",
       f"Web demo chạy cục bộ, trả kết quả trong khoảng {nk['web_giay']} giây mỗi ca, cùng {nk['n_test']} kiểm thử tự động.")
    h2("5.2. Hạn chế")
    ul(f"{du['n_gold']} ca có nhãn bác sĩ không vào huấn luyện ở giai đoạn 2 nhưng đã được xem nhiều lần trong quá "
       "trình phát triển (và dùng để kiểm định chéo ở giai đoạn 1), nên đây là đánh giá nội bộ, không phải tập kiểm "
       "tra độc lập hoàn toàn.",
       "Khoảng tin cậy còn rộng (khoảng ±0,03); chênh lệch cỡ 0,01 giữa các mô hình CNN đơn lẻ chưa phân định được.",
       "AUC chỉ đo khả năng xếp hạng; xác suất chưa được hiệu chỉnh (calibration) và chưa có đánh giá lâm sàng.",
       "Nhãn huấn luyện là nhãn LLM, chỉ khớp bác sĩ khoảng 81% số ô, nên mô hình bị giới hạn bởi chất lượng nhãn này.",
       f"Viêm màng hoạt dịch ({f(D['cuoi']['tung_benh']['Synovitis'])}) và thoái hoá khoang ngoài "
       f"({f(D['cuoi']['tung_benh']['Lateral OA'])}) vẫn còn yếu.")
    h2("5.3. Hướng phát triển")
    ul("Đánh giá trên một tập có nhãn bác sĩ lớn hơn và độc lập, ví dụ tập kiểm tra ẩn của cuộc thi.",
       "Tăng độ phân giải và số lát cho tổn thương nhỏ; thử backbone 3D hoặc huấn luyện riêng cho từng mặt phẳng.",
       "Hiệu chỉnh xác suất và chọn ngưỡng theo từng bệnh để web demo đưa ra gợi ý dễ hiểu hơn.",
       "Cải thiện nhãn yếu bằng cách gộp nhiều LLM, hoặc để bác sĩ duyệt lại những ô mà hai bộ nhãn bất đồng.")
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
                doan_than(par, after=60, thut=False)
                viet_chu(par, td.thay(muc, so))
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
            for i, hang in enumerate([dau] + dong):
                trPr = t.rows[i]._tr.get_or_add_trPr()
                trPr.append(OxmlElement("w:cantSplit"))
                for j, o in enumerate(hang):
                    c = t.cell(i, j)
                    c.width = Cm(rong[j])
                    par = c.paragraphs[0]
                    par.alignment = WD_ALIGN_PARAGRAPH.LEFT if trai[j] else WD_ALIGN_PARAGRAPH.CENTER
                    pf = par.paragraph_format
                    pf.space_before, pf.space_after, pf.line_spacing = Pt(1), Pt(1), 1.15
                    pf.keep_with_next = i < len(dong)   # giu ca bang tren mot trang
                    viet_chu(par, td.thay(o, so), dam_ca=i == 0, co=12)
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
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(6), 1.15
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

    # tom tat: 3 doan 21-23
    g6 = D["cuoi"]["nhom"]["gop_6_cnn"]
    du = D["du_lieu"]
    so = lambda n: f"{n:,}".replace(",", ".")
    tom_tat = [
        "Đồ án giải quyết bài toán của cuộc thi RSNA Knee Abnormality Detection trên Kaggle: từ một ca chụp cộng "
        "hưởng từ (MRI) đầu gối, dự đoán xác suất có mặt của 12 bất thường như rách dây chằng chéo trước, rách sụn "
        f"chêm, thoái hoá khớp và tràn dịch khớp. Khó khăn chính là chỉ {du['n_gold']} trên {so(du['n_study'])} ca "
        f"({f(100 * du['n_gold'] / du['n_study'], 2)}%) có nhãn do bác sĩ gán; các ca còn lại chỉ kèm một báo cáo "
        "chẩn đoán viết tự do bằng 9 ngôn ngữ.",
        "Nhóm xây dựng toàn bộ quy trình từ file DICOM thô: rút nhãn yếu từ báo cáo bằng mô hình ngôn ngữ lớn (LLM) "
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
    viec = {
        "chinh": ("Dữ liệu, mô hình và đánh giá:", [
            "Tải, đọc và chuẩn hoá dữ liệu DICOM; chia tập theo ca, chặn rò rỉ",
            "Rút nhãn yếu từ báo cáo: từ điển luật, nhãn LLM, tự gán nhãn bằng Gemini Flash",
            "Giai đoạn 1: trích đặc trưng DINOv2, huấn luyện các head attention",
            "Phân tích lỗi và đo lại kết quả giai đoạn 1",
            "Giai đoạn 2: xây dựng và huấn luyện CNN 2.5D trên Kaggle",
            "Đánh giá AUC, bootstrap; chốt và gộp mô hình cuối",
            "Xây dựng máy chủ suy luận cho web demo",
        ]),
        "phu": ("Khảo sát, giao diện và tài liệu:", [
            "Khảo sát bài toán, cuộc thi và tài liệu liên quan",
            "Thiết kế giao diện web demo",
            "Kiểm thử web demo trên các ca mẫu",
            "Soạn báo cáo và slide thuyết trình",
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
    # muc luc gon mot trang
    for ten in ("toc 1", "toc 2", "toc 3"):
        st = d.styles[ten]
        st.font.size = Pt(12)
        pf = st.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(0), 1.1
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
        raise SystemExit(f"Thieu {THONG_TIN.relative_to(REPO)} (mau_docx, hoc_phan, gvhd, noi_thang, sinh_vien)")
    tt = json.loads(THONG_TIN.read_text(encoding="utf-8"))
    D = nap_so_lieu()
    print("ve hinh ...")
    H = ve_hinh(D)
    d = docx.Document(tt["mau_docx"])
    sua_phan_dau(d, tt, D)
    tl = ghi_than_bai(d, noi_dung(D, H), H)
    xoa_anh_thua(d)
    d.save(DOCX)
    print(f"  {DOCX.relative_to(REPO)}  ({len(tl)} tai lieu tham khao)")
    if "--pdf" in sys.argv:
        xuat_pdf()


if __name__ == "__main__":
    main()
