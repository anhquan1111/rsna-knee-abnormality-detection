"""Xuat bo slide thuyet trinh PBL6 (14 trang) tu bo 21 slide cua scripts/70.

    uv run --extra bao-cao python scripts/70_xuat_bao_cao.py --slide   # (neu chua co reports/slide/slide.html)
    uv run --extra bao-cao python scripts/72_xuat_slide_pbl6.py        # -> reports/pbl6/slide/ (PNG, PDF, PPTX)

Lay reports/slide/slide.html (da dien so lieu), giu 12 slide noi dung theo GIU, thay bia bang bia
theo khuon truong (logo lay tu file mau Word, ten sinh vien va GVHD tu reports/pbl6/thong_tin.json),
them slide web demo (anh chup do scripts/71 tao o reports/pbl6/hinh/web.png). Slide bi bo van nam
trong trang nhung doi thanh <div hidden> de doan JS ve bieu do khong gap phan tu thieu.
Ket qua o reports/pbl6/ - khong len git (ten, MSSV, anh MRI cua cuoc thi).
"""
from __future__ import annotations

import base64
import html as h
import importlib.util
import json
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NGUON = REPO / "reports" / "slide" / "slide.html"
RA = REPO / "reports" / "pbl6" / "slide"
THONG_TIN = REPO / "reports" / "pbl6" / "thong_tin.json"
ANH_WEB = REPO / "reports" / "pbl6" / "hinh" / "web.png"

# So thu tu (1-based) trong bo 21 slide; "web" la slide moi. Bia moi luon dung dau.
GIU = [2, 3, 4, 5, 7, 13, 14, 15, 17, 18, "web", 20, 21]
# Muc lon tren dong eyebrow: so cu -> so moi (bo slide ngan hon nen danh lai)
MUC = {"1": "1", "2": "2", "3": "3", "5": "4", "7": "5", "9": "7"}
MUC_WEB = "6"

# Sua chu trong slide cu cho dung voi bao cao PBL6 (tim -> thay, moi cap phai khop dung 1 lan)
SUA = [
    ("khi phát triển (dù chưa bao giờ vào train)", "khi phát triển (giai đoạn 2 không train trên 58 ca này)"),
    # giai doan 1 co train tren 4/5 so ca gold (kiem dinh cheo), khong phai "chi de cham"
    ("<b>58 ca gold giữ riêng</b><span>Chỉ để chấm; chia fold theo ca chụp</span>",
     "<b>58 ca gold</b><span>Chia 5 fold theo ca chụp để kiểm định chéo</span>"),
    ("58 ca gold giữ riêng để chấm.", "58 ca gold chia 5 fold để kiểm định chéo."),
]
GHI_CHU_KET = ("Ba ý mang về. Một: với 1,3% nhãn, nhãn rút từ báo cáo là đòn bẩy lớn nhất. Hai: huấn luyện cả mạng trên "
               "chính ảnh MRI vượt hẳn backbone đóng băng, từ 0.764 lên 0.885 với 2 CNN và 0.896 với bản cuối 6 CNN. "
               "Ba: con số chỉ đáng tin khi cách đo đúng; đo lại khiến điểm giảm từ 0.811 xuống 0.764, và ở giai đoạn 2 "
               "mọi lựa chọn được chốt trước khi chấm. Cảm ơn thầy và mời câu hỏi.")


def logo_tu_mau(mau_docx: str) -> list[str]:
    """Hai logo (Bach Khoa, khoa CNTT) trong file Word mau -> data URI."""
    with zipfile.ZipFile(mau_docx) as z:
        return ["data:image/png;base64," + base64.b64encode(z.read(f"word/media/{t}")).decode()
                for t in ("image1.png", "image2.png")]


def bia(tt: dict, logo: list[str]) -> str:
    sv = "".join(f'<tr><td>{h.escape(s["ten"])}</td><td>{s["mssv"]}</td><td>{s["lop"]}</td></tr>'
                 for s in tt["sinh_vien"])
    ghi_chu = ("Kính chào thầy. Nhóm em trình bày đề tài phát hiện 12 bất thường đầu gối trên ảnh MRI của cuộc thi "
               "RSNA trên Kaggle, khi chỉ 58 trên 4.407 ca có nhãn bác sĩ. Bài gồm bốn phần: bài toán và dữ liệu, cách "
               "rút nhãn từ báo cáo chẩn đoán, hai giai đoạn mô hình, rồi kết quả, web demo và giới hạn.")
    return f'''<section class="slide toi" data-ghi-chu="{h.escape(ghi_chu)}">
  <div class="goc" style="align-items:center">
    <div style="display:flex; gap:22px; align-items:center">
      <span class="logo"><img src="{logo[0]}" alt=""><img src="{logo[1]}" alt=""></span>
      <div style="color:var(--phim-chu)">TRƯỜNG ĐẠI HỌC BÁCH KHOA<br>KHOA CÔNG NGHỆ THÔNG TIN</div>
    </div>
    <div>BÁO CÁO {h.escape(tt["hoc_phan"])}<br><span class="g-du-lieu"></span></div>
  </div>
  <div style="display:grid; gap:46px; align-content:center; position:relative">
    <h1 style="max-width:20ch; font-size:92px">Phát hiện bất thường đầu gối trên ảnh MRI với nhãn yếu từ báo cáo chẩn đoán</h1>
    <div class="nhom">
      <div><span class="nh">Giảng viên hướng dẫn</span><b>{h.escape(tt["gvhd"])}</b></div>
      <table><tr><th>Sinh viên thực hiện</th><th>MSSV</th><th>Lớp</th></tr>{sv}</table>
    </div>
    <div id="bia-so" hidden></div>
  </div>
  <div class="goc"><div>{h.escape(tt["noi_thang"].upper())}</div><div>MACRO AUC 0.896 · 58 CA BÁC SĨ GÁN NHÃN</div></div>
</section>'''


def slide_web(anh: Path) -> str:
    src = "data:image/png;base64," + base64.b64encode(anh.read_bytes()).decode()
    ghi_chu = ("Web demo chạy trên máy, chỉ mở ở 127.0.0.1. Chọn một ca mẫu có nhãn bác sĩ, không nằm trong tập huấn "
               "luyện, hoặc tải lên file DICOM nén zip. Máy chủ FastAPI đọc ca theo đúng quy trình lúc huấn luyện, chạy "
               "5 mô hình CNN rồi gộp theo hạng, mất khoảng 0,2 giây mỗi ca trên GPU. Bảng bên trái là 12 xác suất xếp "
               "giảm dần, so với nhãn bác sĩ; khung bên phải là lát cắt MRI mà mô hình chú ý nhiều nhất. Web chạy 5 chứ "
               "không phải 6 mô hình vì trọng số mô hình số 6 bị ghi đè trên Kaggle; dự đoán của nó vẫn được chấm.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">{MUC_WEB} · Web demo</span>
  <h2>Chọn một ca, xem 12 xác suất và lát mô hình chú ý</h2>
  <div style="display:grid; grid-template-columns: minmax(0, 1.75fr) minmax(0, 1fr); gap:44px; min-height:0">
    <img src="{src}" alt="Giao diện web demo" style="width:100%; height:100%; object-fit:cover; object-position:top left; border:1.5px solid var(--ke); border-radius:10px; min-height:0">
    <div class="the"><span class="tt">Cách chạy</span>
      <ul class="ds" style="font-size:26px; gap:18px">
        <li><b>FastAPI ở 127.0.0.1:8790</b>, chỉ chạy trên máy; không gửi ảnh đi đâu.</li>
        <li><b>Ca mẫu có nhãn bác sĩ</b> (ngoài tập huấn luyện) hoặc <b>tải lên DICOM</b> nén zip.</li>
        <li><b>5 CNN gộp theo hạng</b>, khoảng 0,2 giây mỗi ca trên GPU.</li>
        <li><b>Lát được chú ý</b> lấy từ trọng số attention riêng từng bệnh.</li>
      </ul></div>
  </div>
  <div class="chan"><span>RSNA KNEE · PBL6</span><span class="trang"></span></div>
</section>'''


CSS_THEM = """
.logo { display:flex; gap:10px; background:#fff; padding:8px 10px; border-radius:8px; }
.logo img { height:96px; display:block; }
.nhom { display:grid; grid-template-columns: auto auto; gap:70px; align-items:start; justify-content:start;
  font: 400 34px/1.5 var(--f-chu); color: var(--phim-chu); }
.nhom .nh { display:block; font: 500 19px/1.4 var(--f-so); letter-spacing:0.12em; text-transform:uppercase; color: var(--phim-mo); margin-bottom:8px; }
.nhom b { font-weight:600; color:#f3f7fa; }
.nhom table { border-collapse: collapse; }
.nhom th { font: 500 19px/1.4 var(--f-so); letter-spacing:0.12em; text-transform:uppercase; color: var(--phim-mo); text-align:left; padding:0 40px 8px 0; }
.nhom td { padding:4px 40px 4px 0; color:#f3f7fa; }
.nhom td + td { font-family: var(--f-so); font-size:26px; color: var(--phim-chu); }
"""


def main() -> None:
    if not NGUON.exists():
        raise SystemExit("Chua co reports/slide/slide.html: chay scripts/70_xuat_bao_cao.py --slide truoc")
    if not ANH_WEB.exists():
        raise SystemExit("Chua co anh web demo: chay scripts/71_xuat_word_pbl6.py truoc (web demo dang chay)")
    tt = json.loads(THONG_TIN.read_text(encoding="utf-8"))
    goc = NGUON.read_text(encoding="utf-8")

    khoi = list(re.finditer(r'<section class="slide[^"]*".*?</section>', goc, re.S))
    if len(khoi) != 21:
        raise SystemExit(f"Mong doi 21 slide trong {NGUON.name}, thay {len(khoi)}")
    dau, cuoi = goc[:khoi[0].start()], goc[khoi[-1].end():]
    cu = [m.group(0) for m in khoi]

    def sua_slide(s: str) -> str:
        s = re.sub(r'(<span class="eyebrow">)(\d+)( · )', lambda m: m.group(1) + MUC.get(m.group(2), m.group(2)) + m.group(3), s)
        # bao cao PBL6 thong nhat "gan nhan" (khong "gan" dau nga)
        return s.replace("RSNA KNEE · PBL<", "RSNA KNEE · PBL6<").replace("gắn", "gán").replace("Gắn", "Gán")

    hien = [bia(tt, logo_tu_mau(tt["mau_docx"]))]
    for g in GIU:
        hien.append(slide_web(ANH_WEB) if g == "web" else sua_slide(cu[g - 1]))
    # slide ket luan: ghi chu cap nhat 0.896
    hien[-1] = re.sub(r'data-ghi-chu="[^"]*"', f'data-ghi-chu="{h.escape(GHI_CHU_KET)}"', hien[-1], count=1)
    an = [re.sub(r'^<section class="slide[^"]*"', '<div hidden data-bo="1"', s).replace("</section>", "</div>")
          for i, s in enumerate(cu, 1) if i not in GIU and i != 1]
    an.append(re.sub(r'^<section class="slide[^"]*"', '<div hidden data-bo="1"', cu[0]).replace("</section>", "</div>")
              .replace('id="bia-so"', 'id="bia-so-cu"'))

    than = "\n\n".join(hien)
    for tim, thay in SUA:
        if than.count(tim) != 1:
            raise SystemExit(f"Chu can sua khong khop dung 1 lan: {tim}")
        than = than.replace(tim, thay)
    trang_html = (dau.replace("</style>", CSS_THEM + "</style>", 1)
                  .replace("<title>Slide MRI gối 58 nhãn</title>", "<title>Slide PBL6 MRI gối</title>")
                  + than + "\n\n" + "\n".join(an) + cuoi)
    RA.mkdir(parents=True, exist_ok=True)
    trang = RA / "slide.html"
    trang.write_text(trang_html, encoding="utf-8")
    print(f"{len(hien)} slide -> {trang.relative_to(REPO)}")

    spec = importlib.util.spec_from_file_location("xuat70", REPO / "scripts" / "70_xuat_bao_cao.py")
    m70 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m70)
    m70.chup_slide(trang, RA, "Slide_PBL6")


if __name__ == "__main__":
    main()
