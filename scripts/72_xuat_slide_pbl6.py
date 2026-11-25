"""Xuat bo slide thuyet trinh PBL6 (25 trang) tu bo 21 slide cua scripts/70.

    uv run --extra bao-cao python scripts/70_xuat_bao_cao.py --slide   # (neu chua co reports/slide/slide.html)
    uv run --extra bao-cao python scripts/72_xuat_slide_pbl6.py        # -> reports/pbl6/slide/ (PNG, PDF, PPTX)

Lay reports/slide/slide.html (da dien so lieu), giu 12 slide noi dung theo GIU, thay bia bang bia
theo khuon truong (logo lay tu file mau Word, ten sinh vien va GVHD tu reports/pbl6/thong_tin.json),
them slide moi: muc luc, nghien cuu lien quan, du lieu, phan tich anh, quy trinh, tien xu ly, cong thuc
attention, giao thuc giai doan 2 (tach tu slide CNN 2.5D), ROC, phan tich ca, web demo, gioi han va huong tiep
(tach doi slide gioi han cu). Anh va so lieu cua slide moi do scripts/71 tao o reports/pbl6/hinh/ (web.png,
slide/*.png, slide/thong_ke.json) va data/web_demo/phan_tich_ca/. Ghi chu moi slide mo dau bang ten nguoi
trinh bay (NGUOI_NOI, ten lay tu thong_tin.json). Slide bi bo van nam
trong trang nhung doi thanh <div hidden> de doan JS ve bieu do khong gap phan tu thieu.
Ket qua o reports/pbl6/ - khong len git (ten, MSSV, anh MRI cua cuoc thi).
"""
from __future__ import annotations

import base64
import html as h
import importlib.util
import json
import re
import sys
import textwrap
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NGUON = REPO / "reports" / "slide" / "slide.html"
RA = REPO / "reports" / "pbl6" / "slide"
THONG_TIN = REPO / "reports" / "pbl6" / "thong_tin.json"
ANH_WEB = REPO / "reports" / "pbl6" / "hinh" / "web.png"
ANH_SLIDE = REPO / "reports" / "pbl6" / "hinh" / "slide"

# So thu tu (1-based) trong bo 21 slide; chu la slide moi. Bia moi luon dung dau.
GIU = ["muc_luc", 2, "lien_quan", "du_lieu", "eda", 3, "quy_trinh", "tien_xu_ly", 4, 5, 6, 13, "attention", "giao_thuc",
       14, 15, 17, 18, "roc", "phan_tich_ca", "web", "gioi_han", "huong_tiep", 21]
# Nguoi trinh bay theo so slide cuoi (1-based, tinh ca bia): (so thu tu trong thong_tin.json sinh_vien, cac slide,
# cau chuyen loi cho nguoi ke tiep). Sinh vien 1 giu phan kho (giai doan 1, nhan LLM, do lai, giai doan 2, attention,
# phan tich ca, gioi han): 14 slide; sinh vien 2 mo dau, du lieu, roi chot huong tiep va ket luan: 11 slide.
NGUOI_NOI = ((1, range(1, 10), "giai đoạn 1 và phần mô hình"), (0, range(10, 24), "hướng tiếp và kết luận"),
             (1, range(24, 26), ""))
# Muc lon tren dong eyebrow: so cu -> so moi (them muc 2 Du lieu, 3 Quy trinh nen danh lai)
MUC = {"1": "1", "2": "4", "3": "5", "4": "6", "7": "7"}   # slide cu muc 5, 6, 8, 9 khong giu
MUC_WEB = "8"
CHON_SERIES = "96,9"   # % ca du series nhay dich o ca 3 mat phang (SO_NHAT_KY cua scripts/71)
MUC_LUC = [("Bài toán", "12 bất thường, nghiên cứu liên quan"), ("Dữ liệu và phân tích", "4.407 ca, ảnh và báo cáo"),
           ("Quy trình, tiền xử lý", "8 bước, DICOM → khối ảnh"), ("Giai đoạn 1", "DINOv2 đóng băng"),
           ("Nhãn từ báo cáo", "từ điển, LLM"), ("Hành trình và đo lại", "0,811 → 0,764"),
           ("Giai đoạn 2", "CNN 2.5D, attention, model cuối 0,896"), ("Web demo", "chạy cục bộ"),
           ("Giới hạn, kết luận", "điều chưa làm được")]

# Sua chu trong slide cu cho dung voi bao cao PBL6 (tim -> thay, moi cap phai khop dung 1 lan)
SUA = [
    # tieu de, cau ket: noi thang so lieu thay vi an du; "don bay lon nhat" khong dung (nhan +0.12, CNN +0.12)
    ("<h2>Mười hai câu hỏi cho mỗi ca chụp, gần như không có đáp án mẫu</h2>",
     "<h2>12 xác suất cho mỗi ca chụp, chỉ 1,3% số ca có nhãn bác sĩ</h2>"),
    ("<h2>Từng bước cải tiến, một lần đo lại, rồi một bước nhảy</h2>",
     "<h2>Đo lại: 0,811 còn 0,764; giai đoạn 2 lên 0,885</h2>"),
    ("<h2>Vượt hẳn backbone đóng băng</h2>", "<h2>CNN 2.5D đạt 0,885, hơn DINOv2 đóng băng 0,120</h2>"),
    ("<h2>Nhãn hiếm, và báo cáo viết bằng chín thứ tiếng</h2>",
     "<h2>Nhãn hiếm, và báo cáo viết bằng 9 thứ tiếng</h2>"),
    ("<h2>Backbone đóng băng, chỉ học phần đầu, gộp năm góc nhìn</h2>",
     "<h2>Backbone đóng băng, chỉ học phần đầu, gộp 5 góc nhìn</h2>"),
    ("<th class=\"so\">khớp bác sĩ @0.5</th>", "<th class=\"so\">khớp bác sĩ @0,5</th>"),
    ("<p>Nhãn rút từ báo cáo là đòn bẩy lớn nhất khi chỉ 1,3% dữ liệu có nhãn.",
     "<p>Nhãn rút từ báo cáo cho phép dùng 98,7% số ca chưa có nhãn bác sĩ."),
    ("<p>Train cả mạng trên chính ảnh MRI vượt hẳn backbone đóng băng.",
     "<p>Huấn luyện cả mạng trên chính ảnh MRI vượt hẳn backbone đóng băng."),
    # ghi chu: bo cau dan dat, cau noi qua
    ("Khó nhất: chỉ 1,32%", "Khó nhất là chỉ 1,32%"),
    ("rồi gộp theo thứ hạng.", "rồi gộp theo hạng."),
    ("Khoảng tin cậy bắt cặp không cắt qua 0.", "Khoảng tin cậy bắt cặp không chứa 0."),
    ("Em tự viết lại hướng đó, không dùng checkpoint của ai.",
     "Nhóm tự viết lại hướng đó, chỉ dùng trọng số ImageNet công khai, không dùng mô hình người khác đã huấn luyện cho bài này."),
    ("nên báo cáo nguyên như vậy.", "nhóm vẫn báo cáo đúng số đo."),
    ("Đây là tổn thương nhỏ, nằm trên vài lát: đặc trưng học từ ảnh thường không nhìn ra, CNN train trên chính MRI thì học được.",
     ("Đây là tổn thương nhỏ, nằm trên vài lát. Giả thuyết của nhóm là đặc trưng học từ ảnh thường không nhìn ra các "
     "tổn thương này, còn CNN huấn luyện trên chính MRI thì học được.")),
    ("Thầy hỏi nhãn LLM có đúng thật không. Em cho", "Để kiểm tra nhãn LLM có đúng không, nhóm cho"),
    ("Nghĩa là chất lượng nhãn đo trên giấy không quyết định hết chất lượng model.",
     "Ít nhất trong thí nghiệm này, chất lượng nhãn đo trực tiếp chưa phản ánh hết chất lượng mô hình."),
    # slide 3 cu (nhan, ngon ngu) nay thuoc muc Du lieu
    ('<span class="eyebrow">1 · Bài toán</span>\n  <h2>Nhãn hiếm', '<span class="eyebrow">2 · Dữ liệu</span>\n  <h2>Nhãn hiếm'),
    # so do lai (scripts/71 thong_ke_du_lieu): trung vi 30 lat/series; doan "khong benh" dung 65,5% so o
    ("mỗi series khoảng 26 lát.", "mỗi series khoảng 30 lát (trung vị)."),
    ('đoán "không bệnh" cho mọi ca đã đạt 67,2%.', 'đoán "không bệnh" cho mọi ca đã đúng 65,5% số ô.'),
    # header 13.221 series da chon deu khong nen; khong co bang chung JPEG
    ("Một series mỗi mặt phẳng; đọc cả JPEG Lossless và JPEG 2000", "Một series nhạy dịch mỗi mặt phẳng; sắp lát theo InstanceNumber"),
    ("Dùng được 98,7% dữ liệu không nhãn nhờ đọc báo cáo", "Tận dụng 98,7% số ca chưa có nhãn nhờ đọc báo cáo"),
    # chu nhan: khong gioi thieu la cuoc thi Kaggle
    ("các lời giải điểm cao của cuộc thi đều train cả backbone trên MRI", "MRNet cũng huấn luyện cả mạng trên MRI"),
    ("<b>Kaggle T4 × 2, mỗi GPU một model</b>", "<b>2 GPU T4 đám mây, mỗi GPU một model</b>"),
    # giai doan 1 co train tren 4/5 so ca gold (kiem dinh cheo), khong phai "chi de cham"
    ("<b>58 ca gold giữ riêng</b><span>Chỉ để chấm; chia fold theo ca chụp</span>",
     "<b>58 ca gold</b><span>Chia 5 fold theo ca chụp để kiểm định chéo</span>"),
    ("58 ca gold giữ riêng để chấm.", "58 ca gold chia 5 fold để kiểm định chéo."),
    # slide hanh trinh thay cho slide "Diem CV do mot model": gop phan giai thich lan do lai vao ghi chu
    ("Các slide tiếp theo giải thích lần đo lại, rồi tới giai đoạn 2.",
     "Lần đo lại tìm ra hai lỗi. Một: lúc chấm CV, mỗi fold chọn epoch có AUC cao nhất ngay trên các ca đang chấm, "
     "tức chọn trên tập đánh giá. Hai: lúc train model cuối, nhãn mềm làm AUC ra NaN nên hàm âm thầm chuyển sang chọn "
     "theo loss và train tới khoảng epoch 80, trong khi head đạt đỉnh quanh epoch 5. Cách sửa: chốt 9 epoch bằng 10% "
     "ca nhãn LLM giữ riêng, không nhìn ca bác sĩ, và gộp 3 seed."),
]
GHI_CHU_KET = ("Ba ý mang về. Một: khi chỉ 1,3% số ca có nhãn bác sĩ, rút nhãn từ báo cáo cho phép dùng gần hết phần "
               "còn lại, nâng điểm từ 0,627 lên 0,747. Hai: huấn luyện cả mạng trên "
               "chính ảnh MRI vượt hẳn backbone đóng băng, từ 0,764 lên 0,885 với 2 CNN và 0,896 với bản cuối 6 CNN. "
               "Ba: con số chỉ đáng tin khi cách đo đúng; đo lại khiến điểm giảm từ 0,811 xuống 0,764, và ở giai đoạn 2 "
               "mọi lựa chọn được chốt trước khi chấm. Cảm ơn thầy và mời câu hỏi.")


def logo_tu_mau(mau_docx: str) -> list[str]:
    """Hai logo (Bach Khoa, khoa CNTT) trong file Word mau -> data URI."""
    with zipfile.ZipFile(mau_docx) as z:
        return ["data:image/png;base64," + base64.b64encode(z.read(f"word/media/{t}")).decode()
                for t in ("image1.png", "image2.png")]


def bia(tt: dict, logo: list[str]) -> str:
    sv = "".join(f'<tr><td>{h.escape(s["ten"])}</td><td>{s["mssv"]}</td><td>{s["lop"]}</td></tr>'
                 for s in tt["sinh_vien"])
    ghi_chu = ("Kính chào thầy. Nhóm em trình bày đề tài phát hiện 12 bất thường đầu gối trên ảnh MRI, dùng bộ dữ liệu "
               "do Hội Điện quang Bắc Mỹ RSNA công bố, trong đó chỉ 58 trên 4.407 ca có nhãn bác sĩ. Bài đi từ bài "
               "toán và dữ liệu, quy trình xử lý, cách rút nhãn từ báo cáo, hai giai đoạn mô hình, tới kết quả, web "
               "demo và giới hạn. "
               f"Em là {tt['sinh_vien'][1]['ten']}, trình bày bài toán và dữ liệu; bạn {tt['sinh_vien'][0]['ten']} "
               "trình bày hai giai đoạn mô hình, web demo và giới hạn; cuối cùng em trình bày hướng tiếp và kết luận.")
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
  <div class="goc"><div>{h.escape(tt["noi_thang"].upper())}</div><div>MACRO AUC 0,896 · 58 CA BÁC SĨ GÁN NHÃN</div></div>
</section>'''


# Vung cat tu anh chup web (3200 x 2160, chup o 1600 x 1080 x2): bo cot ca mau va dau trang, giu banner ca,
# bang 12 benh va khung MRI de chu trong anh to gap ~1,7 lan khi chieu
CAT_WEB = (872, 600, 3020, 2000)


def slide_web(anh: Path) -> str:
    import io

    from PIL import Image
    im = Image.open(anh)
    if im.size != (3200, 2160):
        raise SystemExit(f"Anh web {im.size} khac 3200x2160: sua CAT_WEB cho khop")
    b = io.BytesIO()
    im.crop(CAT_WEB).save(b, "PNG")
    src = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    ghi_chu = ("Web demo chạy trên máy, chỉ mở ở 127.0.0.1. Chọn một ca mẫu có nhãn bác sĩ, không nằm trong tập huấn "
               "luyện, hoặc tải lên file DICOM nén zip. Máy chủ FastAPI đọc ca theo đúng quy trình lúc huấn luyện, chạy "
               "5 mô hình CNN rồi lấy trung bình xác suất (một ca đơn lẻ thì không xếp hạng được), mất khoảng 0,2 giây "
               "mỗi ca trên GPU. Bảng bên trái là 12 xác suất xếp "
               "giảm dần, so với nhãn bác sĩ; khung bên phải là lát cắt MRI mà mô hình chú ý nhiều nhất. Web chạy 5 chứ "
               "không phải 6 mô hình vì trọng số mô hình số 6 bị ghi đè trên máy chủ đám mây; dự đoán của nó vẫn được chấm.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">{MUC_WEB} · Web demo</span>
  <h2>Chọn một ca, xem 12 xác suất và lát mô hình chú ý</h2>
  <div style="display:grid; grid-template-columns: minmax(0, 2.3fr) minmax(0, 1fr); gap:40px; min-height:0">
    <img src="{src}" alt="Giao diện web demo" style="width:100%; height:100%; object-fit:cover; object-position:top left; border:1.5px solid var(--ke); border-radius:10px; min-height:0">
    <div class="the"><span class="tt">Cách chạy</span>
      <ul class="ds" style="font-size:26px; gap:18px">
        <li><b>FastAPI ở 127.0.0.1:8790</b>, chỉ chạy trên máy; không gửi ảnh đi đâu.</li>
        <li><b>Ca mẫu có nhãn bác sĩ</b> (ngoài tập huấn luyện) hoặc <b>tải lên DICOM</b> nén zip.</li>
        <li><b>5 CNN, lấy trung bình xác suất</b>: một ca đơn lẻ không xếp hạng được; khoảng 0,2 giây mỗi ca trên GPU.</li>
        <li><b>Lát được chú ý</b> lấy từ trọng số attention riêng từng bệnh.</li>
      </ul></div>
  </div>
  <div class="chan"><span>RSNA KNEE · PBL6</span><span class="trang"></span></div>
</section>'''


def uri(ten: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ANH_SLIDE / ten).read_bytes()).decode()


def so_vn(x, n: int = 0) -> str:
    """1.736 / 44,3 kieu Viet."""
    t = f"{x:,.{n}f}"
    return t.replace(",", "_").replace(".", ",").replace("_", ".")


def chan() -> str:
    return '<div class="chan"><span>RSNA KNEE · PBL6</span><span class="trang"></span></div>'


def slide_muc_luc() -> str:
    o = "".join(f'<div><span class="so">{i:02d}</span><b>{h.escape(t)}</b><span>{h.escape(m)}</span></div>'
                for i, (t, m) in enumerate(MUC_LUC, 1))
    ghi_chu = ("Bài trình bày gồm chín phần. Hai phần đầu nói bài toán và dữ liệu: dữ liệu trông thế nào và nhóm rút ra "
               "gì từ phân tích. Phần ba là quy trình từ dữ liệu thô tới mô hình. Ba phần tiếp theo là giai đoạn 1, "
               "cách rút nhãn từ báo cáo và lần đo lại. Sau đó là giai đoạn 2 với model cuối, web demo, và cuối cùng là "
               "giới hạn và kết luận.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">Nội dung</span>
  <h2>Nội dung trình bày</h2>
  <div class="ml">{o}</div>
  {chan()}
</section>'''


def slide_du_lieu(D: dict, T: dict) -> str:
    du = D["du_lieu"]
    anh = "".join(f'<figure><img src="{uri(f"mri_{pl}.png")}" alt=""><figcaption><b>{pl} · {vi}</b>'
                  f'<span>{T["n_lat_mau"][pl]} lát gốc</span></figcaption></figure>'
                  for pl, vi in (("Sagittal", "đứng dọc"), ("Coronal", "đứng ngang"), ("Axial", "nằm ngang")))
    ghi_chu = (f"Bộ dữ liệu có {so_vn(du['n_study'])} ca chụp, {so_vn(du['n_series'])} series, khoảng 569 GB DICOM, đã "
               "ẩn danh. Có ba phần: train.csv chứa báo cáo và 12 cột nhãn, nhưng chỉ 58 ca có nhãn bác sĩ; "
               "train_series.csv cho biết mặt phẳng và series nào nhạy dịch; và thư mục ảnh theo ca, series, lát. Bên "
               "phải là ba mặt phẳng của một ca mẫu. Về thu thập: tải qua API thì sau khoảng 1.500 file tài khoản bị chặn "
               "50 giờ, nên nhóm chuyển bước đọc ảnh lên notebook GPU đám mây có gắn sẵn dữ liệu và chỉ tải kết quả về.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">2 · Dữ liệu</span>
  <h2>{so_vn(du['n_study'])} ca chụp, 569 GB ảnh, chỉ {du['n_gold']} ca có nhãn bác sĩ</h2>
  <div class="hai" style="grid-template-columns: 5fr 8fr; min-height:0">
    <div style="display:grid; gap:22px; align-content:start">
      <div class="ong" style="gap:12px">
        <div class="b"><b>train.csv · {so_vn(du['n_study'])} dòng</b><span>Báo cáo chẩn đoán + 12 cột nhãn; chỉ {du['n_gold']} ca có giá trị</span></div>
        <div class="b"><b>train_series.csv · {so_vn(du['n_series'])} dòng</b><span>Mặt phẳng, cờ nhạy dịch, cờ xoá mỡ của từng series</span></div>
        <div class="b"><b>Ảnh DICOM</b><span>&lt;ca&gt;/&lt;series&gt;/&lt;lát&gt;.dcm · đã ẩn danh ([DATE], UID sinh lại)</span></div>
      </div>
      <div class="luu-y"><b>Thu thập: tải về máy bị chặn</b><p>Sau ~1.500 file qua API: HTTP 429, chờ 50 giờ. Nhóm đọc ảnh trên GPU đám mây có gắn sẵn dữ liệu, chỉ tải kết quả về (vài trăm MB).</p></div>
    </div>
    <div class="ba-anh">{anh}</div>
  </div>
  {chan()}
</section>'''


def thanh(dong: list[tuple[str, float, str]], mau: str, toi_da: float) -> str:
    return "".join(f'<div class="dong"><span class="ten">{h.escape(t)}</span><span class="ray"><span class="dac" '
                   f'style="width:{100 * v / toi_da:.1f}%; background:{mau}"></span></span><span class="gt">{g}</span></div>'
                   for t, v, g in dong)


def slide_eda(D: dict, T: dict) -> str:
    hg = T["hang"]
    tong = sum(hg.values())
    hang = [(k.title() if k != "GE" else "GE", v, so_vn(100 * v / tong, 1) + "%") for k, v in hg.items()]
    mp = T["mat_phang"]
    mat = "".join(
        f'<div class="dong"><span class="ten">{pl}</span><span class="ray">'
        f'<span class="dac" style="width:{100 * v["tong"] / 10000:.1f}%; background:var(--chinh-nhat)"></span>'
        f'<span class="dac" style="width:{100 * v["nhay"] / 10000:.1f}%; background:var(--chinh)"></span></span>'
        f'<span class="gt">{so_vn(v["tong"])}</span></div>' for pl, v in mp.items())
    tt = T["tu_truong"]
    tu_tong = sum(tt.values())
    tu = [("1,5 T", tt["1.5"], so_vn(100 * tt["1.5"] / tu_tong, 1) + "%"),
          ("3,0 T", tt["3.0"], so_vn(100 * tt["3.0"] / tu_tong, 1) + "%"),
          ("1,0–1,2 T", tu_tong - tt["1.5"] - tt["3.0"], so_vn(100 * (tu_tong - tt["1.5"] - tt["3.0"]) / tu_tong, 1) + "%")]
    ghi_chu = (f"Dữ liệu ảnh không đồng nhất ở mọi chiều đo. Mỗi ca có từ 3 đến 14 series, trung bình "
               f"{so_vn(D['du_lieu']['series_moi_study'], 2)}. Mỗi series khoảng 30 lát nhưng có series hơn 300 lát. Ảnh đến từ 5 "
               "hãng máy, chủ yếu 1,5 và 3 Tesla, cạnh ảnh từ 192 đến 1.280 điểm ảnh. Vì vậy phải chọn một series mỗi mặt "
               "phẳng, đưa về cùng số lát, cùng kích thước và chuẩn hoá cường độ. Hãng máy còn là rủi ro: đặc trưng giai "
               "đoạn 1 đoán đúng hãng máy 82,8% số ca, nên model có thể học lối tắt theo nơi chụp.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">2 · Dữ liệu</span>
  <h2>Ảnh không đồng nhất: 5 hãng máy, số series và số lát khác nhau</h2>
  <div style="display:grid; gap:30px; min-height:0; align-content:start">
    <div class="dai-so" style="grid-template-columns:repeat(4,minmax(0,1fr))">
      <div><span class="gt">{so_vn(D['du_lieu']['series_moi_study'], 2)}</span><span class="mt">series mỗi ca (3–14)</span></div>
      <div><span class="gt">{so_vn(T['lat_trung_vi'])}</span><span class="mt">lát mỗi series (trung vị)</span></div>
      <div><span class="gt">{T['rows_min']}–{so_vn(T['rows_max'])}</span><span class="mt">điểm ảnh mỗi cạnh</span></div>
      <div><span class="gt">{so_vn(100 * T['nhay_dich'], 1)}%</span><span class="mt">series nhạy dịch</span></div>
    </div>
    <div class="hai" style="grid-template-columns: repeat(3, minmax(0,1fr)); gap:30px">
      <div class="the"><span class="tt">Hãng máy (số ca)</span><div class="thanh hep">{thanh(hang, "var(--phu)", max(hg.values()))}</div></div>
      <div class="the"><span class="tt">Series theo mặt phẳng</span><div class="thanh hep">{mat}</div>
        <div class="chu-giai" style="font-size:19px"><span><i style="background:var(--chinh)"></i>nhạy dịch</span><span><i style="background:var(--chinh-nhat)"></i>khác</span></div></div>
      <div class="the"><span class="tt">Từ trường (series đã chọn)</span><div class="thanh hep">{thanh(tu, "var(--phu)", tu_tong)}</div></div>
    </div>
    <div class="luu-y"><b>Hệ quả cho thiết kế</b><p>Phải chọn một series mỗi mặt phẳng, đưa về cùng số lát và kích thước, chuẩn hoá cường độ theo series. Hãng máy là lối tắt tiềm ẩn: đặc trưng giai đoạn 1 đoán đúng hãng máy ở 82,8% số ca.</p></div>
  </div>
  {chan()}
</section>'''


def slide_quy_trinh() -> str:
    buoc = [("Thu thập dữ liệu", "4.407 ca, 569 GB; đọc header DICOM"), ("Phân tích dữ liệu", "ảnh, báo cáo, nhãn"),
            ("Tiền xử lý ảnh", "series nhạy dịch → 3 × 24 × 256²"), ("Rút nhãn yếu", "từ điển, LLM cho 4.349 báo cáo"),
            ("Giai đoạn 1", "DINOv2 đóng băng + head attention"), ("Đánh giá, đo lại", "AUC, bootstrap; sửa chọn epoch"),
            ("Giai đoạn 2", "CNN 2.5D train toàn mạng, 6 model"), ("Gộp và triển khai", "gộp theo hạng, web demo")]
    o = "".join(f'<div class="b{" dich" if i >= 6 else ""}"><span class="so">{i + 1:02d}</span><b>{t}</b><span>{m}</span></div>'
                for i, (t, m) in enumerate(buoc))
    ghi_chu = ("Quy trình tám bước. Hai bước đầu tìm hiểu dữ liệu trước khi chọn model: những gì đo được ở đó quyết định "
               "cách tiền xử lý, cách rút nhãn và cách đánh giá. Bước sáu là kiểm tra lại chính cách đo của nhóm, và dẫn "
               "tới thay đổi lớn nhất về phương pháp. Hai bước viền xanh là model cuối và web demo.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">3 · Quy trình</span>
  <h2>8 bước, từ dữ liệu thô tới web demo</h2>
  <div class="qt">{o}</div>
  {chan()}
</section>'''


def slide_tien_xu_ly(T: dict) -> str:
    the = [("txl_goc.png", "Lát gốc", f"512 × 512, {T['n_lat_mau']['Sagittal']} lát; hiển thị min–max"),
           ("txl_chuan.png", "Chuẩn hoá", "cắt phân vị 1–99 trên cả series, đổi về 256 × 256"),
           ("txl_24_lat.png", "24 lát cách đều", "mỗi mặt phẳng một khối 24 × 256 × 256"),
           ("txl_rgb.png", "Cửa sổ 2.5D", "3 lát liền kề → 3 kênh R, G, B")]
    o = "".join(f'<figure><img src="{uri(a)}" alt=""><figcaption><b>{t}</b><span>{m}</span></figcaption></figure>'
                for a, t, m in the)
    ghi_chu = ("Đường đi của một series Sagittal. Lát gốc 512 điểm ảnh, hiển thị thô thì tối vì vài điểm rất sáng. Chuẩn "
               "hoá cắt cường độ ở phân vị 1 và 99 tính trên cả series, không theo từng lát, để giữ độ sáng tương đối giữa "
               "các lát; tính trước khi đổi kích thước vì nội suy làm lệch phân vị. Sau đó lấy 24 lát cách đều. Khi vào "
               "CNN, ba lát liền kề ghép thành ba kênh màu: phần có màu là chỗ ba lát khác nhau, tức thông tin chiều sâu. "
               "Series được chọn theo cờ nhạy dịch, không theo tên vì tên có thể đã bị thay khi ẩn danh.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">3 · Quy trình</span>
  <h2>Tiền xử lý: từ DICOM thô tới khối 3 × 24 × 256 × 256</h2>
  <div style="display:grid; gap:26px; min-height:0; align-content:start">
    <div class="bon-anh">{o}</div>
    <p style="font-size:25px; color:var(--muc-2)">Chọn series <b style="color:var(--muc)">nhạy dịch</b> theo cờ trong train_series.csv; <b style="color:var(--muc)">{CHON_SERIES}%</b> số ca đủ cả ba mặt phẳng. Web demo gọi lại đúng hàm đọc ảnh lúc huấn luyện.</p>
  </div>
  {chan()}
</section>'''


def slide_roc(D: dict) -> str:
    tb = D["cuoi"]["tung_benh"]
    ten = {"Medial OA": "Thoái hoá khoang trong", "Baker's": "Nang Baker", "ACL": "Dây chằng chéo trước",
           "Synovitis": "Viêm màng hoạt dịch", "Lateral OA": "Thoái hoá khoang ngoài"}
    dong = "".join(f'<tr><td>{ten[k]}</td><td class="so">{so_vn(tb[k], 3)}</td></tr>'
                   for k in ("Medial OA", "Baker's", "ACL", "Lateral OA", "Synovitis"))
    ghi_chu = ("Đường cong ROC của model cuối cho từng bệnh trên 58 ca bác sĩ. Mỗi điểm là một ngưỡng: trục ngang là tỷ lệ "
               "ca âm bị báo nhầm, trục dọc là tỷ lệ ca dương được phát hiện; diện tích dưới đường cong là AUC. Thoái hoá "
               "khoang trong và nang Baker phát hiện phần lớn ca dương trước khi báo nhầm ca nào. Viêm màng hoạt dịch nằm "
               "gần đường chéo nhất, đúng là bệnh yếu nhất. Đường cong dạng bậc thang vì mỗi bệnh chỉ có 9 đến 35 ca dương.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">7 · Giai đoạn 2: model cuối</span>
  <h2>Đường cong ROC từng bệnh của model cuối</h2>
  <div class="hai lech2" style="min-height:0">
    <img src="{uri("roc.png")}" alt="ROC" style="width:100%; height:100%; object-fit:contain; object-position:left top; min-height:0">
    <div class="the"><span class="tt">AUC trên 58 ca bác sĩ</span>
      <table>{dong}</table>
      <p class="gc">Mỗi bệnh chỉ có 9–35 ca dương tính nên đường cong có dạng bậc thang. Macro AUC <b class="so">{so_vn(D['cuoi']['nhom']['gop_6_cnn']['auc'], 3)}</b>.</p>
    </div>
  </div>
  {chan()}
</section>'''


def slide_lien_quan() -> str:
    """Muc 1.2 cua bao cao, rut gon: chi so lieu da co trong bao cao (da doi chieu nguon)."""
    goi = [("KneeMRI", "917 chuỗi Sagittal, 1 nhãn (ACL)", "HOG + SVM trên vùng khoanh tay; AUC 0,894"),
           ("MRNet", "1.370 ca, 3 mặt phẳng, 3 nhãn", "CNN 2D mỗi mặt phẳng, max pooling, hồi quy logistic"),
           ("ELNet", "KneeMRI và MRNet", "CNN nhỏ dưới 1 MB, huấn luyện từ đầu")]
    nhan = [("ChestX-ray8", "X-quang ngực, hơn 100.000 ảnh, 8 nhãn", "NLP rút nhãn từ báo cáo"),
            ("CheXpert", "X-quang ngực, 14 nhãn", "Bộ luật, xử lý câu phủ định và không chắc chắn"),
            ("CheXbert", "X-quang ngực, 14 nhãn", "BERT học từ bộ luật và ít nhãn chuyên gia, vượt bộ luật"),
            ("GPT-4", "Báo cáo nhiều ngôn ngữ", "LLM chuyển báo cáo tự do thành dữ liệu có cấu trúc")]
    bang = lambda dong: "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in dong)
    ghi_chu = ("Trước khi làm, nhóm xem các công trình gần nhất. Với MRI gối có KneeMRI, MRNet và ELNet: cả ba đều có "
               "nhãn đầy đủ cho mọi ca và chỉ dự đoán một đến ba nhãn. Hướng rút nhãn từ báo cáo chẩn đoán thì đã có ở "
               "X-quang ngực: ChestX-ray8 dùng xử lý ngôn ngữ tự nhiên, CheXpert dùng bộ luật, CheXbert thay bộ luật bằng "
               "BERT, và gần đây GPT-4 được thử trên báo cáo nhiều ngôn ngữ. Đồ án khác ở hai điểm: 12 nhãn cùng lúc, và "
               "chỉ 1,32% số ca có nhãn bác sĩ. Nhóm đi đúng chuỗi trên: từ điển luật trước, LLM sau.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}">
  <span class="eyebrow">1 · Bài toán</span>
  <h2>Nghiên cứu liên quan: MRI gối và nhãn rút từ báo cáo</h2>
  <div style="display:grid; gap:26px; min-height:0; align-content:start">
    <div class="hai lq">
      <div class="the"><span class="tt">MRI gối: nhãn đầy đủ, 1–3 nhãn</span><table>{bang(goi)}</table></div>
      <div class="the"><span class="tt">Nhãn rút từ báo cáo chẩn đoán</span><table>{bang(nhan)}</table></div>
    </div>
    <div class="luu-y"><b>Đồ án khác ở hai điểm</b><p>12 nhãn cùng lúc, và chỉ 1,32% số ca có nhãn bác sĩ. Nhóm đi theo chuỗi trên: từ điển luật trước, LLM sau, cho báo cáo MRI gối viết bằng 9 ngôn ngữ.</p></div>
  </div>
  {chan()}
</section>'''


def slide_attention() -> str:
    """Cong thuc 3.1-3.4 cua bao cao (muc 3.4)."""
    ct = [("Đặc trưng cửa sổ k", "f<sub>k</sub> = LayerNorm( E(x<sub>k</sub>) + e<sub>p(k)</sub> )"),
          ("Điểm chú ý, 12 bệnh cùng lúc", "s<sub>k</sub> = U · tanh( W f<sub>k</sub> + b ) + c"),
          ("Softmax theo cửa sổ, riêng bệnh l", "a<sub>k,l</sub> = exp(s<sub>k,l</sub>) / Σ<sub>j</sub> exp(s<sub>j,l</sub>)"),
          ("Gộp rồi dự đoán", "g<sub>l</sub> = Σ<sub>k</sub> a<sub>k,l</sub> f<sub>k</sub>,&nbsp;&nbsp; p<sub>l</sub> = σ( w<sub>l</sub> · g<sub>l</sub> + b<sub>l</sub> )")]
    o = "".join(f'<div><span>{t}</span><code class="ct">{c}</code></div>' for t, c in ct)
    ghi_chu = ("Đây là phần đầu attention của giai đoạn 2. Mỗi cửa sổ 2.5D qua EfficientNet thành một vector, cộng vector "
               "nhúng của mặt phẳng. Một mạng nhỏ hai lớp cho điểm chú ý của 12 bệnh cùng lúc, rồi softmax lấy theo chiều "
               "cửa sổ, riêng cho từng bệnh. Khác với attention pooling gốc, mỗi bệnh có một cột trọng số và một vector "
               "riêng, nên lát quan trọng của bệnh phổ biến không lấn lát của bệnh hiếm. Hàm mất mát là binary "
               "cross-entropy trên nhãn mềm. Trọng số a này cũng là thứ web demo dùng để chỉ lát mô hình chú ý.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}" data-k="attention">
  <span class="eyebrow">7 · Giai đoạn 2: CNN 2.5D</span>
  <h2>Attention riêng từng bệnh: mỗi bệnh tự chọn lát của mình</h2>
  <div class="hai" style="min-height:0">
    <div class="the ct-the">{o}</div>
    <div class="the"><span class="tt">Đọc công thức</span>
      <ul class="ds" style="font-size:27px; gap:20px">
        <li><b>K cửa sổ mỗi ca:</b> huấn luyện 3 × 8 = 24, suy luận 3 × 24 = 72 (mỗi lát làm tâm một cửa sổ).</li>
        <li><b>Mỗi bệnh một cột trọng số riêng:</b> xác suất bệnh l chỉ phụ thuộc vector gộp của chính nó, nên bệnh phổ biến không lấn bệnh hiếm.</li>
        <li><b>Mặt phẳng thiếu</b> nhận điểm −∞ trước softmax, trọng số bằng 0.</li>
        <li><b>Mất mát:</b> binary cross-entropy trên nhãn mềm (nhãn LLM chia 100).</li>
      </ul></div>
  </div>
  {chan()}
</section>'''


def slide_giao_thuc(hang: str) -> str:
    ghi_chu = ("Bài học từ lần đo lại được đưa thành giao thức cho giai đoạn 2, chốt trước khi chấm. Huấn luyện bằng nhãn "
               "LLM mềm, 58 ca bác sĩ không vào huấn luyện, và bỏ cả ca có báo cáo trùng với ca bác sĩ. Số epoch cố định, "
               "lấy trọng số EMA ở epoch cuối, không chọn epoch theo điểm ca bác sĩ. Hai mô hình chạy song song trên hai "
               "GPU T4 đám mây. Khoảng tin cậy tính bằng bootstrap 2.000 lần trên 58 ca.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}" data-k="giao_thuc">
  <span class="eyebrow">7 · Giai đoạn 2: CNN 2.5D</span>
  <h2>Chốt cách huấn luyện trước khi chấm 58 ca bác sĩ</h2>
  <div style="display:grid; gap:44px; min-height:0; align-content:start">
    <div class="ong">{hang}
    </div>
    <div class="luu-y"><b>Rút kinh nghiệm từ lần đo lại</b><p>Ở giai đoạn 1, chọn epoch theo điểm của chính các ca đang chấm làm điểm lạc quan: 0,811 đo lại chỉ còn 0,764. Ở giai đoạn 2, số epoch, trọng số EMA và cách gộp các CNN đều chốt trước; khoảng tin cậy tính bằng bootstrap 2.000 lần.</p></div>
  </div>
  {chan()}
</section>'''


# Ca 1 la ca mau 1 tren web; ca 3 khong len web (mo hinh bat 1/3) - cung hai ca voi hinh 5.x cua bao cao
CA_PHAN_TICH = ((1, "Baker's", "Nang Baker"), (3, "Synovitis", "Viêm màng hoạt dịch"))


def slide_phan_tich_ca(D: dict) -> str:
    """Trong so chu y cua 1 benh tren 72 cua so, ve bang HTML (cung so voi muc 5.6 cua bao cao)."""
    import pandas as pd
    goc = REPO / "data" / "web_demo"
    uid = dict(enumerate((goc / "ca_mau" / "danh_sach.txt").read_text().split(), 1))
    tr = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    mau = ("var(--chinh)", "var(--phu)", "var(--cu)")
    ca = []
    for k, ma, ten in CA_PHAN_TICH:
        f = goc / "phan_tich_ca" / f"ca_{k}.json"
        if not f.exists():
            raise SystemExit(f"Thieu {f.relative_to(REPO)}: chay scripts/71_xuat_word_pbl6.py truoc (web demo dang chay)")
        d = json.loads(f.read_text(encoding="utf-8"))
        b = next(x for x in d["benh"] if x["ma"] == ma)
        tong = sum(map(sum, b["chu_y"]))
        a = [[v / tong for v in hang] for hang in b["chu_y"]]
        j, t = b["lat_chinh"]["mat_phang"], b["lat_chinh"]["lat"]
        ca.append(dict(ten=ten, p=b["p"], bac_si=int(tr.loc[uid[k], ma]), a=a, j=j, t=t, w=b["lat_chinh"]["trong_so"],
                       anh=d["mat_phang"][j]["lat"][t], mp=d["mat_phang"][j]["mat_phang"]))
    cao = max(max(map(max, c["a"])) for c in ca) * 1.08
    deu = 1 / 72
    hang = []
    for i, c in enumerate(ca):
        cot = "".join(f'<i style="height:{100 * v / cao:.2f}%; background:{mau[jj]}"></i>'
                      for jj in range(3) for v in c["a"][jj])
        gap = round(c["w"] * 72)
        y = (f"Chú ý dồn vào vài lát liền nhau; lát mạnh nhất nặng gấp khoảng {gap} lần mức chia đều." if i == 0 else
             f"Chú ý trải gần đều trên cả ba mặt phẳng. Đây cũng là bệnh có AUC thấp nhất "
             f"({so_vn(D['cuoi']['tung_benh'][CA_PHAN_TICH[1][1]], 3)}).")
        hang.append(f'''<div class="pt-hang">
      <figure><img src="data:image/jpeg;base64,{c["anh"]}" alt=""><figcaption>{c["mp"]}, lát {c["t"] + 1}</figcaption></figure>
      <div class="pt-bd"><b>Ca {"AB"[i]} · {c["ten"]}</b><span>bác sĩ ghi {"có" if c["bac_si"] else "không"}, mô hình {so_vn(c["p"], 2)}</span>
        <div class="pt-cot">{cot}<em style="bottom:{100 * deu / cao:.2f}%">{"<s>chia đều (1/72)</s>" if i == 0 else ""}</em></div>
        <div class="pt-truc"><span>Sagittal</span><span>Coronal</span><span>Axial</span></div></div>
      <p>{y}</p>
    </div>''')
    ghi_chu = ("Macro AUC cho biết mô hình xếp hạng tốt đến đâu nhưng không cho biết mô hình nhìn vào đâu. Mỗi cột là trọng "
               "số chú ý của một bệnh trên 72 cửa sổ, ba màu là ba mặt phẳng; vạch chấm là mức chia đều. Ca A là ca mẫu "
               "trên web: với nang Baker, chú ý dồn vào hai cụm vài lát ở Sagittal và Coronal. Ca B: với viêm màng hoạt "
               "dịch, chú ý trải gần đều, mô hình chỉ cho 0,39 dù bác sĩ ghi có. Tám ca mẫu quá ít để kết luận chung, nên "
               "đây là quan sát, chưa phải kiểm định.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}" data-k="phan_tich_ca">
  <span class="eyebrow">7 · Giai đoạn 2: model cuối</span>
  <h2>Mô hình nhìn vào đâu: trọng số chú ý trên 72 cửa sổ</h2>
  <div class="pt">{"".join(hang)}</div>
  {chan()}
</section>'''


def slide_gioi_han() -> str:
    ghi_chu = ("Giới hạn đầu tiên là 58 ca: khoảng tin cậy vẫn rộng, và tập này lệch so với toàn bộ dữ liệu vì không ca "
               "nào lành. 58 ca không vào huấn luyện ở giai đoạn 2 nhưng đã dùng để kiểm định chéo ở giai đoạn 1 và được "
               "xem nhiều lần khi phát triển, nên đây là đánh giá nội bộ. Chọn CNN làm kết quả chính thay vì bản gộp "
               "50/50 là chọn một trong hai sau khi thấy điểm; chênh 0,033 nên kết luận không đổi, nhưng nhóm ghi rõ. "
               "Nhãn huấn luyện là nhãn LLM, chỉ khớp bác sĩ khoảng 81% số ô, và mô hình có thể học lối tắt theo hãng "
               "máy mà nhóm chưa đo.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}" data-k="gioi_han">
  <span class="eyebrow">9 · Giới hạn và hướng tiếp</span>
  <h2>Điều bài này chưa làm được</h2>
  <div class="the" style="align-self:start; padding:38px 44px">
    <ul class="ds" style="font-size:30px; gap:26px">
      <li><b>58 ca là trần cứng.</b> Khoảng tin cậy rộng khoảng 0,06–0,09, và tập này lệch: không ca nào lành, trung bình 4,1 bệnh mỗi ca.</li>
      <li><b>58 ca đã được xem nhiều lần</b> khi phát triển (giai đoạn 1 dùng để kiểm định chéo): đây là đánh giá nội bộ, không phải test độc lập.</li>
      <li><b>Chọn CNN thay bản gộp 50/50</b> là chọn 1 trong 2 sau khi xem điểm. Chênh 0,033 nên kết luận không đổi, nhưng ghi rõ.</li>
      <li><b>Nhãn huấn luyện là nhãn LLM</b>, khớp bác sĩ khoảng 81% số ô. Xác suất chưa hiệu chỉnh; mô hình có thể học lối tắt theo hãng máy, chưa đo được ảnh hưởng.</li>
    </ul></div>
  {chan()}
</section>'''


def slide_huong_tiep() -> str:
    the = [("01", "Tập kiểm tra lớn hơn, độc lập", "Có nhãn bác sĩ, chưa từng xem khi phát triển, để phân xử các chênh lệch nhỏ cỡ 0,01."),
           ("02", "Độ phân giải và số lát cao hơn", "Cho tổn thương nhỏ; viêm màng hoạt dịch (0,718) và thoái hoá khoang ngoài (0,840) còn yếu nhất."),
           ("03", "Thêm seed và backbone rồi gộp", "6 CNN hơn 2 CNN +0,011; mỗi mô hình thêm khoảng 3 giờ GPU miễn phí.")]
    o = "".join(f'<div class="b"><span class="so">{s}</span><b>{t}</b><span>{m}</span></div>' for s, t, m in the)
    ghi_chu = ("Hướng tiếp theo, xếp theo thứ tự nên làm. Quan trọng nhất là một tập kiểm tra có nhãn lớn hơn và độc lập, "
               "vì với 58 ca thì các chênh lệch cỡ 0,01 chưa phân xử được. Sau đó tăng độ phân giải và số lát cho các tổn "
               "thương nhỏ, rồi thêm seed và backbone để gộp.")
    return f'''<section class="slide" data-ghi-chu="{h.escape(ghi_chu)}" data-k="huong_tiep">
  <span class="eyebrow">9 · Giới hạn và hướng tiếp</span>
  <h2>Hướng tiếp, theo thứ tự nên làm</h2>
  <div class="qt ht">{o}</div>
  {chan()}
</section>'''


def roc_svg(D: dict) -> str:
    """Ve lai dung ROC cua scripts/71, bang SVG: chu >= 18px, so kieu Viet.

    Chi doc du lieu cuc bo; khong ghi de roc.png. Kiem thu tu ca va AUC truoc khi ve.
    """
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score, roc_curve

    sys.path.insert(0, str(REPO / "src"))
    from rsna_knee.config import LABELS
    from rsna_knee.oof import gold_theo_fold, ma_thu_tu

    y, _, uid = gold_theo_fold()
    with np.load(REPO / "reports" / "day15_model_cuoi_du_doan.npz") as z:
        if str(z["uid_md5"]) != ma_thu_tu(uid):
            raise SystemExit("ROC: thu tu ca du doan khac thu tu ca bac si")
        P = np.mean([np.column_stack([rankdata(z[f"model_{s}"][:, j]) for j in range(12)])
                     for s in range(1, 7)], axis=0)
    ten = ("Dây chằng chéo trước", "Dây chằng bên trong", "Sụn chêm trong", "Sụn chêm ngoài",
           "Thoái hoá khoang trong", "Thoái hoá khoang ngoài", "Thoái hoá bánh chè – đùi", "Tràn dịch khớp",
           "Viêm màng hoạt dịch", "Nang Baker", "Dập xương", "Gãy xương")
    if y.shape != (58, 12) or P.shape != y.shape or len(LABELS) != len(ten):
        raise SystemExit("ROC: mong doi 58 ca, 12 benh")
    auc = [roc_auc_score(y[:, j], P[:, j]) for j in range(12)]
    if abs(float(np.mean(auc)) - D["cuoi"]["nhom"]["gop_6_cnn"]["auc"]) > 5e-4:
        raise SystemExit("ROC: macro AUC tinh lai lech so lieu bao cao")
    for j, k in enumerate(LABELS):
        if abs(auc[j] - D["cuoi"]["tung_benh"][k]) > 5e-4:
            raise SystemExit(f"ROC: AUC benh {k} lech so lieu bao cao")

    # 1040 x 730 duoc hien thi 1:1 tren slide 17; title 20px, chu trong o 18px.
    svg = ['<svg class="roc" viewBox="0 0 1040 730" role="img" aria-label="ROC">']
    for j, t in enumerate(ten):
        ox, oy = 36 + (j % 4) * 250, 8 + (j // 4) * 224
        x, top, w, ht = ox + 36, oy + 54, 200, 148
        for k, line in enumerate(textwrap.wrap(t, 23, break_long_words=False, break_on_hyphens=False)):
            svg.append(f'<text class="roc-tieu-de" x="{ox + 136}" y="{oy + 20 + k * 23}" text-anchor="middle">{h.escape(line)}</text>')
        svg.append(f'<path d="M{x},{top}V{top + ht}H{x + w}" fill="none" stroke="var(--muc-2)" stroke-width="1.5"/>')
        svg.append(f'<path d="M{x},{top + ht}L{x + w},{top}" stroke="var(--truc)" stroke-width="1.5" stroke-dasharray="3 3"/>')
        fpr, tpr, _ = roc_curve(y[:, j], P[:, j])
        points = " ".join(f"{x + a * w:.3f},{top + (1 - b) * ht:.3f}" for a, b in zip(fpr, tpr))
        svg.append(f'<polygon points="{x},{top + ht} {points} {x + w},{top + ht}" fill="var(--phu)" fill-opacity="0.08"/>')
        svg.append(f'<polyline points="{points}" fill="none" stroke="var(--phu)" stroke-width="2.5"/>')
        for v in (0, 0.5, 1):
            if j % 4 == 0:
                svg.append(f'<text x="{x - 8}" y="{top + (1 - v) * ht + 6}" text-anchor="end">{so_vn(v, 1)}</text>')
            if j // 4 == 2:
                svg.append(f'<text x="{x + v * w}" y="{top + ht + 25}" text-anchor="middle">{so_vn(v, 1)}</text>')
        svg.append(f'<text x="{x + w - 6}" y="{top + ht - 40}" text-anchor="end">{int(y[:, j].sum())} ca dương</text>')
        svg.append(f'<text x="{x + w - 6}" y="{top + ht - 14}" text-anchor="end">AUC {so_vn(auc[j], 3)}</text>')
    svg.append('<text x="540" y="724" text-anchor="middle" class="roc-truc">Tỷ lệ dương tính giả (1 − độ đặc hiệu)</text>')
    svg.append('<text transform="translate(20 350) rotate(-90)" text-anchor="middle" class="roc-truc">Độ nhạy</text></svg>')
    return "".join(svg)


def thay_dung_mot(s: str, tim: str, thay: str) -> str:
    """Khong am tham bo qua khi HTML/JS nguon thay doi."""
    if s.count(tim) != 1:
        raise SystemExit(f"Can khop dung 1 lan, thay {s.count(tim)}: {tim[:180]}")
    return s.replace(tim, thay, 1)


def so_viet_trong_chu(s: str) -> str:
    """Chi doi so thap phan trong van ban; giu 4.407, 2.5D, IP va ten file.

    Khong goi tren CSS, JSON hay bieu thuc JS: chung van can dau cham cu phap.
    """
    def doi(m: re.Match) -> str:
        t = m.group(0)
        if re.fullmatch(r"[1-9]\d{0,2}(?:\.\d{3})+", t):
            return t
        return t.replace(".", ",")
    return re.sub(r"(?<![\w.,])\d+\.\d+(?![\w.,])", doi, s)


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
.ml { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:34px 56px; align-content:center; }
.ml > div { border-top:1.5px solid var(--ke); padding-top:20px; display:grid; gap:6px; }
.ml .so { font-size:40px; color: var(--chinh); line-height:1; }
.ml b { font-size:32px; }
.ml span:last-child { font-size:23px; color: var(--muc-2); }
.ba-anh { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:18px; align-content:start; }
.ba-anh figure, .bon-anh figure { margin:0; display:grid; gap:12px; align-content:start; }
.ba-anh img, .bon-anh img { width:100%; aspect-ratio:1; object-fit:cover; display:block; border-radius:8px; background:#000; }
.ba-anh figcaption { font-size:22px; color: var(--muc-2); display:grid; gap:2px; }
.ba-anh figcaption b { color: var(--muc); }
.ba-anh figcaption span, .bon-anh figcaption span { font-size:20px; color: var(--muc-3); }
.bon-anh { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:30px; }
.bon-anh figcaption { display:grid; gap:4px; font-size:24px; }
.bon-anh figure + figure { position:relative; }
.bon-anh figure + figure::before { content:"→"; position:absolute; left:-26px; top:150px; font:28px var(--f-so); color: var(--muc-3); }
.thanh.hep .dong { grid-template-columns: 150px minmax(0,1fr) 92px; font-size:22px; }
.qt { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:36px 52px; align-content:center; }
.qt .b { background: var(--mat); border:1.5px solid var(--ke); border-radius:10px; padding:24px 26px; display:grid; gap:8px; align-content:start; position:relative; min-height:190px; }
.qt .b.dich { border-color: var(--phu); box-shadow: inset 5px 0 0 var(--phu); }
.qt .b .so { font-size:22px; color: var(--muc-3); }
.qt .b b { font-size:28px; line-height:1.2; }
.qt .b span:last-child { font-size:22px; color: var(--muc-2); line-height:1.4; }
.qt .b:not(:nth-child(4n))::after { content:"→"; position:absolute; right:-40px; top:50%; transform:translateY(-50%); font:28px var(--f-so); color: var(--muc-3); }

/* May chieu 1920 x 1080: rong tieu de, san chu va tuong phan. */
:root { --muc-2:#334155; --muc-3:#4b5866; --ke:#a6b2bf; --truc:#8493a3; --phim-mo:#cbd5e1; }
h2 { width:100%; max-width:100%; font-size:58px; text-wrap:balance; word-break:normal; overflow-wrap:normal; }
.slide.toi:first-of-type h1 { width:1700px; max-width:100% !important; font-size:82px !important; text-wrap:balance; }
.nhom .nh, .nhom th, th, .ong .nh, .ong .b .so { font-size:20px; }
.slide .chu-giai { font-size:22px !important; }
.eyebrow { font-size:21px; }
.chan { font-size:18px; }
code { font-size:max(20px, 0.88em); border-width:1.5px; }
.the, .ong .b, .qt .b { border-width:1.5px; }
.ong .b.may { border-color:#8493a3; }
.ong .m, .qt .b::after, .bon-anh figure::before { font-size:30px; font-weight:600; color:var(--muc-2); }
.slide.toi { --muc-2:#cbd5e1; --muc-3:#cbd5e1; }
.slide.toi .phu { opacity:1; }

/* Be Vietnam Pro co chu so deu, dau phay hep; giu font chinh va font ma lenh. */
.so, td.so, th.so, .so-lon, .dai-so .gt, .doi .gt, .ba-so .gt,
.thanh .gt, .luoi .the .gt { font-family:var(--f-chu) !important; font-variant-numeric:tabular-nums; letter-spacing:0; }
svg .chu, svg .chu-so, svg .chu-dam, svg .chu-phu { font-size:18px; }
svg .chu-so { font-family:var(--f-chu); font-variant-numeric:tabular-nums; }
svg .chu, svg .chu-so { fill:var(--muc-3); }
.roc { width:1040px; max-width:100%; height:auto; display:block; }
.roc text { font:400 18px var(--f-chu); font-variant-numeric:tabular-nums; fill:var(--muc-2); }
.roc .roc-tieu-de, .roc .roc-truc { font-size:20px; fill:var(--muc); }

/* Dau cham nam ngoai khoi chu: moi dong xuong hang thang voi chu dong dau. */
ul.ds { padding-left:0; list-style:none; }
ul.ds > li { position:relative; padding-left:1.05em; }
ul.ds > li::before { content:"•"; position:absolute; left:0; top:0; color:var(--muc-3); }

/* Selector theo ten slide (data-k), khong theo vi tri: them, bot slide khong lam lech. */
.slide[data-k="du_lieu"] > .hai { align-self:start; }
.ba-anh { align-content:stretch; grid-template-rows:minmax(0,1fr); }
.ba-anh figure { grid-template-rows:minmax(0,1fr) auto; }
.ba-anh img { height:100%; min-height:0; aspect-ratio:auto; object-fit:contain; }
.slide[data-k="c3"] .thanh .dong { grid-template-columns:190px minmax(0,1fr) 70px; gap:12px; }
.qt { align-content:start; padding-top:12px; }
.slide[data-k="giao_thuc"] .ong .m { visibility:hidden; }
.slide[data-k="giao_thuc"] .ong .b { padding:28px 30px; gap:12px; min-height:220px; }
.slide[data-k="giao_thuc"] .ong .b b { font-size:30px; }
.slide[data-k="giao_thuc"] .ong .b span { font-size:24px; }
.slide[data-k="c13"] .ong { gap:22px; align-content:start; }
.slide[data-k="c13"] .ong .b { min-height:200px; padding:24px 26px; }
.slide[data-k="c13"] .ong .b b { font-size:28px; }
.slide[data-k="c13"] .ong .b span { font-size:23px; }
.slide[data-k="c17"] > .hai { align-self:start; align-items:start; }
.slide[data-k="roc"] > .hai { grid-template-columns:1040px minmax(0,1fr); align-self:start; align-items:start; }
.slide[data-k="roc"] .the { align-self:start; }
.hai.lq { grid-template-columns: minmax(0,1fr) minmax(0,1.15fr); align-items:start; }
.hai.lq table { font-size:21px; }
.hai.lq td { padding:10px 14px 10px 0; vertical-align:top; line-height:1.35; }
.hai.lq td:first-child { white-space:nowrap; }
.ct-the { gap:30px; align-self:start; padding:34px 40px; }
.ct-the > div { display:grid; gap:8px; }
.ct-the > div > span { font-size:23px; color:var(--muc-2); }
code.ct { font:400 40px/1.4 var(--f-tieu-de); background:none; border:0; padding:0; color:var(--muc); }
code.ct sub { font-size:0.62em; }
.slide[data-k="attention"] > .hai { align-self:start; align-items:start; }
.pt { display:grid; gap:30px; align-content:start; min-height:0; }
.pt-hang { display:grid; grid-template-columns: 300px minmax(0,1fr) 400px; gap:40px; align-items:center; }
.pt figure { margin:0; display:grid; gap:8px; }
.pt img { width:300px; height:300px; object-fit:cover; border-radius:8px; display:block; background:#000; }
.pt figcaption { font-size:20px; color:var(--muc-3); }
.pt-bd { display:grid; gap:4px; }
.pt-bd > b { font-size:27px; }
.pt-bd > span { font-size:22px; color:var(--muc-2); }
.pt-cot { position:relative; height:200px; display:flex; align-items:flex-end; gap:2px; border-bottom:1.5px solid var(--muc-2); margin-top:12px; }
.pt-cot i { flex:1; display:block; min-height:1px; }
.pt-cot em { position:absolute; left:0; right:0; border-top:2px dotted var(--muc-3); font-style:normal; }
.pt-cot em s { position:absolute; right:0; bottom:4px; text-decoration:none; font-size:19px; color:var(--muc-3); }
.pt-truc { display:grid; grid-template-columns:repeat(3,1fr); text-align:center; font-size:20px; color:var(--muc-2); padding-top:6px; }
.pt-hang > p { font-size:25px; color:var(--muc-2); line-height:1.45; }
.slide[data-k="gioi_han"] > .the { max-width:1500px; }
.qt.ht { grid-template-columns: repeat(3, minmax(0,1fr)); }
.qt.ht .b { min-height:300px; padding:32px 34px; gap:14px; }
.qt.ht .b b { font-size:32px; }
.qt.ht .b span:last-child { font-size:25px; }
.qt.ht .b:not(:nth-child(4n))::after { content:none; }
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

    D = json.loads(re.search(r"const D = (\{.*?\});\n", goc, re.DOTALL).group(1))
    T = json.loads((ANH_SLIDE / "thong_ke.json").read_text(encoding="utf-8"))
    # Slide CNN 2.5D cu (13) qua day: tach hang "Giao thuc" sang slide rieng; cac id s13-* JS van dien theo id.
    gd2 = sua_slide(cu[12])
    m_gt = re.search(r'\n    <span class="nh">Giao thức, chốt trước khi chấm</span>\n    <div class="hang">\n.*?\n    </div>', gd2, re.S)
    if not m_gt:
        raise SystemExit("Khong tim thay hang Giao thuc trong slide CNN 2.5D")
    hang_gt = m_gt.group(0).replace('<span class="nh">Giao thức, chốt trước khi chấm</span>\n    ', "")
    gd2 = gd2.replace(m_gt.group(0), "")
    moi = {"muc_luc": slide_muc_luc, "lien_quan": slide_lien_quan, "du_lieu": lambda: slide_du_lieu(D, T),
           "eda": lambda: slide_eda(D, T), "quy_trinh": slide_quy_trinh, "tien_xu_ly": lambda: slide_tien_xu_ly(T),
           "attention": slide_attention, "giao_thuc": lambda: slide_giao_thuc(hang_gt), "roc": lambda: slide_roc(D),
           "phan_tich_ca": lambda: slide_phan_tich_ca(D), "web": lambda: slide_web(ANH_WEB),
           "gioi_han": slide_gioi_han, "huong_tiep": slide_huong_tiep}
    hien = [bia(tt, logo_tu_mau(tt["mau_docx"]))]
    for g in GIU:
        s = moi[g]() if isinstance(g, str) else (gd2 if g == 13 else sua_slide(cu[g - 1]))
        # khoa CSS theo ten slide (khong theo vi tri): .slide[data-k="c13"], [data-k="roc"]...
        k = g if isinstance(g, str) else f"c{g}"
        if 'data-k="' not in s[:s.index(">")]:
            s = re.sub(r'^(<section class="slide[^"]*" data-ghi-chu="[^"]*")', rf'\1 data-k="{k}"', s, count=1)
        hien.append(s)
    if sorted(i for _, r, _ in NGUOI_NOI for i in r) != list(range(1, len(hien) + 1)):
        raise SystemExit(f"NGUOI_NOI khong chia dung {len(hien)} slide: sua NGUOI_NOI")
    # slide ket luan: ghi chu cap nhat 0.896
    hien[-1] = re.sub(r'data-ghi-chu="[^"]*"', f'data-ghi-chu="{h.escape(GHI_CHU_KET)}"', hien[-1], count=1)
    an = [re.sub(r'^<section class="slide[^"]*"', '<div hidden data-bo="1"', s).replace("</section>", "</div>")
          for i, s in enumerate(cu, 1) if i not in GIU and i != 1]
    an.append(re.sub(r'^<section class="slide[^"]*"', '<div hidden data-bo="1"', cu[0]).replace("</section>", "</div>")
              .replace('id="bia-so"', 'id="bia-so-cu"'))

    than = "\n\n".join(hien)
    for tim, thay in SUA:
        than = thay_dung_mot(than, tim, thay)
    # thong nhat thuat ngu voi bao cao: "ca gold" -> "ca bac si"; ghi chu thuyet trinh doc bang tieng Viet
    than = than.replace("58 ca gold", "58 ca bác sĩ").replace("ca gold", "ca bác sĩ").replace("báo cáo gold", "báo cáo ca bác sĩ")

    def viet_hoa(m: re.Match) -> str:
        g = m.group(1).replace("chấm CV", "kiểm định chéo")
        g = re.sub(r"\bModel\b", "Mô hình", re.sub(r"\bmodel\b", "mô hình", g))
        return 'data-ghi-chu="' + re.sub(r"\btrain\b(?![._])", "huấn luyện", g) + '"'
    than = re.sub(r'data-ghi-chu="([^"]*)"', viet_hoa, than)
    # Ghi chu GD2 cu noi ca phan giao thuc (nay o slide sau): bo cau cuoi
    than = thay_dung_mot(than, " Giao thức chốt trước: 12 epoch, trọng số EMA ở epoch cuối, 58 ca bác sĩ không bao giờ vào train.\"",
                         " Cách huấn luyện được chốt trước, ở slide sau.\"")
    # Xu ly tung nut chu/ghi chu, khong doi thuoc tinh width, du lieu D hay cu phap JS.
    for m in list(re.finditer(r'data-ghi-chu="[^"]*"|>[^<>]+<', than)):
        tim = m.group(0)
        thay = so_viet_trong_chu(tim)
        if tim != thay:
            than = thay_dung_mot(than, tim, thay)
    # Ghi chu moi slide mo dau bang [ten nguoi trinh bay]; slide cuoi phan dau co cau chuyen loi.
    goi = [sv["ten"].split()[-1] for sv in tt["sinh_vien"]]
    khoi_moi = list(re.finditer(r'<section class="slide[^"]*"[^>]*?data-ghi-chu="', than))
    for i, m in reversed(list(enumerate(khoi_moi, 1))):
        ai = next(sv for sv, r, _ in NGUOI_NOI if i in r)
        than = than[:m.end()] + f"[{h.escape(goi[ai])}] " + than[m.end():]
    for (_, r, chuyen), (sau, _, _) in reversed(list(zip(NGUOI_NOI, NGUOI_NOI[1:]))):
        m = list(re.finditer(r'<section class="slide[^"]*"[^>]*?data-ghi-chu="[^"]*', than))[max(r) - 1]
        than = than[:m.end()] + f" Phần tiếp theo, bạn {h.escape(goi[sau])} trình bày {chuyen}." + than[m.end():]
    dau = thay_dung_mot(dau, "</style>", CSS_THEM + "</style>")
    dau = thay_dung_mot(dau, "<title>Slide MRI gối 58 nhãn</title>", "<title>Slide PBL6 MRI gối</title>")
    trang_html = dau + than + "\n\n" + "\n".join(an) + cuoi
    anh_roc = f'<img src="{uri("roc.png")}" alt="ROC" style="width:100%; height:100%; object-fit:contain; object-position:left top; min-height:0">'
    trang_html = thay_dung_mot(trang_html, anh_roc, roc_svg(D))

    # JS: dinh dang so hien thi, khong sua Number, toFixed toa do hay JSON.
    trang_html = thay_dung_mot(
        trang_html, 'const f = (x, d = 3) => Number(x).toFixed(d);',
        'const f = (x, d = 3) => Number(Number(x).toFixed(d)).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });')
    # Do nhan cot sau khi font tai xong; van dat data-xong o cuoi de bo chup doi duoc.
    trang_html = thay_dung_mot(trang_html, '(() => {', 'document.fonts.ready.then(() => {')
    trang_html = thay_dung_mot(trang_html, '})();', '});')
    # Cung mien [0.45, 0.95], cung chieu cao vung ve 318px (H=440, top=32, bottom=90).
    for tim, thay in (
        ('y: [0.55, 0.80], tick: [0.55, 0.05], w: 340', 'y: [0.45, 0.95], tick: [0.45, 0.05], w: 340'),
        ('y: [0.65, 0.95], tick: [0.65, 0.05], w: 620, h: 330, b: 40, bw: 62, d: 3, aria: "Giai đoạn 2 so với giai đoạn 1"',
         'y: [0.45, 0.95], tick: [0.45, 0.05], w: 620, h: 330, b: 40, bw: 62, d: 3, aria: "Giai đoạn 2 so với giai đoạn 1"'),
        ('y: [0.65, 0.95], tick: [0.65, 0.05], w: 620, h: 330, b: 40, bw: 62, d: 3, aria: "Model cuối so với các nhóm"',
         'y: [0.45, 0.95], tick: [0.45, 0.05], w: 620, h: 330, b: 40, bw: 62, d: 3, aria: "Model cuối so với các nhóm"'),
    ):
        trang_html = thay_dung_mot(trang_html, tim, thay)
    trang_html = thay_dung_mot(trang_html,
        'const W = o.w || 520, H = o.h || 220, m = { l: 46, r: 12, t: 26, b: o.b || 46 };',
        '''const cungTruc = ["s5-cot", "s6-buoc", "s14-cot", "s18-cot"].includes(host.id);
    const W = cungTruc ? host.clientWidth : (o.w || 520), H = cungTruc ? 440 : (o.h || 220);
    const m = cungTruc ? { l: 64, r: 20, t: 32, b: 90 } : { l: 46, r: 12, t: 26, b: o.b || 46 };''')
    # Qua ta slide 14: noi cot ten va hai cot so de chu 18px khong de len nhau.
    trang_html = thay_dung_mot(trang_html,
        'const W = o.w || 620, hang = o.hang || 26, m = { l: o.l || 128, r: 104, t: 6, b: 30 };',
        'const W = o.w || 620, hang = o.hang || 26, m = { l: o.l || 190, r: 170, t: 6, b: 30 };')
    # Chen SVG truoc khi do chu; wrap theo be rong that, khong cat hay bo nhan cot.
    trang_html = thay_dung_mot(trang_html,
        'const lines = r.ten.split("\\n");\n      lines.forEach((t, k) => { el("text", { x: X(i), y: H - m.b + 18 + k * 15, "text-anchor": "middle", class: k ? "chu" : "chu-phu" }, g).textContent = t; });',
        '''host.appendChild(svg);
      const lines = [];
      const probe = el("text", { class: "chu-phu", visibility: "hidden" }, svg);
      const maxWidth = (W - m.l - m.r) / n - 12;
      for (const line of r.ten.split("\\n")) {
        let current = "";
        for (const word of line.split(" ")) {
          probe.textContent = current ? current + " " + word : word;
          if (current && probe.getComputedTextLength() > maxWidth) { lines.push(current); current = word; }
          else current = probe.textContent;
        }
        lines.push(current);
      }
      probe.remove();
      lines.forEach((t, k) => { el("text", { x: X(i), y: H - m.b + 26 + k * 23, "text-anchor": "middle", class: k ? "chu" : "chu-phu" }, g).textContent = t; });''')
    # nhan truc bieu do hanh trinh (trong JS): "DINOv2 dong bang" dinh vao nhan cot ke ben
    nhan_cu = 'ten: "DINOv2 đóng băng\\n58 ca, 1 mặt phẳng"'
    trang_html = thay_dung_mot(trang_html, nhan_cu, 'ten: "DINOv2\\n58 ca, 1 mặt phẳng"')
    # chu do JS dien (chan trang, dong phu slide ket luan) cung thong nhat thuat ngu
    trang_html = (trang_html.replace("58 ca gold", "58 ca bác sĩ").replace("tự gắn", "tự gán").replace("Tự gắn", "Tự gán")
                  .replace("điểm gold", "điểm ca bác sĩ").replace("CA GOLD`", "CA BÁC SĨ`"))
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
