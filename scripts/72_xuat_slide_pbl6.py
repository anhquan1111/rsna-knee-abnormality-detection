"""Xuat bo slide thuyet trinh PBL6 (20 trang) tu bo 21 slide cua scripts/70.

    uv run --extra bao-cao python scripts/70_xuat_bao_cao.py --slide   # (neu chua co reports/slide/slide.html)
    uv run --extra bao-cao python scripts/72_xuat_slide_pbl6.py        # -> reports/pbl6/slide/ (PNG, PDF, PPTX)

Lay reports/slide/slide.html (da dien so lieu), giu 12 slide noi dung theo GIU, thay bia bang bia
theo khuon truong (logo lay tu file mau Word, ten sinh vien va GVHD tu reports/pbl6/thong_tin.json),
them 7 slide moi: muc luc, du lieu, phan tich anh, quy trinh, tien xu ly, ROC, web demo. Anh va so
lieu cua slide moi do scripts/71 tao o reports/pbl6/hinh/ (web.png, slide/*.png, slide/thong_ke.json). Slide bi bo van nam
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
ANH_SLIDE = REPO / "reports" / "pbl6" / "hinh" / "slide"

# So thu tu (1-based) trong bo 21 slide; chu la slide moi. Bia moi luon dung dau.
GIU = ["muc_luc", 2, "du_lieu", "eda", 3, "quy_trinh", "tien_xu_ly", 4, 5, 6, 13, 14, 15, 17, 18, "roc", "web", 20, 21]
# Muc lon tren dong eyebrow: so cu -> so moi (them muc 2 Du lieu, 3 Quy trinh nen danh lai)
MUC = {"1": "1", "2": "4", "3": "5", "4": "6", "7": "7", "9": "9"}   # slide cu muc 5, 6, 8 khong giu
MUC_WEB = "8"
CHON_SERIES = "96,9"   # % ca du series nhay dich o ca 3 mat phang (SO_NHAT_KY cua scripts/71)
MUC_LUC = [("Bài toán", "12 bất thường, macro AUC"), ("Dữ liệu và phân tích", "4.407 ca, ảnh và báo cáo"),
           ("Quy trình, tiền xử lý", "8 bước, DICOM → khối ảnh"), ("Giai đoạn 1", "DINOv2 đóng băng"),
           ("Nhãn từ báo cáo", "từ điển, LLM"), ("Hành trình và đo lại", "0.811 → 0.764"),
           ("Giai đoạn 2", "CNN 2.5D, model cuối 0.896"), ("Web demo", "chạy cục bộ"),
           ("Giới hạn, kết luận", "điều chưa làm được")]

# Sua chu trong slide cu cho dung voi bao cao PBL6 (tim -> thay, moi cap phai khop dung 1 lan)
SUA = [
    # slide gioi han: them tap gold lech, loi tat hang may; huong tiep xep tap kiem tra doc lap len dau (khop muc 6.3)
    ("<li><b>58 ca là trần cứng.</b> Khoảng tin cậy rộng khoảng 0.06–0.09; thêm ảnh hay nhãn máy không làm nó hẹp lại.</li>",
     ("<li><b>58 ca là trần cứng.</b> Khoảng tin cậy rộng khoảng 0.06–0.09, và tập này lệch: không ca nào lành, "
     "trung bình 4,1 bệnh mỗi ca.</li>")),
    (("<li><b>58 ca đã được xem nhiều lần</b> khi phát triển (dù chưa bao giờ vào train): đây là đánh giá nội bộ, "
     "không phải test độc lập hoàn toàn.</li>"),
     ("<li><b>58 ca đã được xem nhiều lần</b> khi phát triển (giai đoạn 1 dùng để kiểm định chéo): đây là đánh giá "
     "nội bộ, không phải test độc lập.</li>")),
    ("AUC đo xếp hạng, chưa nói gì về xác suất chuẩn hay dùng lâm sàng. Rò rỉ theo hãng máy chưa chặn.</li>",
     "Xác suất chưa hiệu chỉnh; mô hình có thể học lối tắt theo hãng máy, chưa đo được ảnh hưởng.</li>"),
    ('<span class="tt">Hướng tiếp, theo thứ tự đáng làm</span>', '<span class="tt">Hướng tiếp</span>'),
    (("<li><b>Thêm seed và backbone</b> rồi gộp: 6 CNN đã hơn 2 CNN +0.011; mỗi lần thêm khoảng 3 giờ GPU miễn phí.</li>\n"
     "        <li><b>Độ phân giải và số lát cao hơn</b> cho tổn thương nhỏ; viêm màng hoạt dịch (0.71) và gãy xương (0.79) "
     "còn yếu nhất.</li>\n        <li><b>Một tập kiểm tra có nhãn lớn hơn 58 ca</b> để phân xử các chênh lệch nhỏ cỡ 0.01.</li>"),
     ("<li><b>Một tập kiểm tra có nhãn lớn hơn và độc lập</b> để phân xử các chênh lệch nhỏ cỡ 0.01.</li>\n"
     "        <li><b>Độ phân giải và số lát cao hơn</b> cho tổn thương nhỏ; viêm màng hoạt dịch (0.718) và thoái hoá "
     "khoang ngoài (0.840) còn yếu nhất.</li>\n        <li><b>Thêm seed và backbone</b> rồi gộp: 6 CNN hơn 2 CNN "
     "+0.011; mỗi lần thêm khoảng 3 giờ GPU miễn phí.</li>")),
    # tieu de, cau ket: noi thang so lieu thay vi an du; "don bay lon nhat" khong dung (nhan +0.12, CNN +0.12)
    ("<h2>Mười hai câu hỏi cho mỗi ca chụp, gần như không có đáp án mẫu</h2>",
     "<h2>12 xác suất cho mỗi ca chụp, chỉ 1,3% số ca có nhãn bác sĩ</h2>"),
    ("<h2>Từng bước cải tiến, một lần đo lại, rồi một bước nhảy</h2>",
     "<h2>Đo lại: 0.811 còn 0.764; giai đoạn 2 lên 0.885</h2>"),
    ("<h2>Vượt hẳn backbone đóng băng</h2>", "<h2>CNN 2.5D đạt 0.885, hơn DINOv2 đóng băng 0.120</h2>"),
    ("<p>Nhãn rút từ báo cáo là đòn bẩy lớn nhất khi chỉ 1,3% dữ liệu có nhãn.",
     "<p>Nhãn rút từ báo cáo cho phép dùng 98,7% số ca chưa có nhãn bác sĩ."),
    ("<p>Train cả mạng trên chính ảnh MRI vượt hẳn backbone đóng băng.",
     "<p>Huấn luyện cả mạng trên chính ảnh MRI vượt hẳn backbone đóng băng."),
    # ghi chu: bo cau dan dat, cau noi qua
    ("Khó nhất: chỉ 1,32%", "Khó nhất là chỉ 1,32%"),
    ("rồi gộp theo thứ hạng.", "rồi gộp theo hạng."),
    ("Khoảng tin cậy bắt cặp không cắt qua 0.", "Khoảng tin cậy bắt cặp không chứa 0."),
    ("Em tự viết lại hướng đó, không dùng checkpoint của ai.",
     "Em tự viết lại hướng đó, chỉ dùng trọng số ImageNet công khai, không dùng mô hình người khác đã huấn luyện cho bài này."),
    ("nên báo cáo nguyên như vậy.", "nhóm vẫn báo cáo đúng số đo."),
    ("Đây là tổn thương nhỏ, nằm trên vài lát: đặc trưng học từ ảnh thường không nhìn ra, CNN train trên chính MRI thì học được.",
     ("Đây là tổn thương nhỏ, nằm trên vài lát. Giả thuyết của nhóm là đặc trưng học từ ảnh thường không nhìn ra các "
     "tổn thương này, còn CNN huấn luyện trên chính MRI thì học được.")),
    ("Thầy hỏi nhãn LLM có đúng thật không. Em cho", "Để kiểm tra nhãn LLM có đúng không, em cho"),
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
GHI_CHU_GIOI_HAN = ("Giới hạn đầu tiên là 58 ca: khoảng tin cậy vẫn rộng, và tập này lệch so với toàn bộ dữ liệu vì "
                    "không ca nào lành. 58 ca không vào huấn luyện ở giai đoạn 2 nhưng đã dùng để kiểm định chéo ở giai "
                    "đoạn 1 và được xem nhiều lần khi phát triển, nên đây là đánh giá nội bộ. Chọn CNN làm kết quả chính "
                    "thay vì bản gộp 50/50 là chọn một trong hai sau khi thấy điểm; chênh 0.033 nên kết luận không đổi, "
                    "nhưng nhóm ghi rõ. Nhãn huấn luyện là nhãn LLM, chỉ khớp bác sĩ khoảng 81% số ô, và mô hình có thể "
                    "học lối tắt theo hãng máy mà nhóm chưa đo. Hướng tiếp theo: một tập kiểm tra có nhãn lớn hơn và độc "
                    "lập, tăng độ phân giải cho tổn thương nhỏ, rồi thêm seed để gộp.")
GHI_CHU_KET = ("Ba ý mang về. Một: khi chỉ 1,3% số ca có nhãn bác sĩ, rút nhãn từ báo cáo cho phép dùng gần hết phần "
               "còn lại, nâng điểm từ 0.627 lên 0.747. Hai: huấn luyện cả mạng trên "
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
    ghi_chu = ("Kính chào thầy. Nhóm em trình bày đề tài phát hiện 12 bất thường đầu gối trên ảnh MRI, dùng bộ dữ liệu "
               "do Hội Điện quang Bắc Mỹ RSNA công bố, trong đó chỉ 58 trên 4.407 ca có nhãn bác sĩ. Bài đi từ bài "
               "toán và dữ liệu, quy trình xử lý, cách rút nhãn từ báo cáo, hai giai đoạn mô hình, tới kết quả, web "
               "demo và giới hạn.")
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
               "không phải 6 mô hình vì trọng số mô hình số 6 bị ghi đè trên máy chủ đám mây; dự đoán của nó vẫn được chấm.")
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
  <h2>Tám bước, từ dữ liệu thô tới web demo</h2>
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
    dong = "".join(f'<tr><td>{ten[k]}</td><td class="so">{tb[k]:.3f}</td></tr>'
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
      <p class="gc">Mỗi bệnh chỉ có 9–35 ca dương tính nên đường cong có dạng bậc thang. Macro AUC <b class="so">{D['cuoi']['nhom']['gop_6_cnn']['auc']:.3f}</b>.</p>
    </div>
  </div>
  {chan()}
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
    moi = {"muc_luc": slide_muc_luc, "du_lieu": lambda: slide_du_lieu(D, T), "eda": lambda: slide_eda(D, T),
           "quy_trinh": slide_quy_trinh, "tien_xu_ly": lambda: slide_tien_xu_ly(T), "roc": lambda: slide_roc(D),
           "web": lambda: slide_web(ANH_WEB)}
    hien = [bia(tt, logo_tu_mau(tt["mau_docx"]))]
    for g in GIU:
        hien.append(moi[g]() if isinstance(g, str) else sua_slide(cu[g - 1]))
    # slide ket luan: ghi chu cap nhat 0.896
    hien[-1] = re.sub(r'data-ghi-chu="[^"]*"', f'data-ghi-chu="{h.escape(GHI_CHU_KET)}"', hien[-1], count=1)
    hien[-2] = re.sub(r'data-ghi-chu="[^"]*"', f'data-ghi-chu="{h.escape(GHI_CHU_GIOI_HAN)}"', hien[-2], count=1)
    an = [re.sub(r'^<section class="slide[^"]*"', '<div hidden data-bo="1"', s).replace("</section>", "</div>")
          for i, s in enumerate(cu, 1) if i not in GIU and i != 1]
    an.append(re.sub(r'^<section class="slide[^"]*"', '<div hidden data-bo="1"', cu[0]).replace("</section>", "</div>")
              .replace('id="bia-so"', 'id="bia-so-cu"'))

    than = "\n\n".join(hien)
    for tim, thay in SUA:
        if than.count(tim) != 1:
            raise SystemExit(f"Chu can sua khong khop dung 1 lan: {tim}")
        than = than.replace(tim, thay)
    # thong nhat thuat ngu voi bao cao: "ca gold" -> "ca bac si"; ghi chu thuyet trinh doc bang tieng Viet
    than = than.replace("58 ca gold", "58 ca bác sĩ").replace("ca gold", "ca bác sĩ").replace("báo cáo gold", "báo cáo ca bác sĩ")

    def viet_hoa(m: re.Match) -> str:
        g = m.group(1).replace("chấm CV", "kiểm định chéo")
        g = re.sub(r"\bModel\b", "Mô hình", re.sub(r"\bmodel\b", "mô hình", g))
        return 'data-ghi-chu="' + re.sub(r"\btrain\b(?![._])", "huấn luyện", g) + '"'
    than = re.sub(r'data-ghi-chu="([^"]*)"', viet_hoa, than)
    trang_html = (dau.replace("</style>", CSS_THEM + "</style>", 1)
                  .replace("<title>Slide MRI gối 58 nhãn</title>", "<title>Slide PBL6 MRI gối</title>")
                  + than + "\n\n" + "\n".join(an) + cuoi)
    # nhan truc bieu do hanh trinh (trong JS): "DINOv2 dong bang" dinh vao nhan cot ke ben
    nhan_cu = 'ten: "DINOv2 đóng băng\\n58 ca, 1 mặt phẳng"'
    if trang_html.count(nhan_cu) != 1:
        raise SystemExit("Khong thay nhan truc DINOv2 trong bieu do hanh trinh")
    trang_html = trang_html.replace(nhan_cu, 'ten: "DINOv2\\n58 ca, 1 mặt phẳng"')
    # chu do JS dien (chan trang, dong phu slide ket luan) cung thong nhat thuat ngu
    trang_html = (trang_html.replace("58 ca gold", "58 ca bác sĩ").replace("tự gắn", "tự gán")
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
