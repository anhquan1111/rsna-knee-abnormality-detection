# Trình bày PBL: Phát hiện 12 bất thường đầu gối từ MRI

Võ Trần Anh Quân · Dữ liệu: cuộc thi *RSNA Knee Abnormality Detection* (Kaggle)

---

## 1. Bài toán (30 giây)

| | |
|---|---|
| **Đầu vào** | Một ca MRI đầu gối: 3 mặt phẳng (Sagittal, Coronal, Axial), mỗi series khoảng 26 lát DICOM. Cả bộ 4.407 ca, 569 GB. |
| **Đầu ra** | 12 xác suất: ACL, MCL, sụn chêm trong/ngoài, thoái hoá 3 khoang, tràn dịch, viêm màng hoạt dịch, nang Baker, dập xương, gãy xương. |
| **Thước đo** | Macro AUC trên 12 nhãn. |
| **Cái khó** | Chỉ **58 / 4.407 ca (1,3%) có nhãn bác sĩ**. 4.349 ca còn lại chỉ có báo cáo chữ tự do, viết bằng 9 ngôn ngữ. |

Vì vậy dự án gồm hai bài toán: **(1)** rút nhãn từ báo cáo chữ để dùng được 98,7% dữ liệu, **(2)** học từ ảnh MRI.
58 ca bác sĩ gán nhãn **không bao giờ dùng để train**, chỉ dùng để chấm.

---

## 2. Quy trình

```
 Báo cáo chữ (9 ngôn ngữ) ──► LLM đọc báo cáo ──► nhãn mềm cho 4.349 ca
                                                  (khớp nhãn bác sĩ: AUC 0.893)
                                                          │
 DICOM 3 mặt phẳng ──► chọn series nhạy dịch ──► chuẩn hoá theo cả series
                       (96,9% ca)                 (percentile 1–99) ──► 24 lát × 256 px
                                                          │
                     ┌────────────────────────────────────┴───────────────────────┐
     GIAI ĐOẠN 1 (máy cá nhân)                           GIAI ĐOẠN 2 (Kaggle GPU T4 ×2)
     DINOv2 đóng băng → đặc trưng từng lát                CNN 2.5D train toàn bộ mạng
     → 5 head attention riêng từng bệnh                   cửa sổ 3 lát liền kề = 3 kênh ảnh
     → gộp theo hạng                                      → attention riêng từng bệnh (MIL)
     = 0.7645                                             → 2 model, gộp theo hạng
                                                          = 0.8848
                     └────────────────────────────────────┬───────────────────────┘
                                                          ▼
                              Chấm trên 58 ca bác sĩ + khoảng tin cậy bootstrap 95%
```

**Nguyên tắc đánh giá (áp cho mọi con số ở dưới):** số epoch, cách gộp và model cuối đều **chốt trước khi xem điểm 58 ca gold**. Đây là bài học rút ra ở giai đoạn 1 (mục 4).

---

## 3. Kết quả

### Hành trình điểm số

| Bước | Macro AUC |
|---|---:|
| Đoán mò | 0.500 |
| DINOv2 đóng băng, chỉ 58 ca, 1 mặt phẳng | 0.526 |
| + dùng cả 3 mặt phẳng | 0.627 |
| + nhãn máy từ từ điển luật (dùng được 4.349 ca) | 0.692 |
| + nhãn LLM thay từ điển | 0.747 |
| + attention riêng từng bệnh | 0.770 |
| Gộp 5 head × 3 seed, **giao thức đã sửa** (kết quả thật của giai đoạn 1) | **0.7645** `[0.715, 0.805]` |
| **CNN 2.5D train toàn bộ mạng, gộp 2 model** (giai đoạn 2) | **0.8848** `[0.851, 0.915]` |

*Các dòng giữa đo theo giao thức cũ, chỉ dùng để so các bước với nhau. Số trong ngoặc là khoảng tin cậy 95%.*

### Giai đoạn 2 so với giai đoạn 1

| Model | Macro AUC trên 58 ca gold | KTC 95% |
|---|---:|---|
| EfficientNet-B3, seed 42 (12 epoch, 2,3 giờ) | 0.8741 | [0.838, 0.907] |
| EfficientNetV2-B2, seed 7 (12 epoch, 1,3 giờ) | 0.8791 | [0.846, 0.909] |
| **CNN: gộp 2 model theo hạng** | **0.8848** | **[0.851, 0.915]** |
| DINOv2 đóng băng (giai đoạn 1) | 0.7645 | [0.717, 0.806] |
| Gộp 50/50 CNN + DINOv2 | 0.8519 | [0.815, 0.883] |

- **CNN hơn DINOv2 +0.120**, KTC bắt cặp `[+0.084, +0.161]`, không chứa 0, tức chênh lệch có ý nghĩa dù chỉ có 58 ca.
- Gộp thêm DINOv2 lại **kéo điểm xuống** (0.852 < 0.885): DINOv2 yếu hơn quá xa nên chỉ thêm nhiễu. Phép gộp này đã chốt trước nên vẫn báo cáo nguyên.

### Từng bệnh (CNN | DINOv2)

| Bệnh | CNN | DINOv2 | | Bệnh | CNN | DINOv2 |
|---|---:|---:|---|---|---:|---:|
| Thoái hoá khoang trong | 0.978 | 0.896 | | Thoái hoá khoang ngoài | 0.873 | 0.743 |
| Nang Baker | 0.968 | 0.899 | | Sụn chêm ngoài | 0.870 | 0.777 |
| **ACL** | **0.966** | **0.683** | | Tràn dịch | 0.860 | 0.868 |
| **Dập xương** | **0.949** | **0.709** | | Thoái hoá bánh chè–đùi | 0.853 | 0.786 |
| Sụn chêm trong | 0.922 | 0.778 | | Gãy xương | 0.792 | 0.672 |
| **MCL** | **0.878** | **0.662** | | Viêm màng hoạt dịch | 0.710 | 0.701 |

Tăng mạnh nhất ở **tổn thương nhỏ, khu trú** (ACL +0.28, dập xương +0.24, MCL +0.22). Đây là chỗ đặc trưng đóng băng của DINOv2 (học từ ảnh thường) không nhìn ra, còn CNN train trên chính MRI thì học được.

---

## 4. Những phát hiện quan trọng (tìm ra nhờ đo, không nhờ đoán)

1. **Điểm 0.8112 ban đầu là ảo.** Lúc đo CV, epoch tốt nhất được chọn ngay trên fold đang chấm. Còn model cuối thì âm thầm train tới khoảng epoch 80: nhãn mềm làm AUC thành `nan`, nên code lùi về chọn theo loss mà không báo. Sửa bằng cách chốt 9 epoch từ trước, chọn bằng một tập nhãn LLM giữ riêng, không nhìn gold. Điểm thật là **0.7645**.
2. **Chọn epoch không cần nhãn bác sĩ.** Ở giai đoạn 2, AUC trên tập val nhãn LLM (212 ca) đi cùng chiều với AUC gold qua từng epoch (đỉnh ở epoch 7–9 cả hai bên). Model cuối lấy epoch 12 như đã chốt, chỉ thấp hơn đỉnh gold khoảng 0.01.
3. **Hai đường rò rỉ dữ liệu.** Thứ nhất, chia tập theo series làm 100% ca validation có series nằm bên train, nên phải chia theo ca. Thứ hai, 1 ca ngoài gold có báo cáo trùng từng ký tự với một ca gold, nên nhãn máy của nó mang luôn đáp án. Chỉ 1 ca / 4.349, nhưng bịt nó lại thì kết luận của một thí nghiệm đổi hẳn. Ca này bị loại khỏi train ở cả hai giai đoạn.
4. **Lỗi kỹ thuật trên Kaggle.** Hai lần chạy đầu bị *Killed* với tốc độ 10,4 giây/bước. Thay vì đoán, tôi viết một kernel chẩn đoán riêng: `nn.DataParallel` trên 2 GPU T4 **chậm hơn 1 GPU khoảng 20 lần** (10,7 so với 0,47 giây/bước). Cách sửa: mỗi GPU train một model độc lập, chạy song song. Lần chạy thứ ba xong trong 3,1 giờ.
5. **Kiểm thử nhắm vào lỗi âm thầm**, loại lỗi cho ra một con số đẹp thay vì báo lỗi: 114 test, ví dụ pooling quên mask, softmax ra `NaN` khi ca thiếu mặt phẳng, cửa sổ 3 lát không liền kề.

---

## 4b. Nhãn LLM: tự gắn lại để kiểm chứng

Nhãn LLM dùng để train là bộ công khai, không rõ LLM và prompt. Em cho **Gemini Flash** đọc lại cả 4.407 báo cáo bằng prompt tự viết (chốt trước, LLM chỉ thấy văn bản), rồi so với bác sĩ trên 58 ca:

| Nguồn nhãn | AUC so với bác sĩ | Khớp bác sĩ ở ngưỡng 0.5 |
|---|---:|---:|
| LLM công khai | 0.893 | 81.8% |
| **LLM tự gắn (Gemini Flash)** | **0.857** [0.823, 0.889] | 81.2% |
| Từ điển tự viết | 0.731 | 44.2% |

- Không nguồn nào đúng 100%: cả hai bộ LLM khớp bác sĩ khoảng 81% số ô.
- Nhãn tự gắn kém bộ công khai 0.036 (kém thật, KTC không chứa 0), chủ yếu ở viêm màng hoạt dịch và dập xương; hơn từ điển 0.126.
- Hai bộ LLM khớp nhau 96,9% trên 4.349 ca không có nhãn bác sĩ.
- Đang train lại CNN bằng nhãn tự gắn để đo ảnh hưởng lên model cuối.

## 5. Giới hạn (nói trước khi thầy hỏi)

- **Chỉ 58 ca để chấm**, nên khoảng tin cậy rộng khoảng ±0.03. Chênh lệch nhỏ cỡ 0.01 giữa hai model CNN **chưa** kết luận được.
- Mỗi model CNN mới train **1 lần (1 seed)**. Ở giai đoạn 1, đổi seed làm điểm dao động với độ lệch chuẩn khoảng 0.01.
- Chọn CNN làm kết quả chính thay vì bản gộp 50/50 là chọn 1 trong 2 sau khi xem điểm. Chênh 0.033 nên kết luận không đổi, nhưng ghi rõ ra đây.
- Nhãn train là **nhãn LLM** (khớp nhãn bác sĩ ở mức AUC 0.893), nên model bị giới hạn bởi chất lượng nhãn đó.
- Viêm màng hoạt dịch (0.71) và gãy xương (0.79) vẫn còn yếu.

---

## 6. Câu lệnh chạy mẫu

**Cài đặt và kiểm thử** (Ubuntu/WSL, Python 3.12, `uv`):

```bash
uv venv --python 3.12 .venv
uv sync --extra data --extra dev --extra bao-cao --extra cnn
uv run pytest -q                                   # 114 test, khoảng 20 giây
```

**Giai đoạn 1: DINOv2 đóng băng** (chạy ở máy; phần trích đặc trưng chạy trên Kaggle):

```bash
uv run python scripts/10_build_manifest.py        # bảng ca bệnh / series
uv run python scripts/11_build_splits.py          # chia fold theo ca, chống rò rỉ
uv run python scripts/20_extract_weak_labels.py   # nhãn máy từ báo cáo chữ
uv run python scripts/45_train_final_ensemble.py  # train 5 head, 9 epoch, khoảng 6 phút
uv run python scripts/48_do_theo_epoch.py --cham  # chấm bộ gộp 5 head × 3 seed -> 0.7645
```

**Giai đoạn 2: CNN 2.5D** (train trên Kaggle GPU, chấm ở máy):

```bash
uv run kaggle kernels push -p kaggle/81_train                   # đẩy notebook train lên Kaggle
uv run kaggle kernels status votrananhquan/rsna-knee-train-25d  # khoảng 3 giờ thì COMPLETE
uv run python scripts/82_cham_cnn_25d.py --tai                  # tải kết quả, chấm 58 ca gold -> 0.8848
```

**Xuất báo cáo và slide:**

```bash
uv run python scripts/70_xuat_bao_cao.py          # -> reports/bao_cao_pbl.html
uv run python scripts/70_xuat_bao_cao.py --slide  # -> reports/slide/ (PNG, PDF, PPTX; cần Chrome/Edge)
```

**Nhãn LLM tự gắn và web demo:**

```bash
uv run python scripts/90_gan_nhan_llm.py          # Gemini Flash gắn nhãn 4.407 báo cáo (chạy tiếp được)
uv run python scripts/91_so_sanh_nhan.py          # so với bác sĩ, nhãn công khai, từ điển trên 58 ca
uv sync --extra demo
uv run python scripts/95_web_demo.py              # mở http://127.0.0.1:8790  (thêm ?mau=mau7 để tự chạy ca mẫu)
```

Chi tiết từng bước nằm trong `README.vi.md`. Kết quả số của giai đoạn 2 ở `reports/day14_cnn_25d.json`.
