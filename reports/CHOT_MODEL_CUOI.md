# Chốt model cuối (ghi TRƯỚC khi xem điểm lần train nhãn tự gắn và lần thêm seed)

Ngày chốt: 2026-11-07. Thời điểm commit file này là bằng chứng: lúc đó lần chạy
`rsna-knee-train-25d-nhan-minh` chưa xong, lần `rsna-knee-train-25d-seed-moi` chưa bắt đầu.

## Model báo cáo chính: gộp 6 CNN

| # | Lần chạy Kaggle | Backbone | Seed | Nhãn train |
|---|---|---|---:|---|
| 1 | `rsna-knee-train-25d` (đã có, 0.885 khi gộp 1+2) | EfficientNet-B3 | 42 | LLM công khai |
| 2 | `rsna-knee-train-25d` | EfficientNetV2-B2 | 7 | LLM công khai |
| 3 | `rsna-knee-train-25d-seed-moi` | EfficientNet-B3 | 123 | LLM công khai |
| 4 | `rsna-knee-train-25d-seed-moi` | EfficientNetV2-B2 | 2026 | LLM công khai |
| 5 | `rsna-knee-train-25d-nhan-minh` | EfficientNet-B3 | 42 | LLM tự gắn (Gemini Flash) |
| 6 | `rsna-knee-train-25d-nhan-minh` | EfficientNetV2-B2 | 7 | LLM tự gắn (Gemini Flash) |

- Mỗi model: trọng số EMA ở **epoch cuối (12)**, không chọn epoch theo gold.
- Gộp: với **từng bệnh**, xếp hạng 58 ca theo xác suất của mỗi model, lấy **trung bình
  hạng, trọng số bằng nhau**. Ca hoà hạng nhận hạng trung bình (`scipy.stats.rankdata`).
- Không loại model nào vì điểm thấp, không đổi trọng số theo bệnh. Model chỉ bị loại khi
  **hỏng kỹ thuật** (lần chạy lỗi, thiếu dự đoán cho ca gold); khi đó ghi rõ lý do.

## Số báo cáo kèm, cũng chốt từ giờ

- Gộp 4 CNN nhãn công khai (#1–#4).
- Gộp 2 CNN nhãn tự gắn (#5–#6), so bắt cặp với #1–#2 (cùng backbone, cùng seed, chỉ khác nhãn).
- Mốc cũ: gộp #1–#2 = **0.8848**.

Mọi số báo cáo nguyên như đo được, kể cả khi gộp 6 thấp hơn mốc cũ. Không lấy số lớn nhất
trong các phương án làm kết quả chính.

## Giới hạn phải nói

58 ca gold chưa bao giờ vào train, nhưng đã được **xem** nhiều lần trong quá trình phát triển
(đường cong theo epoch, so phương án, chọn CNN thay bản gộp với DINOv2). Vì vậy đây là
**đánh giá nội bộ có tái sử dụng tập gold**, không phải tập test độc lập hoàn toàn. Chốt
trước chỉ ngăn lựa chọn tiếp theo, không xoá được các lần xem trước đó.
