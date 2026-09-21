# RSNA Knee Abnormality Detection — PBL

Dự đoán 12 bất thường đầu gối từ MRI đa mặt phẳng. [Trang cuộc thi](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · hạn 2026-10-22.

Tài liệu học và phân tích: `D:\Documents\AI\Update\Hoc_77_Ngay\070_Computer_Vision\071_Data_Preprocessing_Evaluation\`

## Bài toán

- **Input:** một study MRI (trung bình 5.53 series: sagittal/coronal/axial, mỗi series ~26 slice DICOM).
- **Output:** 12 xác suất cho mỗi study — `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
- **Metric:** macro-averaged AUC-ROC trên 12 nhãn.
- **Multi-label:** các nhãn không loại trừ nhau → 12 logit + `BCEWithLogitsLoss`, không dùng softmax.
- **Chỉ 58/4,407 study (1.32%) có nhãn người gán.** 4,349 study còn lại chỉ có cột `Report` — báo cáo văn bản tự do, 9 ngôn ngữ. Bài toán vì vậy có hai nửa: **rút nhãn từ text (NLP)** + **học từ ảnh (CV)**.

## Cấu trúc code

```text
src/rsna_knee/
├── config.py          # 12 nhãn (thứ tự cố định), đường dẫn, seed, short_uid cho MAX_PATH
├── manifest.py        # dựng manifest cấp study từ CSV; cột provenance
├── splits.py          # StratifiedGroupKFold theo study + gộp báo cáo trùng; đo rò rỉ
├── metrics.py         # macro AUC tự viết — luôn trả kèm mẫu số và nhãn bị bỏ
├── dicom_io.py        # đọc DICOM, chuẩn hoá percentile theo series, resize
├── dataset.py         # StudyDataset (1 phần tử = 1 study) + FeatureDataset
├── features.py        # backbone đóng băng → feature từng slice
├── head.py            # pooling slice→study (mean/max/attn) + head 12 chiều + masked BCE
└── reports/
    ├── lexicon.py     # từ điển 9 ngôn ngữ: concept / finding / phủ định / khẳng định bình thường
    ├── extract.py     # bộ rút nhãn 3 trạng thái 1/0/unknown
    └── evaluate.py    # chấm trên gold set, per-label, theo ngôn ngữ
```

## Quy trình chạy

```bash
uv venv --python 3.12 .venv
uv sync --extra data --extra dev     # cài đúng phiên bản ghim trong uv.lock
uv run pytest                        # 60 test, ~15 giây, phải xanh hết
```

Phiên bản được **ghim trong `uv.lock`**, không cài tự do. Lý do: nhiều con số trong tài liệu gắn chặt với hành vi của đúng phiên bản đó — ví dụ `roc_auc_score` trả `nan` thay vì ném lỗi là hành vi của `scikit-learn 1.9.x`, và cả phần đánh giá của ngày 4 dựa trên điều đó.

| Bước | Lệnh | Cần mạng | Đầu ra |
|---|---|:---:|---|
| Liệt kê file trên Kaggle | `python scripts/02_list_files_api.py` | ✅ | `data/manifest/competition_files.csv` |
| Tải ảnh theo listing | `python scripts/04_download_from_listing.py --gold --plane Sagittal` | ✅ | `data/raw/images/` |
| Kiểm toàn vẹn ảnh | `python scripts/05_verify_images.py` | — | báo cáo file hỏng/thiếu |
| **Ngày 3–4** manifest | `python scripts/10_build_manifest.py` | — | `data/interim/study_manifest.csv` |
| **Ngày 4** split + metric | `python scripts/11_build_splits.py` | — | `data/interim/splits_gold.csv` |
| **Ngày 3** soi DICOM thật | `python scripts/30_inspect_dicom.py` | — | số đo tiền xử lý |
| **Ngày 6–7** nhãn yếu | `python scripts/20_extract_weak_labels.py` | — | `data/interim/weak_labels.csv`, `reports/weak_label_scores.csv` |
| **Trích đặc trưng trên Kaggle** | dán `scripts/06_kaggle_extract_features.py` vào một Kaggle notebook | — | `features_dinov2_*.npz`, `dicom_headers.csv` |
| **Ngày 5** frozen backbone | `python scripts/31_train_frozen_head.py` | — | `reports/day5_*.csv`, checkpoint |
| **Ngày 8** thí nghiệm nhãn máy | `python scripts/40_experiment_weak_labels.py` | — | `reports/day8_experiment.csv` |

Ba file CSV gốc đã đủ cho ngày 1, 4, 6 và 7 — không cần tải ảnh.

## Cách lấy ảnh

`train_series.csv` chỉ cho `SeriesInstanceUID` (tên thư mục), không cho tên từng file `.dcm`, mà Kaggle API chỉ tải được file theo đúng tên. `scripts/02_list_files_api.py` giải quyết bằng cách phân trang qua `competition_list_files` (chặn cứng 200 file/trang, có checkpoint để chạy lại được) — **không còn cần bước chạy trên Kaggle notebook** như `02_list_files_kaggle.py` trước đây.

**Trên Windows, đường dẫn gốc của Kaggle vượt giới hạn MAX_PATH 260 ký tự** (`train_series/<64 ký tự>/<64 ký tự>/<64 ký tự>.dcm`). Lỗi báo ra là `FileNotFoundError`, rất dễ chẩn đoán nhầm thành lỗi mạng. Vì vậy thư mục cục bộ dùng 12 ký tự cuối của UID; ánh xạ đầy đủ ở `data/manifest/local_images.csv`.

| Giai đoạn | Tải gì | Dung lượng | Tốc độ đo được |
|---|---|---:|---|
| 1 | 58 study có nhãn, 1 series sagittal | ~1.0 GB | ~4 s/file, ~525 kB/file |
| 2 | 58 study có nhãn, đủ series | ~6.1 GB | — |
| 3 | 500 study (kèm nhãn rút từ report) | ~50 GB | — |
| — | Toàn bộ train, đủ series — **không khả thi** | ~440 GB | — |

> ⚠️ **Kaggle giới hạn tốc độ API.** Tải ~1.500 file `.dcm` cộng với hơn 4.000 request liệt kê trong một buổi làm tài khoản bị chặn: HTTP `429 RESOURCE_EXHAUSTED`, header `retry-after: 179280` giây ≈ **50 giờ**. Lệnh liệt kê vẫn chạy được, chỉ lệnh tải file bị chặn.
>
> **Vì vậy đường đi chính thức là Kaggle notebook, không phải tải về máy:** dataset đã mount sẵn ở `/kaggle/input` nên không tốn request API nào, lại có GPU T4 miễn phí. Dán `scripts/06_kaggle_extract_features.py` vào một notebook, chạy, rồi chỉ tải kết quả về (vài trăm MB thay vì 569 GB). Tải `.dcm` về máy chỉ nên dùng cho một tập con nhỏ để soi dữ liệu thật.

## Kiểm thử

`uv run pytest` — 60 test, không cần dữ liệu ảnh.

Test ở đây **không kiểm "hàm có chạy không"** mà kiểm đúng những hành vi hỏng âm thầm — loại lỗi cho ra một con số đẹp thay vì một dòng lỗi:

| File | Chặn điều gì |
|---|---|
| `test_metrics.py` | macro AUC trả `nan`, nhãn một lớp bị bỏ khỏi mẫu số mà không báo, baseline hằng số phải đúng 0.5 |
| `test_head.py` | pooling quên mask, `max` pad bằng 0 thay vì `-inf`, loss không bỏ qua nhãn `NaN` |
| `test_splits.py` | rò rỉ giữa các fold, báo cáo trùng bị tách hai bên, split không tái lập được |
| `test_reports.py` | phủ định lật nhầm vế sau của câu, `unknown` bị ép thành âm tính, recall đẹp giả tạo |
| `test_dicom_io.py` | chuẩn hoá ra `inf`/`nan`, chuẩn hoá từng lát xoá mất độ sáng tương đối |

Bộ test bắt được **hai lỗi thật ngay lần chạy đầu**:

- `normalize_series` trả `float64` thay vì `float32` như docstring — scalar `float64` của `np.percentile` nâng kiểu cả mảng. Cache to gấp đôi mà hàm vẫn chạy đúng nên không ai thấy.
- Từ điển tiếng Thổ bỏ lọt biến âm `k → ğ` (`yırtık → yırtığı`), tức bỏ lọt một phần nhóm **546 báo cáo tiếng Thổ** (12,4% dataset).

## Quyết định đã chốt

- **Đơn vị dự đoán và đơn vị split:** `StudyInstanceUID`. Chia ngẫu nhiên ở cấp series làm rò rỉ **100%** study của tập validation (đo thật trên 336 series của 58 study gold).
- **Chuẩn hoá cường độ:** percentile 1–99 trên **cả series**. MRI không có đơn vị tuyệt đối nên chia `/255` giữ nguyên bias máy chụp; chuẩn hoá từng lát xoá mất độ sáng tương đối giữa các lát.
- **Metric:** macro AUC tự viết, luôn báo kèm mẫu số. `roc_auc_score` trả `nan` chứ không ném lỗi khi nhãn chỉ có một lớp — `np.nanmean` sẽ âm thầm bỏ nhãn đó.
- **Không báo cáo accuracy:** baseline hằng số đạt accuracy 67.2% trong khi AUC đúng bằng 0.500.
- **Nhãn máy giữ 3 trạng thái** `1/0/NaN`; loss bỏ qua NaN. Ép NaN→0 làm macro F1 tụt 0.800 → 0.614.

## Môi trường

Python 3.12 + uv. Đọc DICOM cần cả `pydicom` lẫn bộ giải nén (`pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`, `gdcm`) — dataset dùng nhiều transfer syntax, trong đó có JPEG Lossless và JPEG 2000.
