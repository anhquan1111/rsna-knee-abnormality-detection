"""Lấy danh sách file .dcm thật của các series đã chọn. CHẠY TRÊN KAGGLE NOTEBOOK.

Vì sao phải chạy trên Kaggle: train_series.csv chỉ cho SeriesInstanceUID, không cho
tên từng file .dcm. Kaggle API lại chỉ tải được từng file theo đúng tên. Trên notebook
thì dữ liệu đã mount sẵn ở /kaggle/input nên chỉ cần os.listdir là có danh sách ngay.

Cách dùng:
 1. Tạo notebook mới, Add Input -> chọn competition RSNA Knee Abnormality Detection.
 2. Upload data/manifest/subset_*.csv (Add Data -> Upload) hoặc dán thẳng danh sách
    SeriesInstanceUID vào biến FALLBACK_SERIES bên dưới.
 3. Dán toàn bộ file này vào một cell rồi chạy.
 4. Tải /kaggle/working/files_manifest.csv về máy, đặt vào data/manifest/.

Chỉ đọc tên file, không decode ảnh, nên chạy rất nhanh và không tốn GPU quota.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

COMPETITION_DIR = Path("/kaggle/input/rsna-knee-abnormality-detection")
SPLIT = "train_series"
OUTPUT_PATH = Path("/kaggle/working/files_manifest.csv")

# Đường dẫn manifest từ bước 01 sau khi upload lên notebook. Sửa cho khớp nơi Kaggle đặt file.
SUBSET_MANIFEST = Path("/kaggle/input/subset-manifest/subset_labeled_Sagittal.csv")

# Nếu không muốn upload file, dán thẳng các SeriesInstanceUID vào đây.
FALLBACK_SERIES: list[tuple[str, str]] = []  # [(StudyInstanceUID, SeriesInstanceUID), ...]

# Số slice giữ lại mỗi series. 0 = giữ hết.
# Lấy các slice ở giữa vì hai đầu chuỗi thường nằm ngoài vùng khớp cần quan tâm —
# đây là giả định cần tự kiểm lại bằng mắt trước khi tin.
MAX_SLICES_PER_SERIES = 0


def read_subset() -> list[tuple[str, str]]:
    if FALLBACK_SERIES:
        return FALLBACK_SERIES
    if not SUBSET_MANIFEST.exists():
        raise SystemExit(
            f"Không thấy {SUBSET_MANIFEST}. Upload manifest từ bước 01, "
            f"hoặc điền FALLBACK_SERIES."
        )
    with SUBSET_MANIFEST.open(newline="", encoding="utf-8") as f:
        return [(r["StudyInstanceUID"], r["SeriesInstanceUID"]) for r in csv.DictReader(f)]


def pick_middle(names: list[str], k: int) -> list[str]:
    """Giữ k slice ở giữa chuỗi, phân bố đều."""
    if k <= 0 or k >= len(names):
        return names
    start = (len(names) - k) // 2
    return names[start:start + k]


def main() -> None:
    pairs = read_subset()
    rows = []
    missing = []
    total_bytes = 0

    for study_uid, series_uid in pairs:
        series_dir = COMPETITION_DIR / SPLIT / study_uid / series_uid
        if not series_dir.is_dir():
            missing.append((study_uid, series_uid))
            continue

        # Sắp xếp theo tên để thứ tự tất định; KHÔNG coi đây là thứ tự giải phẫu của slice.
        # Thứ tự lát cắt thật nằm ở tag DICOM (ImagePositionPatient), phải đọc header mới biết.
        names = sorted(p.name for p in series_dir.iterdir() if p.suffix.lower() == ".dcm")
        names = pick_middle(names, MAX_SLICES_PER_SERIES)

        for name in names:
            path = series_dir / name
            size = path.stat().st_size
            total_bytes += size
            rows.append({
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "file_path": f"{SPLIT}/{study_uid}/{series_uid}/{name}",
                "size_bytes": size,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["StudyInstanceUID", "SeriesInstanceUID", "file_path", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã ghi {OUTPUT_PATH}")
    print(f"  Series yêu cầu : {len(pairs)}")
    print(f"  Series tìm thấy: {len(pairs) - len(missing)}")
    print(f"  File           : {len(rows):,}")
    print(f"  Dung lượng thật: {total_bytes / 1024**3:.2f} GB")
    if missing:
        print(f"  CẢNH BÁO: {len(missing)} series không thấy trong {SPLIT}/ — kiểm lại UID")
        for study_uid, series_uid in missing[:5]:
            print(f"    {study_uid}/{series_uid}")

    print("\nBước tiếp theo: tải files_manifest.csv về máy rồi chạy scripts/03_download_subset.py")


if __name__ == "__main__":
    main()
