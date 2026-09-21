"""Chọn tập con study/series sẽ tải về. Chạy hoàn toàn offline, không cần mạng.

Đọc data/raw/train.csv + train_series.csv, xuất manifest cho bước 02.

Ví dụ:
    python scripts/01_select_subset.py                      # 58 study có nhãn, 1 series sagittal (~1 GB)
    python scripts/01_select_subset.py --planes Sagittal Coronal Axial
    python scripts/01_select_subset.py --strategy n-studies --n 500
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"

# Hiệu chuẩn từ số liệu thật của cuộc thi: 569.76 GB / 819,640 file / ~31,560 series.
# Không phải con số ban tổ chức công bố cho từng series, chỉ là trung bình suy ra.
MB_PER_SERIES = 18.5
FILES_PER_SERIES = 26.0

LABEL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has_full_labels(row: dict[str, str]) -> bool:
    return all(row.get(col, "").strip() != "" for col in LABEL_COLUMNS)


def pick_series_for_study(
    series_rows: list[dict[str, str]],
    planes: list[str],
    prefer_fluid: bool,
) -> list[dict[str, str]]:
    """Với mỗi mặt phẳng yêu cầu, chọn đúng một series của study này.

    Khi một study có nhiều series cùng mặt phẳng, đây là một quyết định mô hình
    thật sự (chuỗi xung nhạy dịch làm nổi tràn dịch/viêm, chuỗi thường thì không).
    Chọn có chủ đích ở đây, đừng để nó thành ngẫu nhiên.
    """
    by_plane: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in series_rows:
        by_plane[row["Anatomical_Plane"]].append(row)

    picked = []
    for plane in planes:
        candidates = by_plane.get(plane, [])
        if not candidates:
            continue
        # Sắp xếp tất định để chạy lại cho ra đúng cùng một tập con.
        candidates.sort(key=lambda r: (
            -int(r.get("Fluid_Sensitive", "0") or 0) if prefer_fluid else 0,
            r["SeriesInstanceUID"],
        ))
        picked.append(candidates[0])
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=["labeled", "n-studies"], default="labeled",
                        help="labeled: chỉ 58 study có nhãn người gán. n-studies: 58 study đó + N-58 study khác")
    parser.add_argument("--n", type=int, default=500, help="Tổng số study khi dùng strategy n-studies")
    parser.add_argument("--planes", nargs="+", default=["Sagittal"],
                        choices=["Sagittal", "Coronal", "Axial"],
                        help="Lấy một series cho mỗi mặt phẳng liệt kê ở đây")
    parser.add_argument("--prefer-fluid", action="store_true",
                        help="Ưu tiên series Fluid_Sensitive=1 khi một mặt phẳng có nhiều series")
    parser.add_argument("--max-slices", type=int, default=0,
                        help="Số slice giữ lại mỗi series ở bước 02 (0 = giữ hết). Chỉ dùng để ước lượng dung lượng")
    parser.add_argument("--out", type=Path, default=None, help="Đường dẫn manifest xuất ra")
    args = parser.parse_args()

    train = load_csv(RAW_DIR / "train.csv")
    series = load_csv(RAW_DIR / "train_series.csv")

    labeled_uids = {r["StudyInstanceUID"] for r in train if has_full_labels(r)}
    all_uids = [r["StudyInstanceUID"] for r in train]

    if args.strategy == "labeled":
        selected_uids = list(labeled_uids)
    else:
        extra = [u for u in all_uids if u not in labeled_uids]
        selected_uids = list(labeled_uids) + extra[: max(0, args.n - len(labeled_uids))]

    selected_uids.sort()
    selected_set = set(selected_uids)

    series_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in series:
        if row["StudyInstanceUID"] in selected_set:
            series_by_study[row["StudyInstanceUID"]].append(row)

    rows_out = []
    studies_without_plane = 0
    for uid in selected_uids:
        picked = pick_series_for_study(series_by_study.get(uid, []), args.planes, args.prefer_fluid)
        if not picked:
            studies_without_plane += 1
            continue
        for row in picked:
            rows_out.append({
                "StudyInstanceUID": uid,
                "SeriesInstanceUID": row["SeriesInstanceUID"],
                "Anatomical_Plane": row["Anatomical_Plane"],
                "Fluid_Sensitive": row.get("Fluid_Sensitive", ""),
                "has_labels": "1" if uid in labeled_uids else "0",
            })

    out_path = args.out or MANIFEST_DIR / f"subset_{args.strategy}_{'_'.join(args.planes)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    n_series = len(rows_out)
    slice_ratio = 1.0 if args.max_slices <= 0 else min(1.0, args.max_slices / FILES_PER_SERIES)
    est_files = n_series * FILES_PER_SERIES * slice_ratio
    est_gb = n_series * MB_PER_SERIES * slice_ratio / 1024

    print(f"Manifest: {out_path}")
    print(f"  Study chọn được       : {len(selected_uids)} (trong đó {len(labeled_uids)} có nhãn người gán)")
    print(f"  Series                : {n_series}")
    if studies_without_plane:
        print(f"  Study bị bỏ (thiếu mặt phẳng yêu cầu): {studies_without_plane}")
    print(f"  Ước tính file         : ~{est_files:,.0f}")
    print(f"  Ước tính dung lượng   : ~{est_gb:.1f} GB"
          + (f" (giữ {args.max_slices} slice/series)" if args.max_slices > 0 else " (giữ hết slice)"))
    print("\nBước tiếp theo: chạy scripts/02_list_files_kaggle.py TRÊN Kaggle notebook.")


if __name__ == "__main__":
    main()
