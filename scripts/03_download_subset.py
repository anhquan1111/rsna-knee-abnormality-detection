"""Tải các file .dcm theo files_manifest.csv về máy. Cần mạng.

Tải lại được: file đã có trên đĩa sẽ bỏ qua, nên ngắt giữa chừng rồi chạy lại vẫn tiếp tục đúng chỗ.

Chuẩn bị:
    uv pip install kaggle
    # kaggle.json để ở ~/.kaggle/kaggle.json (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)

Ví dụ:
    python scripts/03_download_subset.py --dry-run     # xem sẽ tải bao nhiêu, không tải thật
    python scripts/03_download_subset.py
    python scripts/03_download_subset.py --limit 50    # thử 50 file trước cho chắc
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Không thấy {path}. Chạy bước 02 trên Kaggle notebook trước.")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def authenticate():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("Thiếu thư viện kaggle. Chạy: uv pip install kaggle")
    api = KaggleApi()
    api.authenticate()
    return api


def unwrap_if_zipped(dest_dir: Path, file_name: str) -> None:
    """Kaggle có khi trả về .zip thay vì file gốc. Giải nén rồi xóa zip."""
    zipped = dest_dir / f"{file_name}.zip"
    if not zipped.exists():
        return
    with zipfile.ZipFile(zipped) as zf:
        zf.extractall(dest_dir)
    zipped.unlink()


def download_one(api, file_path: str, dest_root: Path) -> bool:
    """Trả về True nếu file đã sẵn sàng trên đĩa sau lời gọi này."""
    target = dest_root / file_path
    if target.exists() and target.stat().st_size > 0:
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.competition_download_file(
                COMPETITION,
                file_path,
                path=str(target.parent),
                force=False,
                quiet=True,
            )
            unwrap_if_zipped(target.parent, target.name)
            if target.exists() and target.stat().st_size > 0:
                return True
            print(f"  [!] Tải xong nhưng không thấy file: {file_path}")
            return False
        except Exception as exc:  # thư viện kaggle ném nhiều loại lỗi mạng khác nhau
            if attempt == MAX_RETRIES:
                print(f"  [x] Bỏ qua sau {MAX_RETRIES} lần thử: {file_path} ({exc})")
                return False
            time.sleep(RETRY_SLEEP_SECONDS * attempt)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "files_manifest.csv")
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Chỉ tải N file đầu (0 = tải hết)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in thống kê, không tải")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    total_bytes = sum(int(r.get("size_bytes", 0) or 0) for r in rows)
    already = [r for r in rows if (args.dest / r["file_path"]).exists()]

    print(f"Manifest        : {args.manifest}")
    print(f"Đích            : {args.dest}")
    print(f"File trong danh sách: {len(rows):,}")
    print(f"Đã có sẵn trên đĩa  : {len(already):,}")
    print(f"Cần tải             : {len(rows) - len(already):,}")
    print(f"Dung lượng ước tính : {total_bytes / 1024**3:.2f} GB")

    if args.dry_run:
        print("\n--dry-run: không tải gì.")
        return

    remaining = len(rows) - len(already)
    if remaining == 0:
        print("\nKhông còn gì để tải.")
        return

    api = authenticate()
    started = time.time()
    ok = failed = 0
    for i, row in enumerate(rows, start=1):
        if download_one(api, row["file_path"], args.dest):
            ok += 1
        else:
            failed += 1

        if i % 25 == 0 or i == len(rows):
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(rows) - i) / rate if rate > 0 else 0
            print(f"  {i:,}/{len(rows):,} | ok {ok:,} | lỗi {failed:,} "
                  f"| {rate:.1f} file/s | còn ~{eta/60:.0f} phút")

    print(f"\nXong: {ok:,} file ok, {failed:,} lỗi.")
    if failed:
        print("Chạy lại script để thử lại các file lỗi — file đã có sẽ được bỏ qua.")
        sys.exit(1)


if __name__ == "__main__":
    main()
