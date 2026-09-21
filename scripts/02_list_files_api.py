"""Liet ke toan bo file cua competition qua Kaggle API, khong can chay tren Kaggle notebook.

Ly do ton tai: `train_series.csv` chi cho SeriesInstanceUID (ten thu muc), khong cho ten
tung file .dcm. Ma `kaggle competitions download -f` bat buoc phai biet ten file chinh xac.
API `competition_list_files` tra ve toan bo cay file nhung bi chan cung o 200 file/trang,
nen phai phan trang va ghi checkpoint de chay lai duoc khi dut mang.

Chay:
    python scripts/02_list_files_api.py                 # chay tiep tu checkpoint
    python scripts/02_list_files_api.py --max-pages 50  # thu truoc cho nhanh
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_CSV = REPO / "data" / "manifest" / "competition_files.csv"
STATE_JSON = REPO / "data" / "manifest" / "competition_files.state.json"
COMPETITION = "rsna-knee-abnormality-detection"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=0, help="0 = khong gioi han")
    ap.add_argument("--restart", action="store_true", help="bo checkpoint, liet ke lai tu dau")
    args = ap.parse_args()

    import kaggle  # import muon: can KAGGLE_CONFIG_DIR/kaggle.json san sang

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    token, seen = None, 0
    if args.restart:
        OUT_CSV.unlink(missing_ok=True)
        STATE_JSON.unlink(missing_ok=True)
    elif STATE_JSON.exists():
        state = json.loads(STATE_JSON.read_text())
        token, seen = state.get("next_page_token"), state.get("seen", 0)
        if token is None:
            print(f"Da liet ke xong tu truoc: {seen:,} file -> {OUT_CSV}")
            return

    fresh = not OUT_CSV.exists()
    fh = OUT_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if fresh:
        writer.writerow(["name", "size_bytes"])

    page, t0 = 0, time.time()
    try:
        while True:
            # Mang chap chon la chuyen binh thuong o day: quet het competition nay can hon
            # 4,000 request lien tiep. Lui dan toi 60s va thu 12 lan - re hon nhieu so voi
            # viec chet giua chung roi phai goi lai tu checkpoint.
            for attempt in range(12):
                try:
                    resp = kaggle.api.competition_list_files(
                        COMPETITION, page_size=200, page_token=token
                    )
                    break
                except Exception as exc:  # 429/503/timeout -> lui dan roi thu lai
                    wait = min(2 ** attempt, 60)
                    print(f"  loi {type(exc).__name__} (lan {attempt+1}/12) -> cho {wait}s",
                          flush=True)
                    time.sleep(wait)
            else:
                raise RuntimeError("that bai 12 lan lien tiep, dung lai; chay lai de tiep tuc")

            for f in resp.files:
                writer.writerow([f.name, f.total_bytes])
            seen += len(resp.files)
            token = resp.next_page_token or None
            page += 1

            fh.flush()
            STATE_JSON.write_text(json.dumps({"next_page_token": token, "seen": seen}))

            if page % 25 == 0:
                rate = seen / max(time.time() - t0, 1e-9)
                print(f"  trang {page:>5} | {seen:>9,} file | {rate:,.0f} file/s")
            if token is None:
                print(f"XONG: {seen:,} file -> {OUT_CSV}")
                break
            if args.max_pages and page >= args.max_pages:
                print(f"Dung theo --max-pages={args.max_pages}. Da co {seen:,} file, chay lai de tiep.")
                break
    finally:
        fh.close()


if __name__ == "__main__":
    main()
