"""NGAY 15 - Train lai CNN 2.5D tren Kaggle bang nhan LLM TU GAN (script 90).

    python scripts/92_day_train_nhan_minh.py      # up nhan len dataset rieng tu, day notebook

Giu NGUYEN notebook 81 (cung kien truc, cung 12 epoch, cung seed, cung chia train/val),
chi doi file nhan: RSNA_NHAN=llm_labels_cua_minh.csv. Nhu vay chenh lech diem gold giua
hai lan chay chi den tu nguon nhan.

Kaggle khong dat duoc bien moi truong cho notebook, nen script tao mot ban chep cua 81 co
them dong `os.environ.setdefault("RSNA_NHAN", ...)` ngay sau `from __future__`, ghi vao
kaggle/83_train_nhan_minh/ (file sinh ra, khong commit) roi day len.
Dataset nhan de RIENG TU: nhan suy tu bao cao cua cuoc thi, khong phat tan lai.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(REPO / ".venv" / "bin" / "kaggle")
NHAN = REPO / "data" / "nhan_llm_cua_minh" / "llm_labels_cua_minh.csv"
DS_ID = "votrananhquan/rsna-knee-nhan-llm-cua-minh"
KERNEL = REPO / "kaggle" / "83_train_nhan_minh"
TEN_FILE = "train_nhan_minh.py"


def chay(*lenh: str) -> str:
    r = subprocess.run([KAGGLE, *lenh], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.stdout + r.stderr


def main() -> None:
    if not NHAN.exists():
        raise SystemExit(f"Chua co {NHAN} - chay scripts/90_gan_nhan_llm.py truoc")
    ds = NHAN.parent / "kaggle_ds"
    ds.mkdir(exist_ok=True)
    shutil.copy(NHAN, ds / NHAN.name)
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "rsna-knee-nhan-llm-cua-minh", "id": DS_ID, "licenses": [{"name": "CC0-1.0"}]}))
    da_co = DS_ID.split("/")[1] in chay("datasets", "list", "--mine", "-s", "rsna-knee-nhan-llm-cua-minh")
    if da_co:
        chay("datasets", "version", "-p", str(ds), "-m", "nhan moi")
    else:
        chay("datasets", "create", "-p", str(ds))          # mac dinh rieng tu

    goc = (REPO / "scripts" / "81_kaggle_train_25d.py").read_text()
    moc = "from __future__ import annotations\n"
    assert goc.count(moc) == 1
    ban = goc.replace(moc, moc + '\nimport os as _os\n_os.environ.setdefault("RSNA_NHAN", "llm_labels_cua_minh.csv")\n', 1)
    (KERNEL / TEN_FILE).write_text(ban)
    chay("kernels", "push", "-p", str(KERNEL))


if __name__ == "__main__":
    main()
