"""NGAY 15 - Train lai CNN 2.5D tren Kaggle bang nhan LLM khac (giu nguyen notebook 81).

    python scripts/92_day_train_nhan_minh.py                       # nhan tu gan (script 90)
    python scripts/92_day_train_nhan_minh.py --nhan cong_khai --mo-hinh "efficientnet_b3:123,tf_efficientnetv2_b2:2026"

Giu NGUYEN notebook 81 (cung kien truc, cung 12 epoch, cung chia train/val), chi doi file
nhan (va tuy chon backbone:seed). Nhu vay chenh lech diem gold giua cac lan chay den tu
nguon nhan, khong tu thu khac.

Kaggle khong dat duoc bien moi truong cho notebook, nen script tao mot ban chep cua 81 co
them dong `os.environ.setdefault(...)` ngay sau `from __future__`, ghi vao kaggle/<thu muc>/
(file sinh ra, khong commit) roi day len. Dataset nhan de RIENG TU: nhan suy tu bao cao cua
cuoc thi, khong phat tan lai.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(REPO / ".venv" / "bin" / "kaggle")
THU_MUC_NHAN = REPO / "data" / "nhan_llm_cua_minh"
DS_ID = "votrananhquan/rsna-knee-nhan-llm-cua-minh"
LAN_CHAY = {   # --nhan -> (file nhan trong dataset, thu muc kernel)
    "cua_minh": ("llm_labels_cua_minh.csv", REPO / "kaggle" / "83_train_nhan_minh"),
    # nhan cong khai, seed moi: dataset rsna-knee-my-assets da gan san trong metadata, khong up gi
    "cong_khai": ("llm_labels_v4_blend.csv", REPO / "kaggle" / "85_train_seed_moi"),
}
TEN_FILE = "train.py"


def chay(*lenh: str) -> str:
    r = subprocess.run([KAGGLE, *lenh], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.stdout + r.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhan", choices=list(LAN_CHAY), default="cua_minh")
    ap.add_argument("--mo-hinh", default="", help="backbone:seed,... (mac dinh: nhu notebook 81)")
    a = ap.parse_args()
    if a.nhan == "cong_khai":
        ten_nhan, thu_muc = LAN_CHAY[a.nhan]
        day_notebook(ten_nhan, thu_muc, a.mo_hinh)
        return
    if not (THU_MUC_NHAN / "llm_labels_cua_minh.csv").exists():
        raise SystemExit("Chua co nhan tu gan - chay scripts/90_gan_nhan_llm.py truoc")
    ten_nhan, thu_muc = LAN_CHAY[a.nhan]

    # Dataset rieng tu chua MOI file nhan da tao; da co thi them phien ban moi
    ds = THU_MUC_NHAN / "kaggle_ds"
    ds.mkdir(exist_ok=True)
    for f in THU_MUC_NHAN.glob("llm_labels_*.csv"):
        shutil.copy(f, ds / f.name)
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "rsna-knee-nhan-llm-cua-minh", "id": DS_ID, "licenses": [{"name": "CC0-1.0"}]}))
    if DS_ID.split("/")[1] in chay("datasets", "list", "--mine", "-s", "rsna-knee-nhan-llm-cua-minh"):
        chay("datasets", "version", "-p", str(ds), "-m", f"them {ten_nhan}")
    else:
        chay("datasets", "create", "-p", str(ds))          # mac dinh rieng tu
    for _ in range(60):   # day notebook truoc khi dataset xu ly xong thi Kaggle bo dataset ra
        if "ready" in chay("datasets", "status", DS_ID):
            break
        time.sleep(20)

    day_notebook(ten_nhan, thu_muc, a.mo_hinh)


def day_notebook(ten_nhan: str, thu_muc: Path, mo_hinh: str) -> None:
    goc = (REPO / "scripts" / "81_kaggle_train_25d.py").read_text()
    moc = "from __future__ import annotations\n"
    assert goc.count(moc) == 1
    dat = f'\nimport os as _os\n_os.environ.setdefault("RSNA_NHAN", "{ten_nhan}")\n'
    if mo_hinh:
        dat += f'_os.environ.setdefault("RSNA_MO_HINH", "{mo_hinh}")\n'
    (thu_muc / TEN_FILE).write_text(goc.replace(moc, moc + dat, 1))
    chay("kernels", "push", "-p", str(thu_muc))


if __name__ == "__main__":
    main()
