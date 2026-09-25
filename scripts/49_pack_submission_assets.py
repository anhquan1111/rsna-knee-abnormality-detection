"""Dong goi asset can thiet cho notebook nop bai (chay o may, khong phai Kaggle).

Vi sao can: notebook nop bai cua Kaggle KHONG CO INTERNET, nen `torch.hub.load` se that
bai. Phai mang san ma nguon DINOv2, trong so backbone va checkpoint head len lam Dataset.

Chay:  python scripts/49_pack_submission_assets.py
Ra:    data/submit_assets/  (~85 MB) - upload ca thu muc nay len Kaggle Datasets
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from rsna_knee.config import DATA_INTERIM, REPO_ROOT

OUT = REPO_ROOT / "data" / "submit_assets"
HUB = Path(torch.hub.get_dir())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Ma nguon DINOv2 tu cache torch.hub
    repo_src = HUB / "facebookresearch_dinov2_main"
    if not (repo_src / "hubconf.py").exists():
        raise SystemExit(
            f"Khong thay ma nguon DINOv2 o {repo_src}.\n"
            "  Chay mot lan o may de torch.hub tai ve (can Internet):\n"
            '    python -c "import torch; torch.hub.load(\'facebookresearch/dinov2\', '
            "'dinov2_vits14')\""
        )
    repo_dst = OUT / "dinov2_repo"
    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    # Bo .git va __pycache__ cho nhe va cho khoi vuong khi upload
    shutil.copytree(repo_src, repo_dst,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    print(f"  ma nguon DINOv2 -> {repo_dst.name}/")

    # 2. Trong so backbone
    w_src = HUB / "checkpoints" / "dinov2_vits14_pretrain.pth"
    if not w_src.exists():
        raise SystemExit(f"Khong thay trong so o {w_src}")
    shutil.copy2(w_src, OUT / w_src.name)
    print(f"  trong so backbone -> {w_src.name} ({w_src.stat().st_size/1024**2:.0f} MB)")

    # 3. Checkpoint head da train
    heads = sorted(DATA_INTERIM.glob("head_*.pt"))
    if not heads:
        raise SystemExit("Khong thay head_*.pt trong data/interim.\n"
                         "  Chay scripts/31_train_frozen_head.py truoc.")
    # CHI dong goi MOT head. Truoc day goi het, va notebook nop bai lay file dau tien
    # rglob tra ve - thu tu do khong xac dinh, nen co the nop bang head cu cua vong truoc
    # ma khong co dau hieu nao bao. Chon theo OOF AUC de lua chon la tuong minh.
    xep = sorted(((torch.load(h, map_location="cpu", weights_only=True), h) for h in heads),
                 key=lambda t: t[0].get("oof_macro_auc") or -1, reverse=True)
    for blob, h in xep:
        print(f"  co: {h.name:<28} pooling {str(blob.get('pooling')):<5} "
              f"OOF AUC {blob.get('oof_macro_auc')}")
    blob, best = xep[0]
    for cu in OUT.glob("head_*.pt"):      # don head cua lan dong goi truoc
        cu.unlink()
    shutil.copy2(best, OUT / best.name)
    print(f"  head -> {best.name} ({best.stat().st_size/1024:.0f} KB) | "
          f"pooling {blob.get('pooling')} | OOF AUC {blob.get('oof_macro_auc')} <- CHON")

    tong = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nXong: {OUT}  ({tong/1024**2:.0f} MB)")
    print("Buoc tiep: Kaggle -> Datasets -> New Dataset -> upload ca thu muc nay,")
    print("           roi Add Input vao notebook nop bai.")


if __name__ == "__main__":
    main()
