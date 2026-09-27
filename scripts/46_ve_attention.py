"""Ve xem model NHIN VAO LAT NAO cho tung benh.

    python scripts/46_ve_attention.py                 # tu chon mot ca gold co anh o may
    python scripts/46_ve_attention.py --uid 520594649149 --benh ACL Effusion

`PerLabelAttnPool` cho moi benh mot bo trong so chu y RIENG tren cac lat. Do la thu duy
nhat trong ca pipeline co the nhin thay bang mat: no tra loi "model dang doc lat nao de
ket luan benh nay".

DOC HINH CHO DUNG
=================
Day KHONG phai giai thich nhan qua. Trong so chu y cao chi noi rang lat do dong gop nhieu
vao vector gop cua benh do - no khong chung minh model dang nhin vao dung cau truc giai
phau cua benh. Mot model hoc phai dau van tay may chup (da do duoc: doan dung hang may
82,8%) van co the cho ra ban do chu y trong rat hop ly.

Hinh nay dung de: (1) kiem rang chu y KHONG deu nhau giua cac benh - neu deu nhau thi
`per_label_attn` khong hoc duoc gi va chi ton tham so; (2) tim ca de soi tay.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rsna_knee.config import DATA_INTERIM, DATA_RAW, LABELS, REPORTS_DIR, STUDY_COL, short_uid
from rsna_knee.dicom_io import normalize_series
from rsna_knee.head import StudyHead
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw

L = list(LABELS)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def doc_series(d: Path):
    import pydicom

    items = []
    for p in sorted(d.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(p))
            items.append((int(getattr(ds, "InstanceNumber", -1)), ds.pixel_array))
        except Exception:
            continue
    if not items:
        raise FileNotFoundError(d)
    items.sort(key=lambda t: t[0])
    return np.stack([a for _, a in items])


def resize(arr: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    out = np.empty((arr.shape[0], size, size), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.asarray(Image.fromarray(arr[i]).resize((size, size), Image.BILINEAR),
                            dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default=None, help="12 ky tu cuoi cua StudyInstanceUID")
    ap.add_argument("--benh", nargs="+", default=None)
    ap.add_argument("--top", type=int, default=4, help="so lat chu y cao nhat de ve")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    ck = DATA_INTERIM / "ensemble" / "head_mp_sagittal.pt"
    if not ck.exists():
        raise SystemExit(f"Chua co {ck}. Chay scripts/45_train_final_ensemble.py truoc.")
    blob = torch.load(ck, map_location="cpu", weights_only=True)
    if blob["pooling"] != "per_label_attn":
        raise SystemExit(f"Head nay dung pooling {blob['pooling']}, khong co chu y "
                         "rieng tung benh de ve.")

    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold = gold_subset(man).set_index(STUDY_COL)
    root = DATA_RAW / "images"
    co = {d.name for d in root.iterdir() if d.is_dir()}

    ung_vien = [(short_uid(str(u)), u) for u in gold.index if short_uid(str(u)) in co]
    if args.uid:
        ung_vien = [t for t in ung_vien if t[0] == args.uid]
    if not ung_vien:
        raise SystemExit(f"Khong tim thay ca gold nao co anh o may (uid={args.uid})")

    # Chon ca co NHIEU benh duong tinh nhat: hinh se cho thay ro hon rang cac benh khac
    # nhau chu y vao nhung lat khac nhau.
    ung_vien.sort(key=lambda t: -int(np.nansum(gold.loc[t[1], L].to_numpy(dtype=float) == 1)))
    ngan, uid = ung_vien[0]
    nhan = gold.loc[uid, L].to_numpy(dtype=float)
    print(f"ca: ...{ngan} | {int(np.nansum(nhan == 1))} benh duong tinh")

    sdir = next(d for d in (root / ngan).iterdir() if d.is_dir())
    vol_goc = doc_series(sdir)
    vol = resize(normalize_series(vol_goc.astype(np.float32)), args.size)
    print(f"  {vol.shape[0]} lat sagittal")

    print("  dang trich dac trung (CPU, hoi lau) ...", flush=True)
    bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()
    x = torch.from_numpy(vol).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    with torch.no_grad():
        f = torch.cat([bb(x[i:i + 8]) for i in range(0, len(x), 8)])

    head = StudyHead(blob["dim"], pooling="per_label_attn")
    head.load_state_dict(blob["state_dict"])
    head.eval()

    with torch.no_grad():
        xb = f.unsqueeze(0)
        mask = torch.ones(1, xb.shape[1], dtype=torch.bool)
        z = head.norm(xb)
        w = head.pool.score(z).masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = w.softmax(dim=1)[0].numpy()          # (n_lat, 12)
        prob = torch.sigmoid(head(xb, mask))[0].numpy()

    # Chu y co THAT SU khac nhau giua cac benh khong? Neu khong thi `per_label_attn` chi
    # la `attn` thuong doi ten, va ca hinh nay lan them tham so deu vo nghia.
    deu = 1.0 / w.shape[0]
    print(f"\n  chu y neu deu nhau: {deu:.4f} moi lat")
    print(f"  {'benh':<18}{'xac suat':>9}{'nhan':>6}{'lat dinh':>10}{'w dinh':>8}")
    chon = args.benh or [l for l in L if np.isfinite(nhan[L.index(l)])]
    for lab in chon:
        j = L.index(lab)
        k = int(w[:, j].argmax())
        t = nhan[j]
        print(f"  {lab:<18}{prob[j]:>9.3f}{('-' if not np.isfinite(t) else int(t)):>6}"
              f"{k:>10}{w[k, j]:>8.4f}")
    # PHEP KIEM DAU TIEN O DAY QUA YEU va da cho ket luan SAI.
    #
    # No do "chenh lech lon nhat giua hai benh bat ky" roi bao la cac benh chu y khac
    # nhau. Nhung chenh lech do bi chi phoi boi viec benh DUONG TINH va benh AM TINH nhin
    # khac nhau - khong phai boi viec moi benh co vung giai phau rieng. No bao "dat" ngay
    # ca khi ca 7 benh duong tinh dung chung dung MOT lat.
    #
    # Hai so do duoi moi tra loi dung cau hoi:
    dinh = w.argmax(axis=0)
    n_lat_rieng = len(set(dinh.tolist()))
    # Entropy chuan hoa: 1.0 = trai deu moi lat, 0.0 = don het vao mot lat.
    p = np.clip(w, 1e-12, 1)
    ent = float((-(p * np.log(p)).sum(axis=0) / np.log(w.shape[0])).mean())
    print(f"\n  so lat rieng biet duoc chon lam dinh: {n_lat_rieng}/{len(L)} benh")
    print(f"  entropy chu y (1 = trai deu, 0 = don mot lat): {ent:.3f}")
    from collections import Counter
    pho = Counter(dinh.tolist()).most_common(1)[0]
    print(f"  lat duoc nhieu benh chon nhat: lat {pho[0]} ({pho[1]}/{len(L)} benh)")
    if n_lat_rieng <= len(L) // 3 or ent < 0.3:
        print("  -> CHU Y BI SUY BIEN: cac benh dung chung qua it lat, va moi benh gan")
        print("     nhu chi dung MOT lat. Day KHONG phai 'moi benh nhin giai phau cua no'")
        print("     ma giong model tim mot lat noi 'ca nay bat thuong' roi tuon moi du")
        print("     doan duong tinh qua do. Khop voi viec dac trung doan dung hang may")
        print("     82,8% - tin hieu o day la CAP TOAN CA, khong phai cap ton thuong.")
    else:
        print("  -> cac benh chu y vao nhung vung khac nhau")

    # --- ve ---
    ve = chon[:6]
    fig, axes = plt.subplots(len(ve), args.top + 1,
                             figsize=(2.0 * (args.top + 1), 2.15 * len(ve)))
    if len(ve) == 1:
        axes = axes[None, :]
    for r, lab in enumerate(ve):
        j = L.index(lab)
        thu_tu = np.argsort(-w[:, j])[:args.top]
        t = nhan[j]
        nhan_txt = "?" if not np.isfinite(t) else ("CO" if t == 1 else "khong")
        axes[r, 0].plot(w[:, j], color="tab:red", lw=1.4)
        axes[r, 0].axhline(deu, color="gray", ls="--", lw=0.8)
        axes[r, 0].scatter(thu_tu, w[thu_tu, j], color="tab:red", s=14, zorder=3)
        axes[r, 0].set_ylabel(f"{lab}\np={prob[j]:.2f} | {nhan_txt}", fontsize=8)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("chu y theo lat", fontsize=9)
        for c, k in enumerate(thu_tu):
            ax = axes[r, c + 1]
            ax.imshow(vol[k], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"lat {k}  w={w[k, j]:.3f}", fontsize=7)
    fig.suptitle(f"Model nhin vao lat nao? - ca ...{ngan} (sagittal, {vol.shape[0]} lat)\n"
                 f"duong dut = muc chu y neu deu nhau ({deu:.3f})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"attention_{ngan}.png"
    fig.savefig(out, dpi=130)
    print(f"\nDa ghi {out}")


if __name__ == "__main__":
    main()
