"""NGAY 14 - Cham CNN 2.5D train toan bo (notebook 81 tren Kaggle) tren 58 ca gold.

    python scripts/82_cham_cnn_25d.py --tai     # tai du_doan.npz + nhat_ky.json tu Kaggle
    python scripts/82_cham_cnn_25d.py           # cham lai tu file da tai

58 ca gold KHONG vao train cua notebook 81, nen diem tren chung la diem ngoai mau truc
tiep - khong can chia fold. Quy tac chot TRUOC khi xem diem (ghi trong notebook 81):
    * moi model = trong so EMA o epoch cuoi; duong cong theo epoch chi de minh bach.
    * CNN cuoi = trung binh THEO HANG cua cac model (moi GPU mot model), trong so bang nhau.
    * gop voi bo DINOv2 cu (5 head x 3 seed, OOF): trung binh THEO HANG, ty le 50/50.
      Hai ben deu chua tung thay 58 ca nay luc train, nen phep gop van la ngoai mau.

Ghi:
    reports/day14_cnn_du_doan.npz   du doan gold moi epoch cua tung model (KHONG nhan,
                                    KHONG UID: chi ma bam thu tu ca, giong file day13)
    reports/day14_cnn_25d.json      diem tung epoch, KTC, tung nhan, phep gop
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.stats import rankdata

from rsna_knee.config import LABELS, REPORTS_DIR, SEED
from rsna_knee.metrics import macro_auc
from rsna_knee.oof import gold_theo_fold, ma_thu_tu, nap_oof

L = list(LABELS)
REPO = Path(__file__).resolve().parents[1]
# --nhan-minh: cham lan train lai bang nhan LLM tu gan (script 90/92), so voi lan nhan cong khai
NHAN_MINH = "--nhan-minh" in sys.argv
TAI_VE = REPO / "data" / ("kaggle_cnn_nhan_minh" if NHAN_MINH else "kaggle_cnn")
KERNEL = "votrananhquan/rsna-knee-train-25d" + ("-nhan-minh" if NHAN_MINH else "")
RA_NPZ = REPORTS_DIR / ("day15_cnn_nhan_minh_du_doan.npz" if NHAN_MINH else "day14_cnn_du_doan.npz")
RA_JSON = REPORTS_DIR / ("day15_cnn_nhan_minh.json" if NHAN_MINH else "day14_cnn_25d.json")
TEN_DINO = ["mp_sagittal", "mp_axial", "mp_coronal", "chung_s224", "chung_b336"]


def hang(p):
    return np.column_stack([rankdata(p[:, j]) for j in range(p.shape[1])])


def ktc(y, p, n=2000, seed=0):
    rng, b = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        r = macro_auc(y[i], p[i], L)
        if r.n_scored == len(L):
            b.append(r.macro_auc)
    return [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])]


def ktc_cap(y, a, b, n=2000, seed=0):
    rng, d = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try:
            d.append(macro_auc(y[i], b[i], L).macro_auc - macro_auc(y[i], a[i], L).macro_auc)
        except ValueError:
            pass
    return [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]


def main() -> None:
    if "--tai" in sys.argv:
        TAI_VE.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(REPO / ".venv" / "bin" / "kaggle"), "kernels", "output", KERNEL, "-p", str(TAI_VE),
                        "-o", "--file-pattern", r"(du_doan\.npz|nhat_ky\.json|\.log)$"], check=True)

    y, fold, uid = gold_theo_fold()
    thu_muc = sorted(f.parent for f in TAI_VE.glob("*/du_doan.npz"))
    if not thu_muc:
        raise SystemExit(f"Khong thay */du_doan.npz trong {TAI_VE} - chay voi --tai")
    mo_hinh, luu = {}, {}
    for d in thu_muc:
        z = np.load(d / "du_doan.npz")
        nk = json.loads((d / "nhat_ky.json").read_text())
        vi_tri = {u: k for k, u in enumerate(z["gold_uid"])}
        thieu = [u for u in uid if u not in vi_tri]
        if thieu:
            raise SystemExit(f"{d.name}: {len(thieu)} ca gold khong co du doan (khong doc duoc anh?)")
        sap = [vi_tri[u] for u in uid]                   # dua ve thu tu cua gold_theo_fold
        ema, tho = z["gold_ema"][:, sap], z["gold_tho"][:, sap]
        mo_hinh[d.name] = {"ema": ema, "nk": nk}
        luu[f"{d.name}__ema"], luu[f"{d.name}__tho"] = ema, tho

    # Bo DINOv2 cu: 5 head x 3 seed o epoch 9, gop theo hang (giong 48 --cham)
    seeds = [SEED] + sorted({int(f.stem.split("_seed")[1]) for f in REPORTS_DIR.glob("day13_epoch_*_seed*.npz")})
    ten = lambda t, s: REPORTS_DIR / (f"day13_epoch_{t}.npz" if s == SEED else f"day13_epoch_{t}_seed{s}.npz")
    dino = np.mean([hang(nap_oof(ten(t, s), uid, fold)[8]) for s in seeds for t in TEN_DINO
                    if ten(t, s).exists()], axis=0)
    cnn = np.mean([hang(m["ema"][-1]) for m in mo_hinh.values()], axis=0)
    gop = 0.5 * hang(cnn) + 0.5 * hang(dino)
    r_cnn, r_dino = macro_auc(y, cnn, L), macro_auc(y, dino, L)

    kq = {"tung_model": {}, "n_dino_seed": len(seeds)}
    for tn, m in mo_hinh.items():
        nk, ema = m["nk"], m["ema"]
        tho = luu[f"{tn}__tho"]
        kq["tung_model"][tn] = {
            "backbone": nk["cau_hinh"].get("BACKBONE"), "epoch_chay": int(ema.shape[0]),
            "n_train": nk.get("n_train"), "gio": nk.get("gio"),
            "auc": round(macro_auc(y, ema[-1], L).macro_auc, 4), "ci": ktc(y, ema[-1]),
            "theo_epoch": [{"epoch": e + 1, "ema": round(macro_auc(y, ema[e], L).macro_auc, 4),
                            "tho": round(macro_auc(y, tho[e], L).macro_auc, 4),
                            "val_llm": nk["epoch"][e].get("val_llm_ema") if e < len(nk["epoch"]) else None}
                           for e in range(ema.shape[0])]}
    kq.update({
        "cnn": {"auc": round(r_cnn.macro_auc, 4), "ci": ktc(y, cnn)},
        "dino": {"auc": round(r_dino.macro_auc, 4), "ci": ktc(y, dino)},
        "gop_50_50": {"auc": round(macro_auc(y, gop, L).macro_auc, 4), "ci": ktc(y, gop)},
        "cnn_tru_dino": {"chenh": round(r_cnn.macro_auc - r_dino.macro_auc, 4), "ci": ktc_cap(y, dino, cnn)},
        "gop_tru_dino": {"chenh": round(macro_auc(y, gop, L).macro_auc - r_dino.macro_auc, 4),
                         "ci": ktc_cap(y, dino, gop)},
        "tung_nhan": {k: {"cnn": round(r_cnn.per_label[k], 3), "dino": round(r_dino.per_label[k], 3),
                          "gop": round(macro_auc(y, gop, L).per_label[k], 3)} for k in L},
    })
    if NHAN_MINH:
        # Cung kien truc, cung seed, cung chia - chi khac nguon nhan: chenh nay la cua nhan
        with np.load(REPORTS_DIR / "day14_cnn_du_doan.npz") as zc:
            if str(zc["uid_md5"]) != ma_thu_tu(uid):
                raise SystemExit("day14_cnn_du_doan.npz: thu tu ca khac - khong so duoc")
            cu = np.mean([hang(zc[k][-1]) for k in zc.files if k.endswith("__ema")], axis=0)
        kq["cnn_nhan_cong_khai"] = {"auc": round(macro_auc(y, cu, L).macro_auc, 4), "ci": ktc(y, cu)}
        kq["nhan_minh_tru_cong_khai"] = {"chenh": round(r_cnn.macro_auc - macro_auc(y, cu, L).macro_auc, 4),
                                         "ci": ktc_cap(y, cu, cnn)}
    np.savez_compressed(RA_NPZ, fold=fold, uid_md5=ma_thu_tu(uid), **luu)
    RA_JSON.write_text(json.dumps(kq, indent=2, ensure_ascii=False))

    for tn, m in kq["tung_model"].items():
        print(f"{tn} ({m['epoch_chay']} epoch, {m['n_train']} ca train, {m['gio']} gio)")
        print("  epoch | EMA    | tho    | val LLM")
        for d in m["theo_epoch"]:
            v = d["val_llm"]
            print(f"  {d['epoch']:>5} | {d['ema']:.4f} | {d['tho']:.4f} | {v if v is None else round(v, 4)}")
        print(f"  EMA epoch cuoi (chot truoc): {m['auc']:.4f}  KTC {m['ci']}\n")
    print(f"  CNN gop {len(mo_hinh)} model theo hang  : {kq['cnn']['auc']:.4f}  KTC {kq['cnn']['ci']}")
    print(f"  DINOv2 cu (5 head x {len(seeds)} seed)     : {kq['dino']['auc']:.4f}  KTC {kq['dino']['ci']}")
    print(f"  CNN - DINOv2                     : {kq['cnn_tru_dino']['chenh']:+.4f}  KTC bat cap {kq['cnn_tru_dino']['ci']}")
    print(f"  Gop 50/50 CNN + DINOv2 theo hang : {kq['gop_50_50']['auc']:.4f}  KTC {kq['gop_50_50']['ci']}")
    print(f"  Gop - DINOv2                     : {kq['gop_tru_dino']['chenh']:+.4f}  KTC bat cap {kq['gop_tru_dino']['ci']}")
    if NHAN_MINH:
        c = kq["nhan_minh_tru_cong_khai"]
        print(f"  CNN nhan cong khai (ngay 14)     : {kq['cnn_nhan_cong_khai']['auc']:.4f}  KTC {kq['cnn_nhan_cong_khai']['ci']}")
        print(f"  Nhan tu gan - nhan cong khai     : {c['chenh']:+.4f}  KTC bat cap {c['ci']}")
    print("\n  Tung nhan (CNN | DINOv2 | gop):")
    for k, v in sorted(kq["tung_nhan"].items(), key=lambda kv: -kv[1]["cnn"]):
        print(f"    {k:<18} {v['cnn']:.3f} | {v['dino']:.3f} | {v['gop']:.3f}")


if __name__ == "__main__":
    main()
