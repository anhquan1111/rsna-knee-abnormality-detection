"""Nhan ket qua tu Kaggle notebook vao repo va kiem tra truoc khi dung.

Chay sau khi tai output cua notebook trich dac trung ve mot thu muc bat ky:

    python scripts/07_ingest_kaggle_features.py data/kaggle_out
    python scripts/07_ingest_kaggle_features.py data/kaggle_out --planes coronal axial

Viec no lam:
  1. Tim moi `features_dinov2_<plane>.npz` + `dicom_headers*.csv` + `extract_summary.json`.
  2. GOP CAC MAT PHANG: noi cac lat cua cung mot study lai voi nhau.
  3. Kiem dac trung: dung so chieu, khong co nan/inf, khop UID trong train.csv.
  4. Doi chieu voi dac trung da trich o may (neu trich lai cung mat phang).
  5. Chuan hoa ten hang may chup, ghi cot `manufacturer_norm` vao manifest header.
  6. Ghi de cache dac trung (ban cu duoc doi ten thanh .npz.bak).

Hai quyet dinh dang ghi lai:

* Gop bang cach NOI LAT chu khong phai noi VECTOR dac trung. Noi vector se doi so chieu
  dau vao cua head va bien day thanh thi nghiem hai bien (them anh + doi kien truc), luc
  do khong con quy duoc thay doi diem so ve nguyen nhan nao. Noi lat giu kien truc y
  nguyen: bien duy nhat thay doi la luong du lieu. Masked pooling von da xu ly so lat
  khac nhau giua cac study nen khong can sua gi them.

* Luu float16. Ba mat phang la 436,950 lat x 384 chieu = 671 MB o float32, qua nang cho
  may nay (da tung segfault khi RAM trong con 3.2 GB). fp16 con mot nua. Day khong phai
  danh doi do chinh xac: GPU Kaggle von da tinh o fp16 luc trich, vong truoc do cosin
  giua ban fp16 va ban fp32 la 0.999988-0.999997.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rsna_knee.config import DATA_INTERIM, DATA_MANIFEST, LABELS, STUDY_COL
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw

# Bay da gap that o ngay 3: bay chuoi `Manufacturer` nhung chi la bon hang.
# Nhom theo chuoi tho se tao nhom gia, va split "theo may chup" van de hai ca cung hang
# nam hai ben. Phai chuan hoa truoc khi dung lam khoa nhom.
VENDOR_RULES = (
    ("siemens", "SIEMENS"),
    ("philips", "PHILIPS"),
    ("ge medical", "GE"),
    ("ge health", "GE"),
    ("gehc", "GE"),                 # viet tat cua GE HealthCare
    # Toshiba Medical -> Canon Medical (2016), Hitachi Medical -> Fujifilm Healthcare
    # (2021). Cung mot dong may duoc doi ten, nen dau van tay thiet bi la mot. Muc dich
    # o day la nhom theo PHAN CUNG chu khong theo phap nhan, nen gop lai.
    ("toshiba", "CANON"),
    ("canon", "CANON"),
    ("hitachi", "FUJIFILM"),
    ("fujifilm", "FUJIFILM"),
    ("united imaging", "UIH"),
)


def normalize_vendor(raw: object) -> str:
    text = str(raw).strip().lower()
    if text in ("", "nan", "none"):
        return "UNKNOWN"
    for needle, name in VENDOR_RULES:
        if needle in text:
            return name
    return text.upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", help="thu muc chua ba file tai tu Kaggle ve")
    ap.add_argument("--planes", nargs="+", default=None,
                    help="mac dinh: tu tim moi features_dinov2_*.npz trong thu muc")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out-name", dest="out_name", default=None,
                    help="ten file dich; mac dinh suy ra tu ten file nguon")
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser()
    if args.planes:
        files = [src / f"features_dinov2_{p.lower()}.npz" for p in args.planes]
    else:
        files = sorted(src.glob("features_dinov2_*.npz"))
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit(f"Khong thay features_dinov2_*.npz trong {src}\n"
                         f"Co trong thu muc do: {sorted(p.name for p in src.glob('*'))[:20]}")

    old_path = DATA_INTERIM / f"features_dinov2_vits14_{args.size}.npz"
    heads = sorted(src.glob("dicom_headers*.csv"))
    summary = src / "extract_summary.json"
    if summary.exists():
        print("=== Tom tat tu Kaggle ===")
        for plane, t in json.loads(summary.read_text()).items():
            if isinstance(t, dict) and "n_studies" in t:
                print(f"  {plane:<10} {t['n_studies']:,} study | {t['n_gold']}/58 gold | "
                      f"{t['n_slices']:,} lat | {t.get('n_failed', '?')} loi")

    print("\n=== 1. Gop dac trung cac mat phang ===")
    # Gop bang cach NOI CAC LAT lai, khong phai gop vector dac trung. Nho vay kien truc
    # model khong doi mot chu nao - bien duy nhat thay doi la LUONG DU LIEU, dung tinh
    # than thi nghiem co kiem soat cua ngay 8.
    nguon = list(files)
    if old_path.exists() and not any("sagittal" in f.name.lower() for f in files):
        # Sagittal da trich o vong 1 va da nam trong cache. No la MOT mat phang nhu hai
        # mat phang moi, nen dua vao cung duong gop thay vi xu ly rieng.
        nguon.insert(0, old_path)

    feats: dict[str, np.ndarray] = {}
    for f in nguon:
        plane = ("sagittal (cache)" if f == old_path
                 else f.stem.split("_")[-1])
        n_lat, n_study = 0, 0
        with np.load(f) as z:
            for k in z.files:
                a = z[k]
                n_lat += a.shape[0]
                n_study += 1
                # Luu float16: 436,950 lat x 384 chieu = 671 MB o float32, qua nang cho
                # may nay (da tung segfault o 230 MB). float16 con mot nua, va sai so cung
                # bac voi fp16 ma GPU Kaggle da dung luc trich - khong them mat mat gi.
                a16 = a.astype(np.float16)
                feats[k] = np.concatenate([feats[k], a16]) if k in feats else a16
        print(f"  {plane:<17} {n_study:,} study | {n_lat:,} lat")

    dims = {f.shape[1] for f in feats.values()}
    n_slices = sum(f.shape[0] for f in feats.values())
    if len(dims) != 1:
        raise SystemExit(f"Dac trung co nhieu so chieu khac nhau: {dims}\n"
                         "  Gan nhu chac chan la dang tron hai backbone khac nhau.")
    dim = next(iter(dims))
    print(f"  GOP LAI           {len(feats):,} study | {n_slices:,} lat | {dim} chieu | "
          f"{n_slices * dim * 2 / 1024**2:.0f} MB trong RAM (float16)")

    bad = [k for k, v in feats.items() if not np.isfinite(v).all()]
    print(f"  study co nan/inf: {len(bad)}" + (f" -> {bad[:5]}" if bad else ""))
    if bad:
        raise SystemExit("Co dac trung khong huu han - khong dung duoc, chay lai tren Kaggle")

    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold = gold_subset(man)
    known = set(man[STUDY_COL])
    lac = set(feats) - known
    print(f"  UID khop train.csv: {len(set(feats) & known):,}/{len(feats):,}"
          + (f"  | LAC: {len(lac)}" if lac else ""))
    have_gold = set(feats) & set(gold[STUDY_COL])
    print(f"  study co nhan nguoi gan: {len(have_gold)}/58"
          + ("  <- DU" if len(have_gold) == 58 else "  <- CON THIEU"))

    print("\n=== 2. Doi chieu voi dac trung da trich o may ===")
    # Doi chieu phai lam tren RIENG mat phang sagittal: dac trung cu o may chi co sagittal,
    # con `feats` gio la ba mat phang noi lai nen so lat khac nhau la dung, khong phai loi.
    # Chi doi chieu khi so sanh CO NGHIA: cung backbone (cung so chieu) va cung cach gop.
    # Cache cu la ban da GOP BA MAT PHANG, con `sag` chi la mot mat phang - so truc tiep
    # thi moi study deu "lech so lat", va canh bao do khong noi len gi ngoai viec hai ben
    # khong cung loai. Canh bao sai con te hon khong canh bao: lan sau gap canh bao that
    # se bi bo qua.
    sag = next((f for f in files if "sagittal" in f.name.lower()), None)
    cung_loai = False
    if sag is not None and old_path.exists():
        with np.load(old_path) as z:
            cung_loai = bool(z.files) and z[z.files[0]].shape[1] == dim
        if not cung_loai:
            print(f"  cache cu co so chieu khac ({dim} vs cu) - khac backbone, khong so duoc")
    if sag is not None and old_path.exists() and cung_loai:
        with np.load(old_path) as z:
            old = {k: z[k] for k in z.files}
        with np.load(sag) as z:
            new_sag = {k: z[k] for k in z.files[:200]}
        shared = sorted(set(old) & set(new_sag))
        if shared:
            # Lech tuyet doi mot minh KHONG doc duoc: dac trung DINOv2 co do lon rat khac
            # nhau giua cac chieu. Thuoc do dung la do tuong dong cosin - no tra loi dung
            # cau hoi can hoi: "hai ben co phai CUNG MOT bieu dien khong?"
            cos, rel, lech_shape = [], [], 0
            for uid in shared[:50]:
                a, b = old[uid], new_sag[uid]
                if a.shape != b.shape:
                    lech_shape += 1
                    continue
                af, bf = a.ravel(), b.ravel()
                cos.append(float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)))
                rel.append(float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12)))

            if lech_shape:
                print(f"  {lech_shape} study lech so lat - KHAC tien xu ly, phai xem lai")
            if cos:
                print(f"  doi chieu {len(cos)} study chung:")
                print(f"    tuong dong cosin : {min(cos):.6f} - {max(cos):.6f}")
                print(f"    lech tuong doi   : {max(rel):.2%} so voi gia tri lon nhat")
                if min(cos) > 0.999:
                    print("    -> CUNG mot bieu dien. Chenh lech den tu GPU fp16 (Kaggle)")
                    print("       vs CPU fp32 (may) - khong anh huong ket qua.")
                else:
                    print("    -> KHAC bieu dien. Kiem lai tien xu ly hai ben co giong nhau")
                    print("       khong (chuan hoa, resize, thu tu lat).")
        else:
            print("  khong co study chung de doi chieu")
    elif sag is None:
        print("  lan nay khong trich lai sagittal (da co san trong cache) - khong can")
        print("  doi chieu. Vong truoc da xac nhan cosin 0.999988-0.999997.")
    else:
        print(f"  chua co {old_path.name} o may - bo qua buoc doi chieu")

    print("\n=== 3. Chuan hoa ten hang may chup ===")
    if not heads:
        raise SystemExit(f"Khong thay dicom_headers*.csv trong {src}")
    hdf = pd.concat([pd.read_csv(h, dtype={"StudyInstanceUID": str,
                                           "SeriesInstanceUID": str}) for h in heads],
                    ignore_index=True)
    print(f"  gop {len(heads)} file header: {', '.join(h.name for h in heads)}")
    hdf = hdf.drop_duplicates(subset=["SeriesInstanceUID"])
    hdf["manufacturer_norm"] = hdf["Manufacturer"].map(normalize_vendor)
    raw_n = hdf["Manufacturer"].nunique(dropna=False)
    norm_n = hdf["manufacturer_norm"].nunique()
    print(f"  chuoi Manufacturer tho : {raw_n}")
    print(f"  sau khi chuan hoa      : {norm_n}")
    print(hdf["manufacturer_norm"].value_counts().to_string())
    per_study = (hdf.groupby("StudyInstanceUID")["manufacturer_norm"]
                 .agg(lambda s: s.mode().iat[0]).reset_index())
    print(f"  study co hang may chup : {len(per_study):,}")

    print("\n=== 4. Ghi vao repo ===")
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST.mkdir(parents=True, exist_ok=True)

    # Ten file dich lay tu ten file NGUON, khong viet cung.
    #
    # Truoc day o day viet cung `features_dinov2_vits14_{size}.npz`. Nhap dac trung cua
    # mot backbone khac vao se GHI DE len cache cu bang du lieu hoan toan khac, duoi mot
    # cai ten noi doi. Ban .bak cuu duoc file nhung khong cuu duoc chuyen moi script sau
    # do doc mot thu va tuong la thu khac.
    if args.out_name:
        ten_dich = args.out_name
    else:
        mau = files[0].stem                      # features_dinov2_vitb14_336_sagittal
        bo = mau.replace("features_", "")
        for pl in ("sagittal", "coronal", "axial"):
            bo = bo.replace(f"_{pl}", "")
        ten_dich = f"features_{bo}.npz" if bo else f"features_dinov2_vits14_{args.size}.npz"
    dst_npz = DATA_INTERIM / ten_dich
    print(f"  ten file dich: {ten_dich}  (suy ra tu ten file nguon)")

    if dst_npz.exists():
        # Neu file cung ten ma KHAC so chieu thi gan nhu chac chan la nham. Dung han.
        with np.load(dst_npz) as z:
            dim_cu = z[z.files[0]].shape[1] if z.files else dim
        if dim_cu != dim:
            raise SystemExit(
                f"{ten_dich} da ton tai voi {dim_cu} chieu, nhung dac trung moi la {dim} "
                f"chieu.\n  Hai backbone khac nhau dung chung mot ten file - dung ten "
                f"khac bang --out-name.")
        backup = dst_npz.with_suffix(".npz.bak")
        shutil.move(str(dst_npz), str(backup))
        print(f"  ban cu doi ten thanh {backup.name}")
    np.savez(dst_npz, **feats)   # khong nen: fp16 gan nhu khong nen duoc,
                                 # nen nen chi ton them vai phut CPU
    hdf.to_csv(DATA_MANIFEST / "dicom_headers.csv", index=False)
    per_study.to_csv(DATA_MANIFEST / "study_manufacturer.csv", index=False)
    print(f"  {dst_npz}  ({dst_npz.stat().st_size/1024**2:,.0f} MB)")
    print(f"  {DATA_MANIFEST / 'dicom_headers.csv'}")
    print(f"  {DATA_MANIFEST / 'study_manufacturer.csv'}")

    print("\n=== Chay tiep ===")
    print("  python scripts/31_train_frozen_head.py --backbone dinov2_vits14 --epochs 80")
    print("  python scripts/40_experiment_weak_labels.py --backbone dinov2_vits14 --epochs 80")


if __name__ == "__main__":
    main()
