"""CHAY TREN KAGGLE NOTEBOOK - khong chay o may local.

Trich dac trung DINOv2 cho nhieu mat phang trong MOT lan chay. Moi mat phang ghi mot
file rieng:
    /kaggle/working/features_<backbone>_<size>_<plane>.npz   dac trung tung lat
    /kaggle/working/dicom_headers_<backbone>_<size>_<plane>.csv  metadata may chup
    /kaggle/working/extract_summary_<backbone>_<size>.json       tom tat

Vi sao chay o day: tai anh qua Kaggle API bi gioi han toc do (HTTP 429). Tren Kaggle,
dataset mount san o /kaggle/input nen khong ton request API nao, lai co GPU T4 mien phi.

CACH DUNG
---------
1. Mo trang competition -> New Notebook (hoac mo lai notebook cu).
2. Settings: Accelerator = GPU T4 x2, Internet = ON (de tai trong so DINOv2).
3. Dan file nay vao MOT cell. Sua PLANES o CONFIG cho dung thu can chay.
4. **Save Version -> Save & Run All (Commit)**, KHONG phai Run All.
   Ban Commit chay tren may chu Kaggle, khong can giu tab mo, va ket qua duoc luu vao
   version nen tai ve bang API duoc - khong phai bam tai tung file tren trinh duyet.
5. Chay xong bao lai; ket qua se duoc tai ve bang `kaggle kernels output`.

THOI GIAN: ~52 phut moi mat phang cho 4,407 study (do that: 149,496 lat, 0 ca loi).

CHAY LAI DUOC
-------------
* Mat phang nao da co file .npz day du trong /kaggle/working thi duoc bo qua.
* Trong mot mat phang, ghi tam moi 300 study; chay lai se doc lai va bo qua study da xong.
* Study co nhan nguoi gan xep LEN DAU, nen 58 ca quan trong nhat xong trong vai phut.
* Mot mat phang loi khong lam mat ket qua cua mat phang da xong truoc do.
* Sang PHIEN MOI thi /kaggle/working trong tron: muon chay tiep phai upload ban .npz da
  tai ve len lam Dataset roi Add Input - script tu tim trong /kaggle/input.
* Tai trong so DINOv2 tu thu lai 6 lan khi gap loi tam thoi (5xx/429/mat ket noi).

GIOI HAN KAGGLE: 12 tieng moi phien, 20 GB o /kaggle/working. Ba mat phang uoc tinh
~600 MB, nam thoai mai trong gioi han.
"""
from __future__ import annotations

# ============================== CONFIG ==============================
# Chay nhieu mat phang trong MOT lan, moi cai ghi mot file rieng. Mat phang nao da co
# file day du thi bo qua, nen chay lai khong lam lai tu dau.
PLANES = ["Sagittal", "Coronal", "Axial"]
BACKBONE = "dinov2_vitb14"   # do o scripts/62 tren T4: B@336 chi dat hon S@224 ~40 phut
SIZE = 336                   # boi cua 14. 336 -> 24x24 = 576 patch moi lat (224 -> 256)
BATCH = 32                   # B@336 nang hon S@224, giam batch cho chac VRAM (T4 co 16 GB)
# Ten file mang theo backbone + size. Tron dac trung cua hai backbone vao mot file la
# loi khong nhin thay duoc; dat ten ro rang la cach re nhat de no khong xay ra.
TEN = f"{BACKBONE}_{SIZE}"
ONLY_LABELLED = False     # True = chi 58 study co nhan nguoi gan (chay ~2 phut de thu)
MAX_STUDIES = 0           # 0 = khong gioi han. CHAY THU LAN DAU: dat 60 (~3 phut)
                          # de chac chan moi thu chay duoc, roi doi ve 0 va Run All lai.
# ====================================================================

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

COMPETITION = "rsna-knee-abnormality-detection"


def find_root() -> Path:
    """Tim thu muc chua train_series.csv trong /kaggle/input.

    Kaggle dat du lieu o CHO KHAC NHAU tuy cach them input:
        /kaggle/input/competitions/<slug>/   <- khi mo notebook tu trang competition
        /kaggle/input/<slug>/                <- khi Add Input thu cong
    Hard-code mot duong dan la hong ngay dong doc CSV dau tien, voi FileNotFoundError
    khong he nhac gi toi chuyen duong dan sai cho.

    Chi thu mot so duong dan ung vien roi glob nong 2 cap - KHONG rglob ca cay, vi
    /kaggle/input co 819,640 file va duyet het se treo vai phut.
    """
    base = Path("/kaggle/input")
    for cand in (base / "competitions" / COMPETITION, base / COMPETITION):
        if (cand / "train_series.csv").exists():
            return cand
    for pattern in ("*/train_series.csv", "*/*/train_series.csv"):
        for hit in base.glob(pattern):
            return hit.parent
    found = sorted(p.name for p in base.glob("*"))[:20]
    raise SystemExit(
        f"Khong thay train_series.csv trong {base}.\n"
        f"Co trong {base}: {found}\n"
        "Kiem lai panel Input ben phai da co 'RSNA Knee Abnormality Detection' chua."
    )


# `find_root()` duoc goi trong main() chu khong o cap module: neu goi ngay luc import thi
# khong the import file nay o bat ky dau ngoai Kaggle (ke ca de kiem thu cac ham cua no).
ROOT: Path | None = None
OUT = Path("/kaggle/working")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
META_TAGS = ("StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber", "Manufacturer",
             "ManufacturerModelName", "MagneticFieldStrength", "Rows", "Columns",
             "PixelRepresentation", "SliceThickness")


# ---------------------------------------------------------------- tien xu ly
def normalize_series(arr: np.ndarray) -> np.ndarray:
    """Percentile 1-99 tren CA series. Giong het ban o src/rsna_knee/dicom_io.py.

    Chuan hoa theo series chu khong theo tung lat: cuong do MRI khong tuyet doi, va ep
    tung lat ve cung mot dai se xoa mat do sang tuong doi giua cac lat.
    `+ 1e-8` chan chia cho 0 khi series toan mot gia tri.
    """
    # `np.percentile` tra ve float64; tru vao mang float32 se nang ca ket qua len float64
    # (cache to gap doi ma ham van chay dung). Ep he so ve float32 truoc.
    p_low, p_high = (np.float32(v) for v in np.percentile(arr, (1, 99)))
    out = np.clip(arr, p_low, p_high).astype(np.float32)
    return (out - p_low) / (p_high - p_low + np.float32(1e-8))


def load_series(series_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Doc ca series, sap theo InstanceNumber. Tra ve (N,H,W) va metadata tung lat.

    Sap xep la bat buoc: ten file la UID ngau nhien nen sorted(glob) cho ra mot chong lat
    bi xao tron, khong phai thu tu giai phau.
    """
    items = []
    for p in sorted(series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(p))
        except Exception as exc:
            print(f"    bo file hong {p.name[-16:]}: {type(exc).__name__}")
            continue
        meta = {t: getattr(ds, t, None) for t in META_TAGS}
        meta["transfer_syntax"] = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
        items.append((int(getattr(ds, "InstanceNumber", -1)), ds, meta))
    if not items:
        raise FileNotFoundError(f"khong doc duoc lat nao trong {series_dir}")

    items.sort(key=lambda t: t[0])
    shapes = {(int(d.Rows), int(d.Columns)) for _, d, _ in items}
    if len(shapes) > 1:                      # hiem, nhung co that trong dataset nay
        raise ValueError(f"series co nhieu kich thuoc: {shapes}")
    return np.stack([d.pixel_array for _, d, _ in items]), [m for _, _, m in items]


def resize_volume(arr: np.ndarray, size: int) -> np.ndarray:
    """Chuan hoa TRUOC roi resize. Lam nguoc lai thi noi suy tao gia tri moi o bien,
    keo percentile lech di."""
    out = np.empty((arr.shape[0], size, size), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.asarray(Image.fromarray(arr[i]).resize((size, size), Image.BILINEAR),
                            dtype=np.float32)
    return out


# ---------------------------------------------------------------- backbone
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def nen_thu_lai(exc: BaseException) -> bool:
    """Loi nay co dang thu lai khong?

    `HTTPError` la lop con cua `URLError`, nen phai xet no TRUOC - neu khong thi loi 404
    (that su khong co) cung bi thu lai vo ich. Dang thu lai: 5xx (su co may chu), 408 va
    429 (timeout, gioi han toc do). Con lai la loi that, bao ngay cho nguoi dung.
    """
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code in (408, 429)
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))


def build_backbone(device: str):
    """Tai backbone DINOv2 (theo bien BACKBONE) va dong bang, thu lai khi GitHub tro chung.

    `torch.hub.load` keo ma nguon tu GitHub roi trong so tu dl.fbaipublicfiles.com. Hai
    dich vu nay thinh thoang tra 5xx - do la su co TAM THOI ben ho, khong phai loi cau
    hinh. Thu lai vai lan gan nhu luc nao cung qua, nen dung de mot lan that bai lam hong
    ca phien chay dai.
    """
    import urllib.error

    model, last = None, None

    for attempt in range(6):
        try:
            model = torch.hub.load("facebookresearch/dinov2", BACKBONE)
            break
        except Exception as exc:
            last = exc
            if not nen_thu_lai(exc) or attempt == 5:
                break
            wait = min(10 * 2 ** attempt, 120)
            print(f"  tai DINOv2 that bai ({type(exc).__name__}: {exc}) "
                  f"- lan {attempt + 1}/6, cho {wait}s roi thu lai", flush=True)
            time.sleep(wait)

    if model is None:
        code = getattr(last, "code", None)
        # Phan biet ro nguyen nhan: bao nham "tat Internet" khi that ra GitHub loi 504 se
        # day nguoi dung di sua dung cai khong hong.
        if code is not None and code >= 500:
            chuan_doan = (f"GitHub tra ve HTTP {code} - su co ben phia GitHub.\n"
                          "  Internet DANG BAT (da toi duoc GitHub roi moi bi tu choi).\n"
                          "  Doi vai phut roi Run All lai thuong la qua.")
        elif code == 403:
            chuan_doan = "GitHub gioi han toc do (403). Doi 10-15 phut roi chay lai."
        else:
            chuan_doan = ("Kha nang cao la Kaggle notebook dang TAT Internet.\n"
                          "  Sua: panel ben phai -> Settings -> Internet = On "
                          "(can xac minh so dien thoai).")
        raise SystemExit(
            f"Khong tai duoc trong so DINOv2 sau 6 lan thu ({type(last).__name__}).\n"
            f"  {chuan_doan}\n"
            "  Cach khac khong phu thuoc GitHub: Add Data -> tim dataset chua trong so\n"
            "  dinov2, roi doi build_backbone() sang torch.load tu /kaggle/input do."
        ) from last

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)   # dong bang that su, khong chi dua vao no_grad
    return model


@torch.no_grad()
def extract(slices: np.ndarray, model, device: str, batch: int) -> np.ndarray:
    """(N,H,W) trong [0,1] -> (N,384). Lap 1 kenh thanh 3 de giu trong so pretrained."""
    x = torch.from_numpy(slices).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    outs = []
    for i in range(0, len(x), batch):
        chunk = x[i:i + batch].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            outs.append(model(chunk).float().cpu())
    return torch.cat(outs).numpy().astype(np.float32)


# ---------------------------------------------------------------- ghi ket qua
def ghi_ket_qua(feats: dict, headers: list[dict], plane: str) -> None:
    """Ghi dac trung va header ra /kaggle/working.

    Goi ca o checkpoint lan o cuoi, nen phai khu trung header: chay lai nhieu lan se doc
    lai file cu roi noi them, de sinh ra dong lap.
    """
    np.savez_compressed(OUT / f"features_{TEN}_{plane.lower()}.npz", **feats)
    if headers:
        hdf = pd.DataFrame(headers)
        keys = [c for c in ("SeriesInstanceUID", "InstanceNumber") if c in hdf.columns]
        if keys:
            hdf = hdf.drop_duplicates(subset=keys, keep="last")
        hdf.to_csv(OUT / f"dicom_headers_{TEN}_{plane.lower()}.csv", index=False)


# ---------------------------------------------------------------- chay
def chay_mot_mat_phang(plane: str, series: pd.DataFrame, labelled: set, device: str) -> dict:
    """Trich dac trung cho MOT mat phang. Tra ve tom tat de gop lai o cuoi."""
    gach = "=" * 70
    print(f"\n{gach}\n=== MAT PHANG: {plane} ===\n{gach}", flush=True)
    want = series[series["Anatomical_Plane"] == plane].copy()
    if ONLY_LABELLED:
        want = want[want["StudyInstanceUID"].isin(labelled)]
    # Moi study lay dung MOT series cua mat phang do
    want = want.groupby("StudyInstanceUID", as_index=False).first()
    # Study co nhan nguoi gan len dau: neu het gio thi phan quan trong nhat da xong
    want["is_gold"] = want["StudyInstanceUID"].isin(labelled)
    want = want.sort_values("is_gold", ascending=False).reset_index(drop=True)
    if MAX_STUDIES:
        want = want.head(MAX_STUDIES)
    print(f"se xu ly {len(want):,} study ({int(want.is_gold.sum())} co nhan nguoi gan), "
          f"mat phang {plane}")

    # --- Chay tiep tu lan truoc ------------------------------------------------------
    # HAI truong hop khac han nhau:
    #   1. CUNG phien (tab van mo): /kaggle/working con nguyen -> doc thang, chay tiep.
    #   2. Phien MOI (da tat may, hoac ban Save Version): /kaggle/working TRONG TRON.
    #      Luc do phai lay ban da tai ve, upload nguoc len lam Dataset roi Add Input.
    #      Khong co buoc nay thi moi lan mo phien moi la chay lai tu dau.
    ten_npz = f"features_{TEN}_{plane.lower()}.npz"
    npz_path = OUT / ten_npz

    nguon = None
    if npz_path.exists():
        nguon = npz_path
    else:
        # Tim ban da upload lam Dataset. Glob nong 2 cap cho nhanh.
        for pattern in (f"*/{ten_npz}", f"*/*/{ten_npz}"):
            for hit in Path("/kaggle/input").glob(pattern):
                nguon = hit
                break
            if nguon:
                break

    feats: dict[str, np.ndarray] = {}
    if nguon:
        with np.load(nguon) as z:
            feats = {k: z[k] for k in z.files}
        print(f"chay tiep: doc lai {len(feats):,} study da xong tu {nguon}")
    else:
        print("bat dau tu dau (khong tim thay ban ghi tam nao)")

    # Ten file header PHAI khop voi ten luc ghi (`ghi_ket_qua`). Truoc day o day con ghi
    # ten cu "dicom_headers.csv" trong khi ban ghi da doi sang co TEN + mat phang - hau
    # qua: chay tiep thi doc lai duoc DAC TRUNG nhung khong doc lai duoc HEADER, va header
    # cua nhung study da xong o lan truoc bi mat han. Khong co gi bao, chi la file CSV
    # cuoi cung thieu mot phan - va phan thieu do chinh la thu dung de phan tich theo hang
    # may o ngay 4/10.
    ten_csv = f"dicom_headers_{TEN}_{plane.lower()}.csv"
    ung_vien = [OUT / ten_csv]
    if nguon is not None:
        ung_vien.append(nguon.parent / ten_csv)

    headers: list[dict] = []
    for cand in ung_vien:
        if cand.exists():
            headers = pd.read_csv(cand).to_dict("records")
            break

    con_lai = want[~want["StudyInstanceUID"].isin(feats)].reset_index(drop=True)
    if len(con_lai) == 0:
        print("khong con study nao de xu ly - da xong tu lan truoc")
    else:
        print(f"con {len(con_lai):,} study can xu ly "
              f"({int(con_lai.is_gold.sum())} co nhan nguoi gan)")

    # Chi tai backbone khi that su con viec: neu da xong het ma GitHub dang hong thi
    # khong co ly do gi de lan chay nay that bai.
    model = build_backbone(device) if len(con_lai) else None
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    want = con_lai                    # tu day tro di chi duyet phan con lai

    for i, row in want.iterrows():
        sdir = ROOT / "train_series" / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
        try:
            vol, metas = load_series(sdir)
            arr = resize_volume(normalize_series(vol), SIZE)
            feats[row["StudyInstanceUID"]] = extract(arr, model, device, BATCH)
            headers.extend(metas)
        except Exception as exc:
            failed.append((row["StudyInstanceUID"], f"{type(exc).__name__}: {exc}"))
            continue

        if (i + 1) % 100 == 0 or i + 1 == len(want):
            done, el = i + 1, time.time() - t0
            print(f"  {done:>5}/{len(want)} study | {el/60:>5.1f} phut | "
                  f"con ~{el/done*(len(want)-done)/60:>5.1f} phut | hong {len(failed)}",
                  flush=True)
            # Ghi tam moi 300 study. Ghi CA hai file, neu khong thi khi chay lai se doc
            # duoc dac trung nhung mat header cua dung nhung study do.
            if (i + 1) % 300 == 0:
                ghi_ket_qua(feats, headers, plane)
                print(f"    (da ghi tam {len(feats):,} study)", flush=True)

    ghi_ket_qua(feats, headers, plane)

    n_slices = sum(f.shape[0] for f in feats.values())
    size_mb = (OUT / f"features_{TEN}_{plane.lower()}.npz").stat().st_size / 1024**2
    summary = {
        "plane": plane, "size": SIZE, "n_studies": len(feats),
        "n_gold": len(set(feats) & labelled), "n_slices": n_slices,
        "dim": int(next(iter(feats.values())).shape[1]) if feats else 0,
        "npz_mb": round(size_mb, 1), "minutes": round((time.time() - t0) / 60, 1),
        "failed": failed[:50], "n_failed": len(failed),
    }
    (OUT / f"extract_summary_{TEN}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== XONG {plane} ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed"}, indent=2))
    if failed:
        print(f"{len(failed)} study loi, vi du dau tien:")
        for uid, err in failed[:5]:
            print(f"  {uid[-16:]}  {err}")
    return summary


def main() -> None:
    global ROOT
    ROOT = find_root()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # dtype=str cho UID: pandas doc UID toan chu so thanh so nguyen, lam vo phep noi
    # duong dan - hoac te hon, mat so 0 dau ma khong bao gi.
    uid = {"StudyInstanceUID": str, "SeriesInstanceUID": str}
    train = pd.read_csv(ROOT / "train.csv", dtype=uid)
    series = pd.read_csv(ROOT / "train_series.csv", dtype=uid)
    labelled = set(train[train[LABELS].notna().all(axis=1)]["StudyInstanceUID"])
    print(f"train.csv {train.shape} | train_series.csv {series.shape} "
          f"| study co nhan nguoi gan: {len(labelled)}")
    print(f"se chay {len(PLANES)} mat phang: {PLANES}")

    tom_tat: dict[str, dict] = {}
    for plane in PLANES:
        # Mot mat phang hong KHONG duoc lam mat ket qua cua mat phang da xong truoc do.
        try:
            tom_tat[plane] = chay_mot_mat_phang(plane, series, labelled, device)
        except Exception as exc:
            print(f"\n!!! MAT PHANG {plane} LOI: {type(exc).__name__}: {exc}", flush=True)
            tom_tat[plane] = {"loi": f"{type(exc).__name__}: {exc}"}
        (OUT / f"extract_summary_{TEN}.json").write_text(json.dumps(tom_tat, indent=2))

    gach = "=" * 70
    print(f"\n{gach}\n=== TAT CA XONG ===\n{gach}")
    for plane, t in tom_tat.items():
        if "loi" in t:
            print(f"  {plane:<10} LOI: {t['loi'][:70]}")
        else:
            print(f"  {plane:<10} {t['n_studies']:,} study | {t['n_gold']}/58 gold | "
                  f"{t['n_slices']:,} lat | {t['npz_mb']:,.0f} MB | {t['minutes']:.0f} phut")
    print(f"\nTai ve tu tab Output: features_{TEN}_*.npz, dicom_headers_{TEN}_*.csv, "
          f"extract_summary_{TEN}.json")


if __name__ == "__main__":
    main()
