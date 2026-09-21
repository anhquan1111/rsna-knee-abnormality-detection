"""Ngay 3: mo file DICOM that va do cac quyet dinh tien xu ly.

Chay:  python scripts/30_inspect_dicom.py
Yeu cau: da tai anh bang scripts/04_download_from_listing.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rsna_knee.config import DATA_RAW
from rsna_knee.dicom_io import (
    load_series,
    normalize_per_slice,
    normalize_series,
    read_header,
    resize_volume,
    scan_series_dir,
)

IMAGES = DATA_RAW / "images"


def main() -> None:
    if not IMAGES.exists():
        raise SystemExit(f"Chua co anh o {IMAGES}. Chay scripts/04_download_from_listing.py truoc.")

    studies = sorted(d for d in IMAGES.iterdir() if d.is_dir())
    series_dirs = [s for st in studies for s in sorted(st.iterdir()) if s.is_dir()]
    all_dcm = list(IMAGES.rglob("*.dcm"))
    total_mb = sum(p.stat().st_size for p in all_dcm) / 1024**2
    print(f"Da tai: {len(studies)} study | {len(series_dirs)} series | "
          f"{len(all_dcm):,} file | {total_mb:,.1f} MB")

    print("\n=== 1. Transfer syntax that su co trong du lieu ===")
    t0 = time.time()
    heads = [read_header(p) for p in all_dcm]
    dt_meta = time.time() - t0
    for name, n in Counter(h["transfer_syntax_name"] for h in heads).most_common():
        print(f"  {name:<40} {n:>5} file")
    print(f"  quet {len(all_dcm):,} header mat {dt_meta:.1f}s "
          f"({dt_meta/len(all_dcm)*1000:.1f} ms/file)")

    print("\n=== 2. Doc header (stop_before_pixels) vs giai nen ca anh ===")
    sample = all_dcm[:40]
    t0 = time.time()
    for p in sample:
        read_header(p)
    t_head = time.time() - t0
    import pydicom
    t0 = time.time()
    for p in sample:
        _ = pydicom.dcmread(str(p)).pixel_array
    t_full = time.time() - t0
    print(f"  chi header : {t_head/len(sample)*1000:>7.1f} ms/file")
    print(f"  ca pixel   : {t_full/len(sample)*1000:>7.1f} ms/file")
    print(f"  -> quet manifest ma giai nen anh thi cham gap {t_full/max(t_head,1e-9):.1f}x")

    print("\n=== 3. Metadata co dung de chong ro ri tang 3 khong? ===")
    manu = Counter(str(h["Manufacturer"]) for h in heads)
    model = Counter(str(h["ManufacturerModelName"]) for h in heads)
    field = Counter(str(h["MagneticFieldStrength"]) for h in heads)
    print(f"  Manufacturer          : {dict(manu)}")
    print(f"  ManufacturerModelName : {len(model)} gia tri -> {list(model)[:4]}")
    print(f"  MagneticFieldStrength : {dict(field)}")
    present = sum(1 for h in heads if h["Manufacturer"] not in (None, "", "None"))
    print(f"  -> {present}/{len(heads)} file con giu tag may chup")

    print("\n=== 4. Kich thuoc va kieu du lieu ===")
    shapes = Counter((int(h["Rows"]), int(h["Columns"])) for h in heads)
    print(f"  kich thuoc anh: {dict(shapes)}")
    print(f"  PixelRepresentation: {dict(Counter(h['PixelRepresentation'] for h in heads))} "
          f"(1 = signed -> pixel_array ra int16)")
    n_slice = Counter(len(list(s.glob('*.dcm'))) for s in series_dirs)
    print(f"  so lat moi series: {dict(sorted(n_slice.items()))}")
    print("  -> so lat khac nhau chinh la ly do phai tu viet collate_fn")

    print("\n=== 5. Cuong do MRI khong tuyet doi ===")
    vols = []
    for sd in series_dirs[:4]:
        v = load_series(sd)
        vols.append(v)
        print(f"  {sd.parent.name[-12:]}  shape {v.pixels.shape}  dtype {v.pixels.dtype}  "
              f"min {v.pixels.min():>6} max {v.pixels.max():>6} "
              f"p99 {np.percentile(v.pixels, 99):>8.0f}")
    hi = [float(np.percentile(v.pixels, 99)) for v in vols]
    print(f"  -> p99 giua cac series lech nhau toi {max(hi)/min(hi):.2f}x")

    print("\n=== 6. Ba cach chuan hoa tren cung mot series ===")
    v = max(vols, key=lambda x: x.pixels.shape[0])   # lay series nhieu lat nhat
    raw255 = v.pixels.astype(np.float32) / 255.0
    per_series = normalize_series(v.pixels)
    per_slice = normalize_per_slice(v.pixels)
    print(f"  series dung de do: {v.pixels.shape[0]} lat")
    print(f"  chia /255        : dai [{raw255.min():.2f}, {raw255.max():.2f}] "
          f"<- vuot khoi [0,1], anh MRI khong phai anh 8-bit")
    print(f"  percentile series: dai [{per_series.min():.2f}, {per_series.max():.2f}]")
    print(f"  percentile slice : dai [{per_slice.min():.2f}, {per_slice.max():.2f}]")

    # Do dung thu bi mat: do SANG TUONG DOI giua cac lat. Sau khi chuan hoa theo tung lat,
    # moi lat bi ep ve cung mot dai -> p99 cua moi lat deu bang 1.0, do lech bang 0.
    p99_series = np.percentile(per_series, 99, axis=(1, 2))
    p99_slice = np.percentile(per_slice, 99, axis=(1, 2))
    print("\n  Do lech chuan cua p99 GIUA CAC LAT (do sang tuong doi giua cac lat):")
    print(f"    percentile ca series: {p99_series.std():.4f}  "
          f"(dai {p99_series.min():.3f}-{p99_series.max():.3f}) <- con giu")
    print(f"    percentile tung lat : {p99_slice.std():.4f}  "
          f"(dai {p99_slice.min():.3f}-{p99_slice.max():.3f}) <- bi xoa sach")
    print("  -> chuan hoa tung lat ep moi lat ve cung mot dai, xoa mat do sang tuong doi")
    print("     giua cac lat - trong khi do chinh la tin hieu phan biet vung benh voi mo lanh")

    print("\n=== 7. Chuan hoa truoc hay resize truoc? ===")
    a = resize_volume(normalize_series(v.pixels), 224)
    b = normalize_series(resize_volume(v.pixels.astype(np.float32), 224))
    print(f"  chuan hoa -> resize: dai [{a.min():.3f}, {a.max():.3f}]")
    print(f"  resize -> chuan hoa: dai [{b.min():.3f}, {b.max():.3f}]")
    print(f"  lech trung binh tuyet doi: {np.abs(a-b).mean():.5f}")

    print("\n=== 8. Cache co dang khong? ===")
    # Do lap nhieu lan roi lay trung vi: mot lan do don le bi nhieu boi cache cua he dieu
    # hanh va se cho ket luan nguoc han.
    sd = max(series_dirs, key=lambda d: len(list(d.glob("*.dcm"))))
    n_rep = 5

    def bench(fn):
        ts = []
        for _ in range(n_rep):
            t0 = time.time(); fn(); ts.append(time.time() - t0)
        return float(np.median(ts))

    t_decode = bench(lambda: resize_volume(normalize_series(load_series(sd).pixels), 224))
    arr = resize_volume(normalize_series(load_series(sd).pixels), 224)
    tmp = DATA_RAW / "_bench.npy"
    np.save(tmp, arr)
    t_cache = bench(lambda: np.load(tmp))

    src_mb = sum(p.stat().st_size for p in sd.glob("*.dcm")) / 1024**2
    cache_mb = tmp.stat().st_size / 1024**2
    n_slices = len(list(sd.glob("*.dcm")))
    print(f"  series {n_slices} lat, do {n_rep} lan lay trung vi")
    print(f"  doc DICOM + chuan hoa + resize : {t_decode*1000:>8.1f} ms | {src_mb:>6.1f} MB tren dia")
    print(f"  doc lai .npy da xu ly san      : {t_cache*1000:>8.1f} ms | {cache_mb:>6.1f} MB")
    print(f"  -> nhanh gap {t_decode/max(t_cache,1e-9):.1f}x, dia {src_mb/cache_mb:.1f}x nho hon")
    print(f"  LUU Y: du lieu nay la Explicit VR Little Endian (KHONG nen). Muc loi cua cache")
    print(f"  phu thuoc chuan nen: file JPEG 2000 phai giai nen nen cache loi hon nhieu.")
    tmp.unlink()

    print("\n=== 9. Thu tu lat: ten file KHONG phai thu tu giai phau ===")
    by_name = [read_header(p)["InstanceNumber"] for p in sorted(sd.glob("*.dcm"))]
    print(f"  InstanceNumber theo thu tu ten file: {by_name[:12]}...")
    print(f"  da sap dung thu tu chua? {by_name == sorted(by_name)}")
    print("  -> phai sap theo InstanceNumber, khong duoc tin sorted(glob)")


if __name__ == "__main__":
    main()
