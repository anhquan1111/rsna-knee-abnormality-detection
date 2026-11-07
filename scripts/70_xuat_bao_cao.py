"""Xuat bao cao PBL: trang HTML tu chua, va (tuy chon) bo slide PNG + PPTX + PDF.

    python scripts/70_xuat_bao_cao.py           # reports/bao_cao_pbl.html (~1 phut)
    python scripts/70_xuat_bao_cao.py --slide   # them reports/slide/ (can Chrome)

Moi con so trong bao cao deu TINH LAI tu file ket qua trong reports/ (va du lieu tho de
dem phan bo nhan/ngon ngu), khong go tay. Chay lai sau moi thi nghiem la bao cao tu khop.

Khung o reports/bao_cao/: khung.html (bao cao), slide.html (21 slide 1920x1080), ve.js
(bo ve bieu do dung chung). Script thay `/*__DU_LIEU__*/` bang so lieu va `/*__VE__*/`
bang ve.js.

--slide: Chrome headless chup tung slide (#s1, #s2...) o do phan giai 2x thanh PNG, in ca
bo ra PDF, roi python-pptx goi PNG thanh PPTX kem ghi chu thuyet trinh (lay tu thuoc tinh
data-ghi-chu cua moi slide). Can extra `bao-cao` (uv sync --extra bao-cao). Tim Chrome
theo bien moi truong CHROME, roi Chrome/Edge cua Windows (khi chay trong WSL), roi PATH.

Chi xuat SO TONG HOP (dem, AUC, KTC). Khong co anh MRI hay noi dung bao cao benh nao lot
vao file - du lieu cuoc thi khong duoc phat tan lai.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rsna_knee.config import LABELS, REPORTS_DIR, SEED
from rsna_knee.manifest import build_study_manifest, gold_subset, load_raw
from rsna_knee.metrics import macro_auc
from rsna_knee.oof import gold_theo_fold, ma_thu_tu, nap_oof

L = list(LABELS)
KHUNG = REPORTS_DIR / "bao_cao" / "khung.html"
KHUNG_SLIDE = REPORTS_DIR / "bao_cao" / "slide.html"
VE = REPORTS_DIR / "bao_cao" / "ve.js"
RA = REPORTS_DIR / "bao_cao_pbl.html"
RA_SLIDE = REPORTS_DIR / "slide"
TEN = ["mp_sagittal", "mp_axial", "mp_coronal", "chung_s224", "chung_b336"]
EPOCH_CHOT = 9
CV_CU = {"mp_sagittal": 0.7106, "mp_axial": 0.7691, "mp_coronal": 0.7394,
         "chung_s224": 0.7698, "chung_b336": 0.7481}
NGON_NGU = {"en": "Anh", "es": "Tây Ban Nha", "tr": "Thổ Nhĩ Kỳ", "hr": "Croatia",
            "el": "Hy Lạp", "de": "Đức", "bg": "Bulgaria", "nl": "Hà Lan", "fr": "Pháp"}


def hang(p):
    return np.column_stack([rankdata(p[:, j]) for j in range(p.shape[1])])


def auc(y, p):
    return macro_auc(y, p, L).macro_auc


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        r = macro_auc(y[i], p[i], L)
        if r.n_scored == len(L):
            b.append(r.macro_auc)
    return [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])]


def boot_ci_nhan(y, p, j, n=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i, j])) == 2:
            b.append(roc_auc_score(y[i, j], p[i, j]))
    return [round(float(v), 3) for v in np.percentile(b, [2.5, 97.5])]


def best_tung_fold(P, y, fold):
    out, chon = np.empty_like(P[0]), []
    for k in np.unique(fold):
        m = fold == k
        s = []
        for e in range(P.shape[0]):
            try:
                s.append(auc(y[m], P[e, m]))
            except ValueError:
                s.append(-1.0)
        e = int(np.argmax(s))
        chon.append(e + 1)
        out[m] = P[e, m]
    return out, chon


def file_seed(t, s):
    return REPORTS_DIR / (f"day13_epoch_{t}.npz" if s == SEED else f"day13_epoch_{t}_seed{s}.npz")


def main() -> None:
    d: dict = {}

    # ---------------- du lieu
    train, series = load_raw()
    man = build_study_manifest(train, series)
    gold = gold_subset(man)
    langs = pd.read_csv(REPORTS_DIR / "_langs.csv")["language"].value_counts()
    d["du_lieu"] = {
        "n_study": int(len(man)), "n_series": int(len(series)), "n_gold": int(len(gold)),
        "series_moi_study": round(float(series.groupby("StudyInstanceUID").size().mean()), 2),
        "nhan_duong": [{"nhan": k, "n": int(gold[k].sum())} for k in L],
        "ngon_ngu": [{"ma": k, "ten": NGON_NGU.get(k, k), "n": int(v)} for k, v in langs.items()],
    }

    # ---------------- chon epoch (ngay 13)
    y, fold, uid = gold_theo_fold()
    z = {t: {"per_ep": nap_oof(file_seed(t, SEED), uid, fold)} for t in TEN}
    n_ep = min(v["per_ep"].shape[0] for v in z.values())
    gop_e = lambda e: np.mean([hang(z[t]["per_ep"][e - 1]) for t in TEN], axis=0)

    d["duong_epoch"] = {
        "gop": [round(auc(y, gop_e(e)), 4) for e in range(1, n_ep + 1)],
        "thanh_phan": {t: [round(auc(y, z[t]["per_ep"][e]), 4) for e in range(n_ep)] for t in TEN},
        "chung_s224_dai": [round(auc(y, z["chung_s224"]["per_ep"][e]), 4)
                           for e in range(z["chung_s224"]["per_ep"].shape[0])],
    }
    kg = json.loads((REPORTS_DIR / "day13_khong_gold_chung_s224.json").read_text())
    d["khong_gold"] = {"auc": [round(v, 4) for v in kg["auc_theo_epoch"]], "dinh": kg["epoch_dinh"]}

    best = {}
    epoch_chon = {}
    for t in TEN:
        best[t], epoch_chon[t] = best_tung_fold(z[t]["per_ep"][:n_ep], y, fold)
    gop_best = np.mean([hang(best[t]) for t in TEN], axis=0)
    g9, g_dai = gop_e(EPOCH_CHOT), gop_e(n_ep)

    rng = np.random.default_rng(0)
    chenh = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        try:
            chenh.append(auc(y[i], g9[i]) - auc(y[i], g_dai[i]))
        except ValueError:
            pass
    d["chon_epoch"] = {
        "cu": round(auc(y, gop_best), 4), "chot": round(auc(y, g9), 4),
        "dai": round(auc(y, g_dai), 4), "n_ep_dai": n_ep, "epoch_chot": EPOCH_CHOT,
        "ci_chot": boot_ci(y, g9), "cv_cu_bao_cao": 0.8112,
        "chenh_chot_dai": round(auc(y, g9) - auc(y, g_dai), 4),
        "ci_chenh": [round(float(v), 4) for v in np.percentile(chenh, [2.5, 97.5])],
        "p_duong": round(float(np.mean(np.array(chenh) > 0)), 3),
        "epoch_chon_tung_fold": epoch_chon,
    }
    d["thanh_phan"] = [{"ten": t, "cv_cu": CV_CU[t],
                        "cv_chot": round(auc(y, z[t]["per_ep"][EPOCH_CHOT - 1]), 4)} for t in TEN]

    # ---------------- nhieu seed
    seeds = sorted({int(f.stem.split("_seed")[1]) for f in REPORTS_DIR.glob("day13_epoch_*_seed*.npz")})
    seeds = [SEED] + [s for s in seeds if all(file_seed(t, s).exists() for t in TEN)]
    moi_seed, tat_ca = [], []
    for s in seeds:
        hs = [hang(nap_oof(file_seed(t, s), uid, fold)[EPOCH_CHOT - 1]) for t in TEN]
        tat_ca += hs
        moi_seed.append({"seed": s, "auc": round(auc(y, np.mean(hs, axis=0)), 4)})
    gop_cuoi = np.mean(tat_ca, axis=0)
    d["seed"] = {"moi_seed": moi_seed, "gop": round(auc(y, gop_cuoi), 4),
                 "ci_gop": boot_ci(y, gop_cuoi), "n": len(seeds)}
    if len(seeds) > 1:
        d["seed"]["do_lech"] = round(float(np.std([m["auc"] for m in moi_seed], ddof=1)), 4)

    # AUC tung nhan cua model cuoi (gop moi seed da co)
    r = macro_auc(y, gop_cuoi, L)
    d["tung_nhan"] = sorted([
        {"nhan": k, "auc": round(r.per_label[k], 3), "ci": boot_ci_nhan(y, gop_cuoi, j),
         "n_duong": int(np.nansum(y[:, j]))}
        for j, k in enumerate(L)], key=lambda v: -v["auc"])

    # ---------------- giai doan 2: CNN 2.5D (notebook 81, cham bang 82)
    # So tong (AUC, KTC, chenh bat cap) lay nguyen tu day14_cnn_25d.json de khop README va
    # TRINH_BAY.md; KTC tung nhan tinh lai o day tu du doan (file khong mang nhan).
    f_cnn = REPORTS_DIR / "day14_cnn_25d.json"
    if f_cnn.exists():
        kc = json.loads(f_cnn.read_text())
        with np.load(REPORTS_DIR / "day14_cnn_du_doan.npz") as zc:
            if str(zc["uid_md5"]) != ma_thu_tu(uid):
                raise SystemExit("day14_cnn_du_doan.npz: thu tu ca khac phep chia hien tai")
            p_cnn = np.mean([hang(zc[f"{t}__ema"][-1]) for t in kc["tung_model"]], axis=0)
        rc = macro_auc(y, p_cnn, L)
        d["cnn"] = {
            "auc": kc["cnn"]["auc"], "ci": kc["cnn"]["ci"],
            "gop_50_50": kc["gop_50_50"], "chenh": kc["cnn_tru_dino"],
            "mo_hinh": [{"ten": t, "backbone": m["backbone"], "auc": m["auc"], "ci": m["ci"], "gio": m["gio"],
                         "epoch": len(m["theo_epoch"]), "n_train": m["n_train"],
                         "gold": [e["ema"] for e in m["theo_epoch"]],
                         "val_llm": [round(e["val_llm"], 4) for e in m["theo_epoch"]]}
                        for t, m in kc["tung_model"].items()],
            "tung_nhan": sorted([{"nhan": k, "cnn": round(rc.per_label[k], 3), "ci": boot_ci_nhan(y, p_cnn, j),
                                  "dino": round(r.per_label[k], 3)} for j, k in enumerate(L)],
                                key=lambda v: -(v["cnn"] - v["dino"])),
        }

    # ---------------- ngay 15: nhan LLM tu gan (script 91) va model cuoi gop 6 CNN (script 93)
    f_nhan, f_cuoi = REPORTS_DIR / "day15_nhan_llm.json", REPORTS_DIR / "day15_model_cuoi.json"
    if f_nhan.exists():
        kn = json.loads(f_nhan.read_text())
        d["nhan_llm"] = {k: {"ten": v["ten"], "auc": v["macro_auc"], "ci": v["ci"], "khop": v["nguong_05"]["khop"],
                             "tung_benh": v["tung_benh"]} for k, v in kn["nguon"].items()}
        d["nhan_llm"]["chenh_cong_khai"] = kn["tu_gan_tru_cong_khai"]
        d["nhan_llm"]["khop_hai_bo"] = kn["hai_bo_llm_ngoai_gold"]["khop_nguong_05"]
    if f_cuoi.exists():
        kf = json.loads(f_cuoi.read_text())
        d["cuoi"] = {"nhom": {k.split(" ")[0]: {"auc": v["auc"], "ci": v["ci"], "model": v["model"]}
                              for k, v in kf["nhom"].items()},
                     "so_sanh": {k.split(" ")[0]: v for k, v in kf["so_sanh"].items()},
                     "tung_model": [{"so": int(s), **v, **{x: kf["danh_sach"][int(s) - 1][x] for x in ("mo_hinh", "nhan")}}
                                    for s, v in kf["tung_model"].items()],
                     "tung_benh": kf["nhom"]["gop_6_cnn (chinh)"]["tung_benh"]}

    # ---------------- cac ket qua cu (doc tu file da ghi)
    ls = json.loads((REPORTS_DIR / "day10a_label_source.json").read_text())
    d["nguon_nhan"] = {
        "A": round(ls["A_gold_only"], 4), "B": round(ls["B_tu_dien"], 4),
        "D": round(ls["D_llm"], 4), "E": round(ls["E_llm_bit"], 4),
        "auc_tu_dien": round(ls["auc_nguon_tren_gold"]["tu_dien"], 4),
        "auc_llm": round(ls["auc_nguon_tren_gold"]["llm"], 4),
        "chenh": round(ls["chenh_D_B"], 4), "ci": [round(v, 4) for v in ls["ci95"]],
    }
    ps = json.loads((REPORTS_DIR / "day12_phuong_sai_seed.json").read_text())
    d["seed_cu"] = {"diem": [round(v, 4) for v in ps["diem_tung_seed"]],
                    "do_lech": round(ps["do_lech_chuan"], 4), "be_rong": round(ps["be_rong"], 4)}
    ea = json.loads((REPORTS_DIR / "day10_error_analysis.json").read_text())
    d["hang_may"] = [{"hang": h["hang"], "n": h["n"], "auc": round(h["auc"], 4),
                      "ci": [round(v, 3) for v in h["ci95"]]}
                     for h in ea["theo_hang"] if h.get("auc") is not None]
    wl = pd.read_csv(REPORTS_DIR / "weak_label_scores.csv")
    d["tu_dien"] = [{"nhan": r_.label, "phu": round(r_.coverage, 3),
                     "precision": round(r_.precision, 3)} for r_ in wl.itertuples()]

    html = dien(KHUNG, d)
    # Khung viet theo dang trang Artifact (khong co doctype). Mo truc tiep tren trinh duyet
    # thi thieu doctype se roi vao quirks mode, nen ban o may duoc boc them.
    RA.write_text('<!doctype html>\n<html lang="vi">\n' + html + "\n</html>\n", encoding="utf-8")
    if "--artifact" in sys.argv:
        ra2 = Path(sys.argv[sys.argv.index("--artifact") + 1])
        ra2.write_text(html, encoding="utf-8")
        print(f"Ban Artifact: {ra2}")
    print(f"Da ghi {RA.relative_to(REPORTS_DIR.parent)} ({RA.stat().st_size/1024:.0f} KB)"
          f" | {len(seeds)} seed | bo gop cuoi {d['seed']['gop']:.4f} {d['seed']['ci_gop']}")
    if "--slide" in sys.argv:
        xuat_slide(d)


def dien(khung: Path, d: dict) -> str:
    """Thay cho dat so lieu va bo ve vao khung."""
    html = khung.read_text(encoding="utf-8")
    for cho in ("/*__DU_LIEU__*/", "/*__VE__*/"):
        if cho not in html:
            raise SystemExit(f"{khung.name} thieu cho dat {cho}")
    return (html.replace("/*__DU_LIEU__*/", "const D = " + json.dumps(d, ensure_ascii=False) + ";")
                .replace("/*__VE__*/", VE.read_text(encoding="utf-8")))


def tim_chrome() -> tuple[str, bool]:
    """(duong dan, co phai ban Windows). Ban Windows can duong dan dang \\wsl.localhost."""
    if os.environ.get("CHROME"):
        p = os.environ["CHROME"]
        return p, p.endswith(".exe")
    for p in ("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
              "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"):
        if Path(p).exists():
            return p, True
    for ten in ("google-chrome", "chromium", "chromium-browser"):
        if shutil.which(ten):
            return shutil.which(ten), False
    raise SystemExit("Khong tim thay Chrome/Edge. Dat bien CHROME=<duong dan> roi chay lai.")


def xuat_slide(d: dict) -> None:
    from PIL import Image
    try:
        from pptx import Presentation
        from pptx.util import Emu
    except ImportError:
        raise SystemExit("Thieu python-pptx: uv sync --extra bao-cao")

    RA_SLIDE.mkdir(exist_ok=True)
    html = dien(KHUNG_SLIDE, d)
    trang = RA_SLIDE / "slide.html"
    trang.write_text('<!doctype html>\n<html lang="vi">\n' + html + "\n</html>\n", encoding="utf-8")
    ghi_chu = [re.sub(r"\s+", " ", g).strip()
               for g in re.findall(r'<section class="slide[^"]*" data-ghi-chu="([^"]*)"', html)]
    n = len(re.findall(r'<section class="slide', html))
    if len(ghi_chu) != n:
        raise SystemExit(f"{n} slide nhung chi {len(ghi_chu)} ghi chu - slide nao cung phai co data-ghi-chu")

    chrome, la_win = tim_chrome()
    doi = (lambda p: subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True,
                                    check=True).stdout.strip()) if la_win else str
    url = (lambda p: "file:///" + doi(p).replace("\\", "/")) if la_win else (lambda p: p.resolve().as_uri())
    ho_so = Path(tempfile.mkdtemp(prefix="chrome_bao_cao_"))
    chung = [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
             f"--user-data-dir={doi(ho_so)}", "--virtual-time-budget=8000"]
    try:
        for cu in RA_SLIDE.glob("slide_*.png"):
            cu.unlink()
        anh = []
        for i in range(1, n + 1):
            png = RA_SLIDE / f"slide_{i:02d}.png"
            for _ in range(3):   # Chrome tu WSL thinh thoang thoat som ma khong ghi file
                subprocess.run(chung + ["--window-size=1920,1080", "--force-device-scale-factor=2",
                                        f"--screenshot={doi(png)}", url(trang) + f"#s{i}"],
                               capture_output=True, timeout=120)
                if png.exists():
                    break
            if not png.exists():
                raise SystemExit(f"Chrome khong chup duoc slide {i}")
            with Image.open(png) as im:
                if im.size != (3840, 2160):
                    raise SystemExit(f"slide {i}: anh {im.size}, mong doi 3840x2160")
            anh.append(png)
        pdf = RA_SLIDE / "bao_cao_pbl.pdf"
        subprocess.run(chung + ["--no-pdf-header-footer", f"--print-to-pdf={doi(pdf)}", url(trang)],
                       capture_output=True, timeout=180)
    finally:
        shutil.rmtree(ho_so, ignore_errors=True)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)   # 16:9, 13.333 x 7.5 in
    trong = prs.slide_layouts[6]
    for png, gc in zip(anh, ghi_chu):
        s = prs.slides.add_slide(trong)
        s.shapes.add_picture(str(png), 0, 0, prs.slide_width, prs.slide_height)
        s.notes_slide.notes_text_frame.text = gc
    pptx = RA_SLIDE / "bao_cao_pbl.pptx"
    prs.save(pptx)
    print(f"Slide: {n} PNG 3840x2160 + {pptx.name} ({pptx.stat().st_size/1024**2:.1f} MB)"
          + (f" + {pdf.name} ({pdf.stat().st_size/1024**2:.1f} MB)" if pdf.exists() else " (PDF loi)"))


if __name__ == "__main__":
    main()
