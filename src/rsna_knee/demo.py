"""Chay model CNN 2.5D (notebook 81) tren MOT ca chup, cho web demo (scripts/95_web_demo.py).

Tien xu ly va kien truc lay NGUYEN tu `scripts/81_kaggle_train_25d.py` (nap bang importlib),
khong chep lai: web demo va luc train dung cung mot ham doc DICOM, cung mot cua so 2.5D.
Lech o day la lech am tham - model van ra so, chi la so sai.

Hai nguon ca:
  * ca mau: 8 ca gold dong goi san thanh zip DICOM (KHONG ca nao vao train CNN), kem nhan
    bac si de doi chieu;
  * ca tai len: thu muc/zip DICOM. Mat phang suy tu ImageOrientationPatient, series nhay
    dich doan tu SeriesDescription (khong co train_series.csv nhu luc train).
"""
from __future__ import annotations

import base64
import importlib.util
import io
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS

REPO = Path(__file__).resolve().parents[2]
THU_MUC_MODEL = REPO / "data" / "web_demo" / "model"
TEN_VI = {"ACL": "Dây chằng chéo trước (ACL)", "MCL": "Dây chằng bên trong (MCL)",
          "Medial Meniscus": "Sụn chêm trong", "Lateral Meniscus": "Sụn chêm ngoài",
          "Medial OA": "Thoái hoá khoang trong", "Lateral OA": "Thoái hoá khoang ngoài",
          "PF OA": "Thoái hoá bánh chè – đùi", "Effusion": "Tràn dịch khớp",
          "Synovitis": "Viêm màng hoạt dịch", "Baker's": "Nang Baker",
          "Contusion": "Dập xương", "Fracture": "Gãy xương"}
NHAY_DICH = ("T2", "PD", "DP", "STIR", "FS", "FAT", "SPAIR", "SPIR", "TIRM", "SAT", "DIXON W")


def _nap_81():
    spec = importlib.util.spec_from_file_location("nb81", REPO / "scripts" / "81_kaggle_train_25d.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class MayDoan:
    """Nap 2 model mot lan; `doan()` khoa GPU de hai yeu cau khong chay chong."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.nb = _nap_81()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []
        for f in sorted(THU_MUC_MODEL.glob("*/model_cuoi.pt")):
            backbone = f.parent.name.rsplit("_s", 1)[0]
            net = self.nb.Mang25D(backbone, pretrained=False)
            net.load_state_dict({k: v.float() for k, v in torch.load(f, map_location="cpu").items()})
            self.models.append((f.parent.name, net.to(self.dev).eval()))
        if not self.models:
            raise SystemExit(f"Khong thay */model_cuoi.pt trong {THU_MUC_MODEL}. Tai ve bang:\n"
                             "  kaggle kernels output votrananhquan/rsna-knee-train-25d "
                             f"-p {THU_MUC_MODEL} --file-pattern 'model_cuoi\\.pt$'")
        self.khoa = threading.Lock()
        # Chay nong mot lan: lan dau cuDNN chon thuat toan mat ~5 giay, khong de nguoi xem cho
        nb = self.nb
        self.doan(np.zeros((len(nb.PLANES), nb.DEPTH, nb.SIZE, nb.SIZE), np.uint8), np.ones(len(nb.PLANES), bool))

    # ------------------------------------------------------------ doc anh
    def doc_ca(self, series_theo_mat: dict[str, list[Path]]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """(x uint8 (3,D,S,S), co bool (3,), thong tin tung mat phang) - dung `mot_ca` cua 81."""
        nb = self.nb
        x = np.zeros((len(nb.PLANES), nb.DEPTH, nb.SIZE, nb.SIZE), dtype=np.uint8)
        co = np.zeros(len(nb.PLANES), dtype=bool)
        tt = []
        for j, pl in enumerate(nb.PLANES):
            dong = {"mat_phang": pl, "co": False, "n_lat_goc": 0, "series": ""}
            for d in series_theo_mat.get(pl, []):
                try:
                    x[j], n = nb.doc_series(d)
                    co[j] = True
                    mo_ta = getattr(d, "mo_ta", "")
                    if not mo_ta or "dummy" in mo_ta.lower():   # du lieu cuoc thi da xoa ten series
                        mo_ta = f"{n} lát gốc"
                    dong.update(co=True, n_lat_goc=int(n), series=mo_ta)
                    break
                except Exception:
                    continue
            tt.append(dong)
        return x, co, tt

    # ------------------------------------------------------------ du doan
    def doan(self, x: np.ndarray, co: np.ndarray) -> dict:
        torch, nb = self.torch, self.nb
        D = x.shape[1]
        tam = np.arange(D)
        cua_so = np.stack([np.clip(tam - 1, 0, D - 1), tam, np.clip(tam + 1, 0, D - 1)], 1)
        w = torch.from_numpy(x[:, cua_so])[None].to(self.dev)        # (1, P, D, 3, S, S)
        c = torch.from_numpy(co)[None].to(self.dev)
        p_moi, chu_y = {}, []
        with self.khoa, torch.no_grad():
            t0 = time.perf_counter()
            for ten, net in self.models:
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev != "cpu"):
                    logit = net(nb.vao_mang(nb.chuan_hoa(w)), c).float()
                p_moi[ten] = torch.sigmoid(logit)[0].cpu().numpy()
                chu_y.append(net.chu_y[0].float().cpu().numpy())               # (P*D, L)
            ms = (time.perf_counter() - t0) * 1000
        p = np.mean(list(p_moi.values()), axis=0)
        a = np.mean(chu_y, axis=0).reshape(len(nb.PLANES), D, len(LABELS))   # (P, D, L)
        return {"p": p, "p_moi": p_moi, "chu_y": a, "ms": ms}

    # ------------------------------------------------------------ dong goi cho web
    def ket_qua(self, x, co, tt, nhan_bac_si: dict | None = None) -> dict:
        from PIL import Image

        d = self.doan(x, co)
        lat = []
        for j in range(x.shape[0]):
            anh = []
            for t in range(x.shape[1]):
                b = io.BytesIO()
                Image.fromarray(x[j, t]).save(b, "JPEG", quality=82)
                anh.append(base64.b64encode(b.getvalue()).decode())
            lat.append(anh if co[j] else [])
        benh = []
        for k, ma in enumerate(LABELS):
            a = d["chu_y"][:, :, k]
            j, t = np.unravel_index(int(np.argmax(a)), a.shape)
            benh.append({
                "ma": ma, "ten": TEN_VI[ma], "p": round(float(d["p"][k]), 4),
                "p_tung_model": {m: round(float(v[k]), 4) for m, v in d["p_moi"].items()},
                "nhan_bac_si": None if not nhan_bac_si or pd.isna(nhan_bac_si.get(ma)) else int(nhan_bac_si[ma]),
                "chu_y": [[round(float(v), 4) for v in a[jj]] for jj in range(a.shape[0])],
                "lat_chinh": {"mat_phang": int(j), "lat": int(t), "trong_so": round(float(a[j, t]), 4)},
            })
        return {"mat_phang": [{**tt[j], "lat": lat[j]} for j in range(len(tt))], "benh": benh,
                "model": [m for m, _ in self.models], "ms": round(d["ms"]), "thiet_bi": self.dev}


# ---------------------------------------------------------------- ca mau
THU_MUC_MAU = REPO / "data" / "web_demo" / "ca_mau"


def ca_mau() -> list[dict]:
    """8 ca gold dong goi san thanh zip DICOM (3 mat phang) - KHONG ca nao vao train CNN.

    Zip tao bang notebook CPU `votrananhquan/rsna-knee-ca-mau-demo` (o may chi co 1 mat phang
    moi ca). Ca mau di DUNG duong cua ca tai len, chi them nhan bac si de doi chieu."""
    ds = THU_MUC_MAU / "danh_sach.txt"
    if not ds.exists():
        return []
    train = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    ra = []
    for k, uid in enumerate(ds.read_text().split(), 1):
        z = THU_MUC_MAU / f"ca_mau_{k}.zip"
        if z.exists() and uid in train.index:
            g = train.loc[uid, list(LABELS)]
            ra.append({"id": f"mau{k}", "ten": f"Ca mẫu {k}", "zip": z, "uid": uid, "nhan_bac_si": g.to_dict(),
                       "n_benh": int(g.sum()), "mo_ta": ", ".join(TEN_VI[m] for m in LABELS if g[m] == 1) or "Không bệnh nào"})
    return ra


# ---------------------------------------------------------------- ca tai len
class _ThuMuc(type(Path())):
    mo_ta = ""


def tach_series_tai_len(tep: list[tuple[str, bytes]]) -> tuple[dict[str, list[Path]], Path, list[str]]:
    """Gom file DICOM (hoac zip) theo series, suy mat phang, xep series nhay dich len truoc.

    Tra (series_theo_mat, thu_muc_tam, canh_bao). Goi `shutil.rmtree(thu_muc_tam)` khi xong.
    """
    import zipfile

    import pydicom

    tam = Path(tempfile.mkdtemp(prefix="rsna_demo_"))
    goc = tam / "goc"
    goc.mkdir()
    k = 0
    for ten, du_lieu in tep:
        if ten.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(du_lieu)) as z:
                for m in z.infolist():
                    if not m.is_dir():
                        (goc / f"{k:06d}.dcm").write_bytes(z.read(m))
                        k += 1
        else:
            (goc / f"{k:06d}.dcm").write_bytes(du_lieu)
            k += 1

    series: dict[str, dict] = {}
    for f in goc.iterdir():
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            uid = str(ds.SeriesInstanceUID)
        except Exception:
            continue
        s = series.setdefault(uid, {"tep": [], "mo_ta": str(getattr(ds, "SeriesDescription", "")),
                                    "huong": getattr(ds, "ImageOrientationPatient", None)})
        s["tep"].append(f)
    canh_bao = [] if series else ["Không đọc được file DICOM nào."]

    theo_mat: dict[str, list[tuple[int, Path]]] = {}
    for uid, s in series.items():
        if s["huong"] is None or len(s["tep"]) < 3:
            continue
        r, c = np.array(s["huong"][:3], float), np.array(s["huong"][3:], float)
        pl = ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(np.cross(r, c))))]
        d = _ThuMuc(tam / uid)
        d.mkdir()
        for i, f in enumerate(s["tep"]):
            (d / f"{i:05d}.dcm").symlink_to(f)
        d.mo_ta = s["mo_ta"]
        mo_ta = s["mo_ta"].upper()
        diem = sum(t in mo_ta for t in NHAY_DICH) - 2 * ("T1" in mo_ta)
        theo_mat.setdefault(pl, []).append((-diem, d))
    for pl in ("Sagittal", "Coronal", "Axial"):
        if pl not in theo_mat:
            canh_bao.append(f"Thiếu mặt phẳng {pl}: model vẫn chạy nhưng kém tin cậy hơn.")
    return ({pl: [d for _, d in sorted(v, key=lambda t: t[0])] for pl, v in theo_mat.items()},
            tam, canh_bao)


def don(tam: Path) -> None:
    shutil.rmtree(tam, ignore_errors=True)
