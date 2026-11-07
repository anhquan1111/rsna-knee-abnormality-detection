"""CHAY TREN KAGGLE (GPU T4 x2, Internet bat de tai trong so pretrained) - train CNN 2.5D.

Tu doc DICOM cua cuoc thi ra cache tren o dia cuc bo (~60 phut, 4 CPU), roi train TOAN BO
mang - backbone va head - tren nhan LLM mem cua 4.349 ca, roi cham tren 58 ca bac si gan
nhan. 58 ca nay KHONG bao gio vao train, nen diem cua chung la diem ngoai mau, khong can
chia fold.

VI SAO LAM VAY (tu cac loi giai cong khai cua cuoc thi)
======================================================
Backbone dong bang (DINOv2, ngay 1-13) dung o ~0.76 tren 58 ca gold - mot loi giai cong
khai dung DINOv2 + attention tung benh giong het cung ra 0.768. Cac doi diem cao deu train
CA backbone tren MRI, dung cua so 3 lat lien nhau (2.5D) o do phan giai cao hon, va gom
cua so bang attention rieng tung benh (MIL). File nay lam dung ba dieu do, tu viet, khong
dung checkpoint cua nguoi khac.

KIEN TRUC
=========
    moi ca: 3 mat phang x T cua so, moi cua so = 3 lat lien nhau lam 3 kenh anh
    -> CNN (timm, pretrained ImageNet) -> 1 vector moi cua so (+ nhung mat phang)
    -> attention RIENG tung benh tren moi cua so cua ca -> 12 logit
Train lay ngau nhien T_TRAIN cua so moi mat phang (rai deu theo chieu sau); cham dung
moi lat lam tam cua so.

GIAO THUC DANH GIA - chot TRUOC khi chay
========================================
* So epoch co dinh (EPOCHS), lich lr cosine. Moi model: trong so EMA o epoch cuoi.
  Khong chon epoch theo diem gold - day chinh la loi da sua o ngay 13.
* Hai model (MO_HINH), moi GPU mot model trong tien trinh rieng. CNN cuoi = trung binh
  THEO HANG cua cac model, trong so bang nhau.
* Diem gold cua moi epoch van duoc ghi lai, chi de ve duong cong cho minh bach.
* 5% ca nhan LLM giu lai lam val de theo doi (AUC so voi nhan LLM nhi phan hoa o 0.5).
* Chan ro ri: ca ngoai gold co bao cao trung tung ky tu voi mot ca gold bi bo khoi train.

RA (/kaggle/working/<backbone>_s<seed>/, moi model mot thu muc)
================================================================
    du_doan.npz   logit cua 58 ca gold va ca val o moi epoch (KHONG chua nhan)
    nhat_ky.json  cau hinh, diem tung epoch, thoi gian
    model_cuoi.pt trong so EMA cuoi (fp16)

CHAY THU O MAY: RSNA_INPUT, RSNA_OUT, RSNA_CACHE, RSNA_PRETRAINED=0, RSNA_EPOCHS=1,
RSNA_BS=2, RSNA_MAX_STUDIES=<n>, RSNA_MO_HINH=resnet10t:1,resnet10t:2, RSNA_THU_FORK=1, ... (RSNA_CACHE da co index.csv thi dung lai, khong doc DICOM)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Hoi so GPU qua NVML, KHONG khoi tao CUDA o tien trinh cha: cha se fork moi GPU mot tien
# trinh con train rieng, ma CUDA da khoi tao truoc fork thi tien trinh con khong dung duoc.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

E = os.environ.get
# ============================== CONFIG ==============================
# Moi GPU train MOT model trong tien trinh rieng; model CNN cuoi = trung binh theo hang cac model
MO_HINH = E("RSNA_MO_HINH", "efficientnet_b3:42,tf_efficientnetv2_b2:7")   # backbone:seed,...
PRETRAINED = E("RSNA_PRETRAINED", "1") == "1"
EPOCHS = int(E("RSNA_EPOCHS", "12"))
BS = int(E("RSNA_BS", "4"))               # so CA moi buoc (B3: 4 ca = 11 GB VRAM tren T4 16 GB)
ACCUM = int(E("RSNA_ACCUM", "2"))         # cong don gradient -> batch hieu dung BS * ACCUM = 8 ca
T_TRAIN = int(E("RSNA_T_TRAIN", "8"))     # cua so moi mat phang moi ca khi train
LR_ENC, LR_HEAD, WD = 2e-4, 1e-3, 1e-2
WARMUP = 0.5                              # epoch
EMA_DECAY = 0.999
VAL_FRAC = 0.05
SEED = 42
GIO_TOI_DA = float(E("RSNA_GIO", "10.5"))   # Kaggle cat o 12 gio
MAX_STUDIES = int(E("RSNA_MAX_STUDIES", "0"))
NHAN = E("RSNA_NHAN", "llm_labels_v4_blend.csv")   # file nhan mem; ngay 15: nhan tu gan (script 90)
THU_FORK = E("RSNA_THU_FORK", "0") == "1"   # chay thu o may 1 GPU: van tach 2 tien trinh
SIZE, DEPTH = 256, 24                     # cache: anh SIZE x SIZE, DEPTH lat moi mat phang
PLANES = ["Sagittal", "Coronal", "Axial"]
CACHE = Path(E("RSNA_CACHE", "/tmp/rsna_ca"))   # o dia cuc bo, KHONG phai /kaggle/working (se bi up len)
# ====================================================================
COMPETITION = "rsna-knee-abnormality-detection"
INPUT = Path(E("RSNA_INPUT", "/kaggle/input"))
OUT = Path(E("RSNA_OUT", "/kaggle/working"))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
L = len(LABELS)


def tim(ten: str, sau: int = 4) -> Path:
    for k in range(sau):
        for hit in sorted(INPUT.glob("/".join(["*"] * k + [ten]))):
            return hit
    raise SystemExit(f"Khong thay {ten} trong {INPUT} (sau {sau} cap) - kiem lai Input cua notebook")


def ram() -> str:
    """RAM cua ca container (cgroup) - lan chay dau bi Killed giua epoch 1 ma khong co dau vet."""
    for f in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        if os.path.exists(f):
            return f"RAM {int(open(f).read()) / 1e9:.1f}GB"
    return "RAM ?"


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC de THEO DOI trong luc train. Du doan co NaN/inf (tran so fp16 o model tho) thi
    tra nan va bao, KHONG nem loi: lan chay nhan tu gan dau tien chet o day sau 1 epoch."""
    from sklearn.metrics import roc_auc_score
    if not np.isfinite(p).all():
        print(f"  CANH BAO: {int((~np.isfinite(p)).sum())} du doan khong huu han - AUC = nan", flush=True)
        return float("nan")
    s = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])
         if len(np.unique(y[~np.isnan(y[:, j]), j])) == 2 and not np.isnan(y[:, j]).any()]
    return float(np.mean(s)) if s else float("nan")


# ---------------------------------------------------------------- tien xu ly
# Doc DICOM cua moi ca MOT lan, ghi cache gon ra o dia CUC BO de doc nhanh qua nhieu epoch.
# Lan dau tach phan nay ra notebook CPU rieng va gan output (13 GB, 4.407 file) lam input:
# Kaggle keo tung file ve khi doc (~2 MB/s, 10.8 s/buoc) va RAM phinh toi khi bi Killed o
# phut 55. Doc DICOM tu du lieu cuoc thi (mount san, nhanh) thi khong vuong.
#
#   moi ca -> <CACHE>/<StudyInstanceUID>.npz
#       x   uint8 (3, DEPTH, SIZE, SIZE)   3 mat phang Sagittal, Coronal, Axial
#       co  bool  (3,)                     mat phang nao doc duoc
#       n   int   (3,)                     so lat goc cua series da chon
#
# * Moi mat phang MOT series, uu tien Fluid_Sensitive=1 (quy tac "fluid" cua script 06):
#   cach cu lay series dung truoc trong CSV nen Sagittal chi 48.7% la xung nhay dich.
# * Chuan hoa percentile 1-99 tren CA series (khong theo tung lat), roi resize SIZE.
# * Lay mau lai DEPTH lat cach deu theo chieu sau, thu tu lat theo InstanceNumber.
def thu_tu_series(ts: pd.DataFrame) -> pd.DataFrame:
    """Sap series theo thu tu uu tien: Fluid_Sensitive=1 truoc, giu thu tu CSV khi hoa.
    Series dau tien cua moi (ca, mat phang) la series duoc chon; cac series sau la duong
    lui khi series dau doc hong."""
    if "Fluid_Sensitive" in ts.columns:
        ts = ts.sort_values("Fluid_Sensitive", ascending=False, kind="mergesort")
    return ts


def doc_series(d: Path) -> tuple[np.ndarray, int]:
    """(DEPTH, SIZE, SIZE) uint8 va so lat goc.

    Percentile tinh tren TOAN BO pixel cua series truoc khi resize: lam nguoc lai thi noi
    suy tao gia tri moi o bien va keo percentile lech. Lay pixel tung lat rieng nen series
    co lat khac kich thuoc (hiem, nhung co) van doc duoc.
    """
    import pydicom
    from PIL import Image

    lat = []
    for p in d.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(p))
            lat.append((int(getattr(ds, "InstanceNumber", 0) or 0), ds.pixel_array.astype(np.float32)))
        except Exception:
            continue
    if not lat:
        raise FileNotFoundError(f"khong doc duoc lat nao trong {d.name}")
    lat.sort(key=lambda t: t[0])
    tat_ca = np.concatenate([a.ravel() for _, a in lat])
    lo, hi = (np.float32(v) for v in np.percentile(tat_ca, (1, 99)))
    idx = np.round(np.linspace(0, len(lat) - 1, DEPTH)).astype(int)
    out = np.empty((DEPTH, SIZE, SIZE), dtype=np.uint8)
    cache: dict[int, np.ndarray] = {}
    for k, i in enumerate(idx):
        if i not in cache:
            a = np.clip((lat[i][1] - lo) / (hi - lo + np.float32(1e-8)), 0, 1)
            a = np.asarray(Image.fromarray(a, mode="F").resize((SIZE, SIZE), Image.BILINEAR))
            cache[i] = np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
        out[k] = cache[i]
    return out, len(lat)


def mot_ca(viec: tuple[str, dict[str, list[Path]], Path]) -> dict:
    uid, series, ra = viec
    x = np.zeros((len(PLANES), DEPTH, SIZE, SIZE), dtype=np.uint8)
    co = np.zeros(len(PLANES), dtype=bool)
    n = np.zeros(len(PLANES), dtype=np.int32)
    loi, lui = [], 0
    for j, pl in enumerate(PLANES):
        for k, d in enumerate(series.get(pl, [])):
            try:
                x[j], n[j] = doc_series(d)
                co[j] = True
                lui += k > 0
                break
            except Exception as exc:
                loi.append(f"{pl}#{k}: {type(exc).__name__}")
    np.savez_compressed(ra / f"{uid}.npz", x=x, co=co, n=n)
    return {"StudyInstanceUID": uid, **{f"co_{pl}": bool(co[j]) for j, pl in enumerate(PLANES)},
            **{f"n_{pl}": int(n[j]) for j, pl in enumerate(PLANES)}, "lui_series": lui,
            "loi": "; ".join(loi)}


def tao_cache(goc: Path, uids: list[str], ra: Path) -> pd.DataFrame:
    """Tien xu ly `uids` ra thu muc `ra`, tra ve bang chi muc (moi ca mot dong)."""
    from multiprocessing import Pool

    import shutil

    t0 = time.time()
    ra.mkdir(parents=True, exist_ok=True)
    print(f"o dia trong o {ra}: {shutil.disk_usage(ra).free / 1e9:.0f}GB (can ~13GB)", flush=True)
    ts = pd.read_csv(goc / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    uu_tien = thu_tu_series(ts[ts["Anatomical_Plane"].isin(PLANES) & ts["StudyInstanceUID"].isin(set(uids))])
    dau = uu_tien.groupby(["StudyInstanceUID", "Anatomical_Plane"]).head(1)
    print(f"tien xu ly {len(uids):,} ca, {os.cpu_count()} CPU -> {ra} | series nhay dich duoc chon: "
          f"{dau['Fluid_Sensitive'].mean():.1%}", flush=True)
    viec: dict[str, dict[str, list[Path]]] = {}
    for r in uu_tien.itertuples():
        viec.setdefault(r.StudyInstanceUID, {}).setdefault(r.Anatomical_Plane, []).append(
            goc / "train_series" / r.StudyInstanceUID / r.SeriesInstanceUID)
    dong = []
    with Pool(os.cpu_count()) as pool:
        for k, r in enumerate(pool.imap_unordered(mot_ca, [(u, viec.get(u, {}), ra) for u in uids], chunksize=4), 1):
            dong.append(r)
            if k % 500 == 0 or k == len(uids):
                tg = time.time() - t0
                print(f"  {k:,}/{len(uids):,} ca | {tg / 60:.1f} phut | con ~{tg / k * (len(uids) - k) / 60:.0f} phut"
                      f" | {ram()}", flush=True)
    idx = pd.DataFrame(dong)
    print(f"tien xu ly xong {(time.time() - t0) / 60:.1f} phut | thieu mat phang: "
          + ", ".join(f"{pl} {1 - idx[f'co_{pl}'].mean():.1%}" for pl in PLANES)
          + f" | ca co loi {(idx['loi'] != '').sum()}", flush=True)
    return idx


# ---------------------------------------------------------------- du lieu
class CaDataset(Dataset):
    def __init__(self, uids, y, thu_muc: Path, train: bool):
        self.uids, self.y, self.dir, self.train = list(uids), y.astype(np.float32), thu_muc, train

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        with np.load(self.dir / f"{self.uids[i]}.npz") as z:
            x, co = z["x"], z["co"]
        D = x.shape[1]
        if self.train:      # rai deu: chia chieu sau thanh T_TRAIN doan, moi doan lay 1 lat
            mep = np.linspace(0, D, T_TRAIN + 1).astype(int)
            tam = np.array([np.random.randint(a, max(a + 1, b)) for a, b in zip(mep[:-1], mep[1:])])
        else:
            tam = np.arange(D)
        cua_so = np.stack([np.clip(tam - 1, 0, D - 1), tam, np.clip(tam + 1, 0, D - 1)], 1)
        w = x[:, cua_so]                                  # (mat phang, T, 3, H, W) uint8
        return torch.from_numpy(w), torch.from_numpy(co), torch.from_numpy(self.y[i]), i


def tang_cuong(x: torch.Tensor) -> torch.Tensor:
    """Tren GPU, CUNG mot bien doi cho moi cua so cua mot (ca, mat phang): xoay +-10 do,
    co gian 0.9-1.1, dich +-6%, gamma/tuong phan/do sang. KHONG lat trai-phai: lat se doi
    ben trong/ngoai cua khop (Medial <-> Lateral)."""
    B, P, T, C, H, W = x.shape
    n = B * P
    dev = x.device
    goc = (torch.rand(n, device=dev) - 0.5) * (2 * math.pi * 10 / 180)
    tl = 1 / (0.9 + 0.2 * torch.rand(n, device=dev))
    cos, sin = torch.cos(goc) * tl, torch.sin(goc) * tl
    dich = (torch.rand(n, 2, device=dev) - 0.5) * 0.12
    theta = torch.stack([torch.stack([cos, -sin, dich[:, 0]], 1),
                         torch.stack([sin, cos, dich[:, 1]], 1)], 1)
    v = x.reshape(n, T * C, H, W)
    luoi = F.affine_grid(theta, v.shape, align_corners=False)
    v = F.grid_sample(v, luoi, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4)
    tuong_phan = 0.85 + 0.3 * torch.rand(n, 1, 1, 1, device=dev)
    sang = (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.1
    v = (v.clamp(0, 1) ** gamma) * tuong_phan + sang
    return v.clamp(0, 1).reshape(B, P, T, C, H, W)


# ---------------------------------------------------------------- model
class Mang25D(nn.Module):
    def __init__(self, backbone: str, pretrained: bool):
        super().__init__()
        import timm
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.enc.num_features
        self.mat_phang = nn.Parameter(torch.zeros(3, d))
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.2)
        # Attention RIENG tung benh, giong PerLabelAttnPool cua repo: moi benh mot bo trong
        # so tren cac cua so, va logit cua benh nao chi an theo vector gop cua chinh no.
        self.diem = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Linear(256, L))
        self.w = nn.Parameter(torch.empty(L, d))
        self.b = nn.Parameter(torch.zeros(L))
        nn.init.kaiming_uniform_(self.w, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, co: torch.Tensor) -> torch.Tensor:
        B, P, T, C, H, W = x.shape
        f = self.enc(x.reshape(B * P * T, C, H, W)).float().reshape(B, P, T, -1)
        f = self.norm(f + self.mat_phang[None, :, None])
        f = f.reshape(B, P * T, -1)
        mask = co[:, :, None].expand(B, P, T).reshape(B, P * T)
        mask = mask | ~mask.any(1, keepdim=True)   # ca khong co mat phang nao: tranh softmax(-inf) = NaN
        a = self.diem(f).masked_fill(~mask[..., None], float("-inf")).softmax(1)   # (B,K,L)
        self.chu_y = a.detach()                    # web demo doc lai: cua so nao cho benh nao
        g = self.drop(torch.einsum("bkl,bkd->bld", a, f))
        return (g * self.w).sum(-1) + self.b


def chuan_hoa(x_u8: torch.Tensor) -> torch.Tensor:
    return x_u8.float().div_(255)


def vao_mang(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - 0.45) / 0.225


@torch.no_grad()
def du_doan(model, loader, dev) -> np.ndarray:
    model.eval()
    out = np.zeros((len(loader.dataset), L), dtype=np.float32)
    for x, co, _, i in loader:
        x = vao_mang(chuan_hoa(x.to(dev, non_blocking=True)))
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev != "cpu"):
            out[i.numpy()] = model(x, co.to(dev)).float().cpu().numpy()
    return out


# ---------------------------------------------------------------- chay
def pss() -> str:
    """Bo nho THAT cua tien trinh (PSS). So cgroup o `ram()` tinh ca page cache cua 13GB cache
    vua ghi nen khong noi duoc tien trinh co ro ri hay khong."""
    try:
        return f"PSS {int(open('/proc/self/smaps_rollup').read().split('Pss:')[1].split()[0]) / 1e6:.2f}GB"
    except Exception:
        return "PSS ?"


def chuan_bi() -> dict:
    """Chia ca, tao cache cuc bo (neu chua co). Chay MOT lan truoc khi tach tien trinh train."""
    goc = tim("train_series.csv").parent
    llm = pd.read_csv(tim(NHAN), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    train = pd.read_csv(goc / "train.csv", dtype={"StudyInstanceUID": str})

    la_gold = train[LABELS].notna().any(axis=1)
    khoa = train["Report"].astype(str).str.strip().str.lower().map(lambda t: hashlib.md5(t.encode()).hexdigest())
    gold = train[la_gold]
    khac = train[~la_gold & ~khoa.isin(set(khoa[la_gold])) & train["StudyInstanceUID"].isin(llm.index)]
    n_ro_ri = int((~la_gold & khoa.isin(set(khoa[la_gold]))).sum())
    if MAX_STUDIES:
        khac = khac.iloc[:MAX_STUDIES]

    cache, xoa_cache_cuoi = CACHE, False
    if (cache / "index.csv").exists():          # chay thu o may: dung lai cache da tao
        idx = pd.read_csv(cache / "index.csv", dtype={"StudyInstanceUID": str}, keep_default_na=False)
    else:
        import shutil
        cache.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(cache.parent).free < 16e9 and OUT.exists():
            # /tmp hep: ghi vao /kaggle/working (20GB) va xoa truoc khi ket thuc, keo Kaggle up 13GB
            cache, xoa_cache_cuoi = OUT / "_cache_tam", True
        idx = tao_cache(goc, list(gold["StudyInstanceUID"]) + list(khac["StudyInstanceUID"]), cache)
        idx.to_csv(cache / "index.csv", index=False)
    co_anh = idx[[f"co_{pl}" for pl in PLANES]].astype(bool).any(axis=1)
    co_cache = set(idx.loc[co_anh, "StudyInstanceUID"])
    gold = gold[gold["StudyInstanceUID"].isin(co_cache)]
    khac = khac[khac["StudyInstanceUID"].isin(co_cache)]
    la_val = np.random.default_rng(SEED).random(len(khac)) < VAL_FRAC
    tr, va = khac[~la_val], khac[la_val]
    d = {"cache": cache, "xoa_cache_cuoi": xoa_cache_cuoi,
         "tr": tr["StudyInstanceUID"].tolist(), "y_tr": llm.loc[tr["StudyInstanceUID"], LABELS].to_numpy(float),
         "va": va["StudyInstanceUID"].tolist(),
         "y_va": (llm.loc[va["StudyInstanceUID"], LABELS].to_numpy(float) >= 0.5).astype(float),
         "gold": gold["StudyInstanceUID"].tolist(), "y_gold": gold[LABELS].to_numpy(float),
         "n_bo_trung_bao_cao": n_ro_ri, "n_bo_khong_anh": int((~co_anh).sum())}
    print(f"train {len(tr):,} | val nhan LLM {len(va)} | gold {len(gold)} | bo {n_ro_ri} ca trung bao cao gold"
          f" | bo {d['n_bo_khong_anh']} ca khong doc duoc mat phang nao", flush=True)
    return d


def train_mot(ten: str, backbone: str, seed: int, gpu: int | None, d: dict, han: float) -> None:
    """Train MOT model tren MOT GPU, ghi ket qua vao OUT/<ten>/. `han`: moc time.time() phai xong."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if gpu is None:
        dev = "cpu"
    else:
        torch.cuda.set_device(gpu)
        dev = f"cuda:{gpu}"
    la_gpu = dev != "cpu"
    ra = OUT / ten
    ra.mkdir(parents=True, exist_ok=True)

    def p(msg: str) -> None:
        print(f"[{ten}] {msg}", flush=True)

    p(f"thiet bi {dev} | backbone {backbone} | seed {seed} | pretrained {PRETRAINED}")
    nw = max(1, min(8, os.cpu_count() or 2) // max(1, torch.cuda.device_count()))   # chia CPU cho cac tien trinh
    lt = DataLoader(CaDataset(d["tr"], d["y_tr"], d["cache"], True), batch_size=BS, shuffle=True,
                    num_workers=nw, pin_memory=la_gpu, drop_last=True,
                    worker_init_fn=lambda k: np.random.seed(seed * 1000 + k + int(time.time()) % 1000))
    lg = DataLoader(CaDataset(d["gold"], d["y_gold"], d["cache"], False), batch_size=2, num_workers=nw)
    lv = DataLoader(CaDataset(d["va"], d["y_va"], d["cache"], False), batch_size=2, num_workers=nw)

    from timm.utils import ModelEmaV2
    for lan in range(5):     # tai trong so tu Hugging Face: thu lai khi mang chap chon
        try:
            net = Mang25D(backbone, PRETRAINED).to(dev)
            break
        except Exception as exc:
            if lan == 4:
                raise
            p(f"tai backbone loi ({type(exc).__name__}), thu lai sau {15 * (lan + 1)}s")
            time.sleep(15 * (lan + 1))
    ema = ModelEmaV2(net, decay=EMA_DECAY)

    # Do toc do tren 10 batch that (forward + backward, KHONG buoc optimizer) roi chot so
    # epoch vua ngan sach thoi gian. Chon theo THOI GIAN, khong nhin diem gold - va chot
    # truoc khi train, nen lich lr cosine van dung tron ven.
    net.train()
    t_do, n_do, cho_do = None, 0, 0.0
    tc = time.perf_counter()
    for k, (x, co, y, _) in enumerate(lt):
        if k == 2:
            torch.cuda.synchronize() if la_gpu else None
            t_do, cho_do = time.perf_counter(), 0.0
        elif k > 2:
            cho_do += time.perf_counter() - tc
        x = vao_mang(tang_cuong(chuan_hoa(x.to(dev, non_blocking=True))))
        with torch.autocast("cuda", dtype=torch.float16, enabled=la_gpu):
            loss = F.binary_cross_entropy_with_logits(net(x, co.to(dev)).float(), y.to(dev))
        loss.backward()
        net.zero_grad(set_to_none=True)
        if k >= 2:
            n_do += 1
        if k == 11:
            break
        if la_gpu:
            torch.cuda.synchronize()
        tc = time.perf_counter()
    giay_buoc = (time.perf_counter() - t_do) / max(1, n_do) if t_do else 0.0
    phut_epoch = giay_buoc * len(lt) / 60 * 1.15 + 1.5       # +15% cho cap nhat, +1.5 phut cham
    con_phut = (han - time.time()) / 60
    epochs = max(1, min(EPOCHS, int(con_phut // max(phut_epoch, 1e-6))))
    p(f"do toc do: {giay_buoc:.2f}s/buoc (cho du lieu {cho_do / max(1, n_do):.2f}s) -> ~{phut_epoch:.0f} phut/epoch; "
      f"ngan sach con {con_phut:.0f} phut -> chay {epochs}/{EPOCHS} epoch")

    nhom = [{"params": net.enc.parameters(), "lr": LR_ENC},
            {"params": [q for n, q in net.named_parameters() if not n.startswith("enc.")], "lr": LR_HEAD}]
    opt = torch.optim.AdamW(nhom, weight_decay=WD)
    moi_epoch = len(lt) // ACCUM                              # so buoc optimizer moi epoch
    tong = epochs * moi_epoch
    khoi = max(1, int(WARMUP * moi_epoch))
    lich = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / khoi if s < khoi else 0.5 * (1 + math.cos(math.pi * (s - khoi) / max(1, tong - khoi))))
    scaler = torch.amp.GradScaler("cuda", enabled=la_gpu)

    cau_hinh = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, bool))}
    nk = {"ten": ten, "cau_hinh": {**cau_hinh, "BACKBONE": backbone, "SEED_MO_HINH": seed, "EPOCHS_CHAY": epochs},
          "n_train": len(d["tr"]), "n_val": len(d["va"]), "n_gold": len(d["gold"]),
          "n_bo_trung_bao_cao": d["n_bo_trung_bao_cao"], "n_bo_khong_anh": d["n_bo_khong_anh"], "epoch": []}
    P_gold_ema, P_gold_tho, P_val = [], [], []
    for ep in range(1, epochs + 1):
        net.train()
        te, tong_loss, cho = time.time(), 0.0, 0.0
        opt.zero_grad(set_to_none=True)
        tc = time.perf_counter()
        for k, (x, co, y, _) in enumerate(lt, 1):
            cho += time.perf_counter() - tc                  # thoi gian dung cho DataLoader
            x = vao_mang(tang_cuong(chuan_hoa(x.to(dev, non_blocking=True))))
            with torch.autocast("cuda", dtype=torch.float16, enabled=la_gpu):
                loss = F.binary_cross_entropy_with_logits(net(x, co.to(dev)).float(), y.to(dev))
            scaler.scale(loss / ACCUM).backward()
            if k % ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                lich.step()
                ema.update(net)
            tong_loss += loss.item()
            if k % 100 == 0 or k == len(lt):
                tg = time.time() - te
                p(f"epoch {ep} buoc {k}/{len(lt)} | loss {tong_loss / k:.4f} | {tg / k:.2f}s/buoc "
                  f"(cho du lieu {cho / k:.2f}s) | con epoch ~{tg / k * (len(lt) - k) / 60:.0f} phut | {pss()} | {ram()}")
            if time.time() > han:
                break
            tc = time.perf_counter()

        pg_ema, pg_tho = du_doan(ema.module, lg, dev), du_doan(net, lg, dev)
        pv = du_doan(ema.module, lv, dev)
        P_gold_ema.append(pg_ema), P_gold_tho.append(pg_tho), P_val.append(pv)
        dong = {"epoch": ep, "loss": round(tong_loss / max(1, k), 5),
                "gold_ema": macro_auc(d["y_gold"], pg_ema), "gold_tho": macro_auc(d["y_gold"], pg_tho),
                "val_llm_ema": macro_auc(d["y_va"], pv), "phut_epoch": round((time.time() - te) / 60, 1),
                "gio_da_chay": round((time.time() - t0) / 3600, 2)}
        nk["epoch"].append(dong)
        p(f"== epoch {ep}: gold EMA {dong['gold_ema']:.4f} | gold tho {dong['gold_tho']:.4f} | "
          f"val LLM {dong['val_llm_ema']:.4f} | {dong['phut_epoch']} phut")
        # Ghi sau MOI epoch: lo het gio giua chung van con ket qua da co
        np.savez_compressed(ra / "du_doan.npz", gold_uid=np.array(d["gold"], dtype=str),
                            gold_ema=np.stack(P_gold_ema), gold_tho=np.stack(P_gold_tho),
                            val_uid=np.array(d["va"], dtype=str), val_ema=np.stack(P_val))
        torch.save({k: v.half() for k, v in ema.module.state_dict().items()}, ra / "model_cuoi.pt")
        (ra / "nhat_ky.json").write_text(json.dumps(nk, indent=2, ensure_ascii=False))
        if time.time() > han:
            p("het ngan sach thoi gian - dung som")
            nk["dung_som"] = True
            break

    nk["gio"] = round((time.time() - t0) / 3600, 2)
    (ra / "nhat_ky.json").write_text(json.dumps(nk, indent=2, ensure_ascii=False))
    p(f"XONG. Model cuoi (EMA, epoch {nk['epoch'][-1]['epoch']}): gold {nk['epoch'][-1]['gold_ema']:.4f}")


def main() -> None:
    t0 = time.time()
    han = t0 + GIO_TOI_DA * 3600
    n_gpu = torch.cuda.device_count()          # NVML, CHUA khoi tao CUDA (xem dau file) - con fork duoc
    mo_hinh = [(f"{b}_s{s}", b, int(s)) for b, s in (m.split(":") for m in MO_HINH.split(","))]
    print(f"{n_gpu} GPU | model: {', '.join(t for t, _, _ in mo_hinh)}", flush=True)
    d = chuan_bi()

    if n_gpu >= 2 and len(mo_hinh) >= 2 or THU_FORK:
        # Moi GPU mot model, moi model mot tien trinh. KHONG dung nn.DataParallel: tren 2 T4 cua
        # Kaggle no cham 20 lan (10.7 s/buoc so voi 0.47 s/buoc tren 1 GPU, do o lan chan doan).
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        con = [ctx.Process(target=train_mot, args=(*mo_hinh[j], j % max(1, n_gpu), d, han), name=mo_hinh[j][0])
               for j in range(1, min(len(mo_hinh), max(n_gpu, 2)))]
        for c in con:
            c.start()
        train_mot(*mo_hinh[0], 0, d, han)
        for c in con:
            c.join()
            print(f"tien trinh {c.name} ket thuc, ma thoat {c.exitcode}", flush=True)
        if len(mo_hinh) > max(n_gpu, 2):
            print(f"BO QUA {len(mo_hinh) - n_gpu} model vi khong du GPU", flush=True)
    else:
        for ten, b, s in mo_hinh:
            train_mot(ten, b, s, 0 if n_gpu else None, d, han)

    if d["xoa_cache_cuoi"]:
        import shutil
        shutil.rmtree(d["cache"], ignore_errors=True)
    print(f"XONG TAT CA sau {(time.time() - t0) / 3600:.2f} gio", flush=True)


if __name__ == "__main__":
    main()
