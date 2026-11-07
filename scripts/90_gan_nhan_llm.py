"""NGAY 15 - Tu gan nhan 12 benh cho moi bao cao bang LLM (Gemini Flash qua `agy`).

    python scripts/90_gan_nhan_llm.py --thu       # 1 lo 25 bao cao, xem dinh dang
    python scripts/90_gan_nhan_llm.py             # ca 4.407 bao cao, chay tiep duoc
    python scripts/90_gan_nhan_llm.py --gop       # gop cac lo thanh mot file CSV

VI SAO TU LAM
=============
Nhan LLM dung tu ngay 10 la bo cong khai (`stevenleehans/rsna-knee-llm-report-labels`),
khong ghi LLM nao, prompt nao. File nay lam lai tu dau, MINH BACH: prompt nam ngay duoi day,
model ghi trong file ket qua, va ca 58 ca gold cung duoc gan nhan de do (script 91).

QUY TAC CHOT TRUOC (khong sua sau khi xem diem gold)
====================================================
* Prompt viet MOT lan, theo dinh nghia y khoa cua tung benh - khong chinh theo diem gold.
* LLM chi thay van ban bao cao, kem so thu tu trong lo (r0..r24). Khong gui UID, khong
  gui nhan bac si.
* Gold va ngoai gold tron chung trong cung cac lo, cung mot prompt.
* Diem ra la so nguyen 0-100 = xac suat bao cao KET LUAN co benh -> chia 100 lam nhan mem.

Moi lo ghi ngay ra data/nhan_llm_cua_minh/lo/<so>.json, nen chay lai chi lam lo con thieu.
Can `agy` (Antigravity CLI) da dang nhap; chay o thu muc da tin cua agy (AGY_THU_MUC).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna_knee.config import LABELS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RA = REPO / "data" / "nhan_llm_cua_minh"
LO = RA / "lo"
MODEL = "gemini-3.8-flash-medium"
CO_LO = 25
SONG_SONG = 4
AGY_THU_MUC = Path(os.environ.get("AGY_THU_MUC", Path.home() / ".cache" / "xuong-phim" / "agy_trong"))

PROMPT = """You are a musculoskeletal radiologist. Below are {n} knee MRI reports, each tagged r0..r{m}.
Reports may be in English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch or French.

For EVERY report, score 12 findings with an integer 0-100 = probability that the report
CONCLUDES the finding is present in this knee. Definitions:
  ACL               anterior cruciate ligament tear (partial or complete), or failed/torn ACL graft.
                    Intact native ligament or intact graft = low. Mucoid degeneration without tear ~30.
  MCL               medial collateral ligament injury of any grade (sprain, partial or complete tear).
  Medial Meniscus   tear of the medial meniscus (any type, incl. complex, radial, root, bucket-handle).
                    Intrasubstance degenerative signal WITHOUT a tear ~25. Post-meniscectomy alone ~40.
  Lateral Meniscus  same, for the lateral meniscus.
  Medial OA         osteoarthritis of the medial femorotibial compartment (cartilage loss,
                    osteophytes, subchondral changes). Mild ~65, moderate/severe ~95.
  Lateral OA        same, lateral femorotibial compartment.
  PF OA             same, patellofemoral compartment (incl. chondromalacia patellae with cartilage loss).
  Effusion          joint effusion beyond physiological. Small ~70, moderate/large ~95,
                    "minimal/physiological fluid" ~20.
  Synovitis         synovitis, synovial thickening or hypertrophy (incl. Hoffa fat pad synovitis).
  Baker's           popliteal (Baker's) cyst of any size.
  Contusion         bone contusion / bone bruise / traumatic bone marrow edema.
  Fracture          acute or subacute fracture of any bone in the knee (incl. avulsion,
                    osteochondral, impaction). Old healed fracture ~30.
Scoring rules:
  * explicitly present ~95; explicitly absent or normal ~3; not mentioned at all ~8;
  * hedged ("possible", "cannot exclude", "suspicious for") ~50; "likely/probable" ~75.
  * judge only from the text; do not guess from typical co-occurrence.

Return ONLY one JSON object, no prose, no markdown fence:
{{"r0": [ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture], "r1": [...], ...}}
with exactly {n} keys and exactly 12 integers per key, in that order.

REPORTS
{bao_cao}
"""


def lam_lo(k: int, bao_cao: list[str]) -> dict:
    p = PROMPT.format(n=len(bao_cao), m=len(bao_cao) - 1,
                      bao_cao="\n\n".join(f"### r{i}\n{t.strip()}" for i, t in enumerate(bao_cao)))
    loi = ""
    for lan in range(3):
        try:
            r = subprocess.run(["agy", "-p", p, "--output-format", "json", "--model", MODEL],
                               cwd=AGY_THU_MUC, capture_output=True, text=True, timeout=900)
            tra = json.loads(r.stdout.strip().splitlines()[-1])
            txt = re.sub(r"^```(?:json)?|```$", "", tra["response"].strip(), flags=re.M).strip()
            d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            diem = [d[f"r{i}"] for i in range(len(bao_cao))]
            if all(len(v) == len(LABELS) and all(0 <= int(x) <= 100 for x in v) for v in diem):
                return {"lo": k, "diem": [[int(x) for x in v] for v in diem], "model": MODEL,
                        "lan_thu": lan + 1, "token": tra.get("usage", {})}
            loi = "sai so cot"
        except Exception as exc:          # het han muc, JSON hong, thieu khoa ...
            loi = f"{type(exc).__name__}: {str(exc)[:200]}"
        time.sleep(20 * (lan + 1))
    raise RuntimeError(f"lo {k}: {loi}")


def main() -> None:
    train = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str})
    uid, bao_cao = train["StudyInstanceUID"].tolist(), train["Report"].astype(str).tolist()
    lo = [list(range(i, min(i + CO_LO, len(uid)))) for i in range(0, len(uid), CO_LO)]
    LO.mkdir(parents=True, exist_ok=True)

    if "--gop" in sys.argv:
        gop(uid, lo)
        return
    viec = [k for k in range(len(lo)) if not (LO / f"{k:03d}.json").exists()]
    if "--thu" in sys.argv:
        viec = viec[:1]
    print(f"{len(lo)} lo x {CO_LO} bao cao | con {len(viec)} lo | model {MODEL}", flush=True)
    t0, xong, hong = time.time(), 0, []
    with ThreadPoolExecutor(SONG_SONG) as ex:
        tuong_lai = {ex.submit(lam_lo, k, [bao_cao[i] for i in lo[k]]): k for k in viec}
        for f in as_completed(tuong_lai):
            k = tuong_lai[f]
            try:
                (LO / f"{k:03d}.json").write_text(json.dumps(f.result()))
                xong += 1
            except Exception as exc:
                hong.append(k)
                print(f"  HONG {exc}", flush=True)
            tg = time.time() - t0
            print(f"  {xong + len(hong)}/{len(viec)} lo | {tg / 60:.0f} phut | con ~"
                  f"{tg / max(1, xong + len(hong)) * (len(viec) - xong - len(hong)) / 60:.0f} phut", flush=True)
    print(f"XONG {xong} lo, hong {len(hong)}: {hong}" if hong else f"XONG {xong} lo", flush=True)
    if not hong and "--thu" not in sys.argv:
        gop(uid, lo)


def gop(uid: list[str], lo: list[list[int]]) -> None:
    thieu = [k for k in range(len(lo)) if not (LO / f"{k:03d}.json").exists()]
    if thieu:
        raise SystemExit(f"Con thieu {len(thieu)} lo: {thieu[:10]}... - chay lai khong co --gop")
    y = np.full((len(uid), len(LABELS)), np.nan)
    model = set()
    for k, idx in enumerate(lo):
        d = json.loads((LO / f"{k:03d}.json").read_text())
        y[idx] = np.array(d["diem"], dtype=float) / 100
        model.add(d["model"])
    out = pd.DataFrame(y, columns=list(LABELS))
    out.insert(0, "StudyInstanceUID", uid)
    out.to_csv(RA / "llm_labels_cua_minh.csv", index=False)
    (RA / "nguon_goc.json").write_text(json.dumps(
        {"model": sorted(model), "co_lo": CO_LO, "n_bao_cao": len(uid),
         "prompt": PROMPT, "script": "scripts/90_gan_nhan_llm.py"}, indent=2, ensure_ascii=False))
    print(f"Da ghi {RA / 'llm_labels_cua_minh.csv'} ({len(uid)} ca)")


if __name__ == "__main__":
    main()
