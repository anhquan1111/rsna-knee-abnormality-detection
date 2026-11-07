"""Web demo: chon ca mau hoac tai ca DICOM len, model CNN 2.5D tra 12 xac suat + ban do chu y.

    uv sync --extra demo
    uv run python scripts/95_web_demo.py          # mo http://127.0.0.1:8790

Chi nghe 127.0.0.1 (luat may: web khong ra mang ngoai). Anh tai len nam trong thu muc tam,
xu ly xong la xoa. Can:
  * data/web_demo/model/<backbone>_s<seed>/model_cuoi.pt   trong so EMA tu notebook 81
  * data/web_demo/ca_mau/ca_mau_<k>.zip + danh_sach.txt    ca mau (tuy chon)
Giao dien: web_demo/index.html (Gemini Flash soan nhap, Claude soat va sua).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

from rsna_knee import demo  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONG = 8790
GIOI_HAN_MB = 400

app = FastAPI(title="RSNA Knee demo", docs_url=None, redoc_url=None)
may = demo.MayDoan()
MAU = {m["id"]: m for m in demo.ca_mau()}


@app.get("/")
def trang_chu():
    return FileResponse(REPO / "web_demo" / "index.html")


@app.get("/api/ca-mau")
def danh_sach_mau():
    return [{"id": m["id"], "ten": m["ten"], "n_benh": m["n_benh"], "mo_ta": m["mo_ta"]} for m in MAU.values()]


@app.post("/api/du-doan")
async def du_doan(request: Request, mau: str | None = None):
    if mau:
        if mau not in MAU:
            return JSONResponse({"loi": f"Không có ca mẫu {mau}"}, status_code=400)
        m = MAU[mau]
        tep, ten, nhan = [(m["zip"].name, m["zip"].read_bytes())], m["ten"], m["nhan_bac_si"]
    else:
        form = await request.form(max_files=5000)
        tep = [(f.filename or "tep", await f.read()) for f in form.getlist("tep") if hasattr(f, "read")]
        if not tep:
            return JSONResponse({"loi": "Chưa có file nào được tải lên."}, status_code=400)
        if sum(len(b) for _, b in tep) > GIOI_HAN_MB * 2**20:
            return JSONResponse({"loi": f"Quá {GIOI_HAN_MB} MB."}, status_code=400)
        ten, nhan = (tep[0][0] if len(tep) == 1 else f"{len(tep)} file DICOM"), None

    series, tam, canh_bao = demo.tach_series_tai_len(tep)
    try:
        x, co, tt = may.doc_ca(series)
        if not co.any():
            return JSONResponse({"loi": "Không đọc được mặt phẳng nào từ các file này. Cần DICOM MRI "
                                        "có ImageOrientationPatient."}, status_code=400)
        kq = may.ket_qua(x, co, tt, nhan)
    finally:
        demo.don(tam)
    return {"ten": ten, "nguon": "mau" if mau else "tai_len", "canh_bao": canh_bao, **kq}


if __name__ == "__main__":
    print(f"Mo http://127.0.0.1:{CONG}  ({len(MAU)} ca mau, model: {', '.join(n for n, _ in may.models)})")
    uvicorn.run(app, host="127.0.0.1", port=CONG, log_level="warning")
