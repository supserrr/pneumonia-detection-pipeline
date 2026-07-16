"""
FastAPI service for the Pneumonia chest X-ray classification pipeline.

Endpoints
---------
GET  /                -> web UI (uptime, visualizations, predict, upload, retrain)
GET  /status          -> model up-time, version info, current metrics
GET  /visualizations  -> dataset statistics for the UI charts
POST /predict         -> classify ONE uploaded chest X-ray image
POST /upload          -> bulk-upload labelled images for retraining
POST /retrain         -> trigger for retraining (background task)
GET  /retrain/status  -> progress/result of the last retraining run
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import CLASS_NAMES, UPLOAD_DIR, dataset_statistics  # noqa: E402
from src.model import METADATA_PATH  # noqa: E402
from src import prediction  # noqa: E402

app = FastAPI(title="Pneumonia Chest X-ray Classifier API", version="1.0.0")

START_TIME = time.time()
_retrain_state = {"status": "idle", "detail": None, "started_at": None, "finished_at": None}
_stats_cache: dict | None = None


@app.on_event("startup")
def _warm_up() -> None:
    """Load the model at startup so the first request is fast."""
    try:
        prediction.get_model()
    except Exception as exc:  # model file missing on a fresh clone
        print(f"WARNING: model not loaded at startup: {exc}")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text()


# --------------------------------------------------------------------------- #
# Status / model up-time
# --------------------------------------------------------------------------- #
@app.get("/status")
def status() -> dict:
    uptime = time.time() - START_TIME
    meta = {}
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
    return {
        "status": "up",
        "uptime_seconds": round(uptime, 1),
        "uptime_human": _human_time(uptime),
        "model_saved_at": meta.get("saved_at"),
        "model_note": meta.get("note"),
        "model_metrics": meta.get("metrics", {}),
        "retraining": _retrain_state["status"],
    }


def _human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return f"{d}d {h}h {m}m {s}s"


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
@app.get("/visualizations")
def visualizations() -> dict:
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = dataset_statistics()
    return _stats_cache


# --------------------------------------------------------------------------- #
# Prediction (single data point)
# --------------------------------------------------------------------------- #
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file (PNG/JPEG).")
    data = await file.read()
    try:
        result = prediction.predict_image(data)
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc
    result["filename"] = file.filename
    return result


# --------------------------------------------------------------------------- #
# Bulk upload of new training data
# --------------------------------------------------------------------------- #
@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    label: str = Form(...),
) -> dict:
    label = label.upper().strip()
    if label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    dest = UPLOAD_DIR / label
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        data = await f.read()
        name = f"upload_{int(time.time()*1000)}_{saved:04d}.png"
        try:
            from PIL import Image
            import io

            Image.open(io.BytesIO(data)).convert("L").save(dest / name)
            saved += 1
        except Exception:
            continue  # skip non-image files silently
    total = sum(1 for _ in UPLOAD_DIR.rglob("*.png"))
    return {"saved": saved, "label": label, "total_pending_upload_images": total}


# --------------------------------------------------------------------------- #
# Retraining trigger
# --------------------------------------------------------------------------- #
@app.post("/retrain")
def retrain(epochs: int = 3) -> dict:
    if _retrain_state["status"] == "running":
        raise HTTPException(409, "A retraining job is already running.")
    thread = threading.Thread(target=_run_retraining, args=(epochs,), daemon=True)
    thread.start()
    return {"message": "Retraining started (uses saved model as pre-trained base).", "epochs": epochs}


def _run_retraining(epochs: int) -> None:
    from src.model import retrain_model
    from src.preprocessing import TEST_DIR, load_dataset

    _retrain_state.update(status="running", detail=None, started_at=time.strftime("%H:%M:%S"))
    try:
        X_test, y_test = load_dataset(TEST_DIR)
        result = retrain_model(epochs=epochs, X_test=X_test, y_test=y_test)
        prediction.reload_model()  # serve the new model immediately
        # archive consumed uploads so they are not retrained on twice
        if UPLOAD_DIR.exists() and result["n_new_uploaded"] > 0:
            archive = UPLOAD_DIR.parent / "uploads_archive"
            archive.mkdir(exist_ok=True)
            shutil.copytree(UPLOAD_DIR, archive / time.strftime("%Y%m%d_%H%M%S"))
            shutil.rmtree(UPLOAD_DIR)
        _retrain_state.update(status="done", detail=result, finished_at=time.strftime("%H:%M:%S"))
    except Exception as exc:
        _retrain_state.update(status="failed", detail=str(exc), finished_at=time.strftime("%H:%M:%S"))


@app.get("/retrain/status")
def retrain_status() -> JSONResponse:
    return JSONResponse(_retrain_state)
