# vis10n server: serves the UI and exposes the API the windows talk to
# Run with: python main.py
import asyncio
import re
import threading
from dataclasses import fields
from pathlib import Path

import torch as t
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predict import load_model, predict
from train_model import SimpleMLPTrainingArgs, train

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

app = FastAPI()

# Most recent fully trained model, kept in memory until saved
latest_model = None


# Routes
# UI
@app.get("/")
def index():
    return FileResponse(ROOT / "window.html")


# API
@app.get("/api/ping")
def ping():
    return {"ok": True}


def build_args(raw: dict) -> SimpleMLPTrainingArgs:
    """Build training args from the page's values, keeping only known fields cast to their types."""
    arg_fields = {f.name: f.type for f in fields(SimpleMLPTrainingArgs)}
    return SimpleMLPTrainingArgs(**{
        name: arg_fields[name](value) for name, value in raw.items() if name in arg_fields
    })


@app.websocket("/ws/train")
async def train_socket(ws: WebSocket):
    """
    Page sends {"type": "start", "args": {...}}, then optionally {"type": "cancel"}.
    Server streams train()'s batch/epoch updates, then {"type": "done", "cancelled": bool}
    or {"type": "error", "message": str}.
    """
    await ws.accept()

    loop = asyncio.get_running_loop()
    updates = asyncio.Queue()
    cancel = threading.Event()

    # train() runs in its own thread; this hands its updates back to the event loop
    def send(update: dict):
        loop.call_soon_threadsafe(updates.put_nowait, update)

    def run(args: SimpleMLPTrainingArgs):
        global latest_model
        try:
            _, _, model = train(args, on_update=send, cancel=cancel)
            if not cancel.is_set():
                latest_model = model
            send({"type": "done", "cancelled": cancel.is_set()})
        except Exception as e:
            send({"type": "error", "message": str(e)})

    # Listen for cancel while updates are being sent
    async def listen():
        try:
            while True:
                message = await ws.receive_json()
                if message.get("type") == "cancel":
                    cancel.set()
        except (WebSocketDisconnect, RuntimeError):
            cancel.set()

    try:
        start = await ws.receive_json()
        args = build_args(start.get("args", {}))
    except WebSocketDisconnect:
        return
    except (TypeError, ValueError) as e:
        await ws.send_json({"type": "error", "message": f"bad args: {e}"})
        await ws.close()
        return

    threading.Thread(target=run, args=(args,), daemon=True).start()
    listener = asyncio.create_task(listen())

    try:
        while True:
            update = await updates.get()
            await ws.send_json(update)
            if update["type"] in ("done", "error"):
                break
        await ws.close()
    except (WebSocketDisconnect, RuntimeError):
        # Page went away mid-training: stop train() on its next batch
        cancel.set()
    finally:
        listener.cancel()


@app.get("/api/models")
def list_models():
    return {"models": sorted(p.stem for p in MODELS_DIR.glob("*.pt"))}


class PredictRequest(BaseModel):
    model: str
    pixels: list[list[int]]


# Last loaded model, reused until a different model is picked or its file changes
loaded = {"path": None, "mtime": None, "model": None}


@app.post("/api/predict")
def predict_digit(req: PredictRequest):
    path = MODELS_DIR / f"{req.model}.pt"
    if not re.fullmatch(r"[\w\- ]+", req.model) or not path.exists():
        raise HTTPException(404, "model not found")

    if len(req.pixels) != 28 or any(len(row) != 28 for row in req.pixels):
        raise HTTPException(400, "pixels must be 28x28")

    mtime = path.stat().st_mtime
    if loaded["path"] != path or loaded["mtime"] != mtime:
        loaded.update(path=path, mtime=mtime, model=load_model(path))

    digit, probabilities = predict(loaded["model"], req.pixels)
    return {"digit": digit, "probabilities": probabilities}


class SaveRequest(BaseModel):
    name: str


@app.post("/api/save")
def save_model(req: SaveRequest):
    if latest_model is None:
        raise HTTPException(400, "no trained model")

    name = req.name.strip()
    if not re.fullmatch(r"[\w\- ]+", name):
        raise HTTPException(400, "invalid name")

    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.pt"
    t.save(latest_model.state_dict(), path)
    return {"ok": True, "path": str(path.relative_to(ROOT))}


if __name__ == "__main__":
    # Reload needs the app as an import string rather than the object
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
