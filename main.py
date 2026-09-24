# vis10n server: serves the UI and exposes the API the windows talk to
# Run with: python main.py
import asyncio
import importlib
import json
import re
import subprocess
import threading
import time
import traceback
import urllib.request
import webbrowser
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import torch as t
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dataloaders import cifar, imagenet
from models import get_model
from models import list_models as list_architectures
from optimizers import list_optimizers
from predict import load_model, predict, predict_cifar, predict_imagenet, predict_photo
from device import device
from training.train_mnist_mlp import SimpleMLPTrainingArgs, train

ROOT = Path(__file__).parent
WEIGHTS_DIR = ROOT / "weights"

app = FastAPI()

# Most recent fully trained run (model + what produced it), kept in memory until saved
latest_run = None


# Routes
# UI
@app.get("/")
def index():
    return FileResponse(ROOT / "window.html")


# Images and other files the UI loads
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


# API
@app.get("/api/ping")
def ping():
    return {"ok": True}


@app.get("/api/architectures")
def architectures():
    """Architecture files in models/, each with its module docstring as a description."""
    result = []
    for name in list_architectures():
        try:
            description = (importlib.import_module(f"models.{name}").__doc__ or "").strip()
        except Exception as e:
            traceback.print_exc()
            description = f"error: {e}"
        result.append({"name": name, "description": description})
    return {"architectures": result}


@app.get("/api/optimizers")
def optimizers_available():
    """Optimizer classes the training code can be pointed at."""
    return {"optimizers": list_optimizers()}


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

    def run(args: SimpleMLPTrainingArgs, model_name: str):
        global latest_run
        try:
            started = time.perf_counter()
            loss, accuracy, model = train(args, get_model(model_name), on_update=send, cancel=cancel)
            if not cancel.is_set():
                latest_run = {
                    "model": model,
                    "model_name": model_name,
                    "args": args,
                    "loss": loss,
                    "accuracy": accuracy,
                    "train_seconds": time.perf_counter() - started,
                    "finished": datetime.now(),
                }
            send({"type": "done", "cancelled": cancel.is_set()})
        except Exception as e:
            traceback.print_exc()   # full details in the server terminal; the page only gets the message
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
        model_name = start.get("model")
        if model_name not in list_architectures():
            raise ValueError(f"unknown architecture: {model_name}")
    except WebSocketDisconnect:
        return
    except (TypeError, ValueError) as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return

    threading.Thread(target=run, args=(args, model_name), daemon=True).start()
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
    """Every saved model with its info file (None for models saved before info files existed)."""
    models = []
    for path in sorted(WEIGHTS_DIR.glob("*.pt")):
        info_path = path.with_suffix(".json")
        try:
            info = json.loads(info_path.read_text()) if info_path.exists() else None
        except (OSError, json.JSONDecodeError):
            info = None
        models.append({"name": path.stem, "info": info})
    return {"models": models}


TEST_SETS = {"CIFAR-10": cifar, "ImageNet": imagenet}


@app.get("/api/sample")
def sample_image(dataset: str = "CIFAR-10", index: int | None = None):
    """A test-set image to feed a model: random unless an index is given."""
    if dataset not in TEST_SETS:
        raise HTTPException(400, f"no test set for {dataset}")
    try:
        return TEST_SETS[dataset].sample(index)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except IndexError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/models/{name}")
def delete_model(name: str):
    """Deletes a saved model's weights and info file."""
    path = WEIGHTS_DIR / f"{name}.pt"
    if not re.fullmatch(r"[\w\- ]+", name) or not path.exists():
        raise HTTPException(404, "model not found")

    # Forget it if it's the model currently loaded for predictions
    if loaded["path"] == path:
        loaded.update(path=None, mtime=None, model=None)

    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


class PredictRequest(BaseModel):
    model: str
    dataset: str = "MNIST"
    pixels: list[list[int]] | None = None   # MNIST: the drawing
    index: int | None = None                # CIFAR-10: which test image


# Last loaded model, reused until a different model is picked or its file changes
loaded = {"path": None, "mtime": None, "model": None}


@app.post("/api/predict")
def predict_image(req: PredictRequest):
    """MNIST takes a 28x28 drawing; CIFAR-10 takes the index of a test-set image."""
    path = WEIGHTS_DIR / f"{req.model}.pt"
    info_path = path.with_suffix(".json")
    if not re.fullmatch(r"[\w\- ]+", req.model) or not path.exists():
        raise HTTPException(404, "model not found")
    if not info_path.exists():
        raise HTTPException(400, "no info file")

    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError):
        raise HTTPException(400, "could not read info file")

    # A model only makes sense on the dataset it was trained for
    trained_on = info.get("dataset", "MNIST")
    if trained_on != req.dataset:
        raise HTTPException(400, f"model predicts {trained_on}")

    if req.dataset == "MNIST":
        if not req.pixels or len(req.pixels) != 28 or any(len(row) != 28 for row in req.pixels):
            raise HTTPException(400, "pixels must be 28x28")
    elif req.dataset in TEST_SETS:
        if req.index is None:
            raise HTTPException(400, "no image chosen")
    else:
        raise HTTPException(400, f"cannot predict {req.dataset}")

    # The info file says which architecture to build before loading the weights into it
    mtime = path.stat().st_mtime
    if loaded["path"] != path or loaded["mtime"] != mtime:
        try:
            loaded.update(path=path, mtime=mtime, model=load_model(path, info["model"]))
        except Exception:
            traceback.print_exc()
            raise HTTPException(500, "could not load model")

    if req.dataset == "MNIST":
        digit, probabilities = predict(loaded["model"], req.pixels)
        return {"label": digit, "digit": digit, "probabilities": probabilities}

    runner = predict_cifar if req.dataset == "CIFAR-10" else predict_imagenet
    try:
        label, probabilities = runner(loaded["model"], req.index)
        true_label, true_class = TEST_SETS[req.dataset].label(req.index)
    except IndexError as e:
        raise HTTPException(404, str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "could not run the model on this image")

    return prediction(req.dataset, label, probabilities, true_label, true_class)


def prediction(dataset: str, label: int, probabilities: list[float],
               true_label: int | None = None, true_class: str | None = None) -> dict:
    """What a prediction looks like to the UI: the answer, the runners up, the truth."""
    names = TEST_SETS[dataset].classes()
    ranked = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:5]
    return {
        "label": label,
        "class_name": names[label],
        "confidence": probabilities[label],
        "top5": [{"label": i, "class_name": names[i], "probability": probabilities[i]} for i in ranked],
        "true_label": true_label,
        "true_class": true_class,
        # 1000 probabilities is a lot to send for every guess, so only the small sets get them
        "probabilities": probabilities if len(probabilities) <= 10 else None,
    }


@app.post("/api/predict/photo")
async def predict_uploaded_photo(request: Request, model: str, dataset: str = "ImageNet"):
    """Runs a photo the user picked through a model. The body is the raw image file."""
    path = WEIGHTS_DIR / f"{model}.pt"
    info_path = path.with_suffix(".json")
    if not re.fullmatch(r"[\w\- ]+", model) or not path.exists():
        raise HTTPException(404, "model not found")
    if not info_path.exists():
        raise HTTPException(400, "no info file")

    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError):
        raise HTTPException(400, "could not read info file")

    trained_on = info.get("dataset", "MNIST")
    if trained_on != dataset:
        raise HTTPException(400, f"model predicts {trained_on}")
    if dataset != "ImageNet":
        raise HTTPException(400, f"cannot send photos to a {dataset} model")

    data = await request.body()
    if not data:
        raise HTTPException(400, "no image")

    mtime = path.stat().st_mtime
    if loaded["path"] != path or loaded["mtime"] != mtime:
        try:
            loaded.update(path=path, mtime=mtime, model=load_model(path, info["model"]))
        except Exception:
            traceback.print_exc()
            raise HTTPException(500, "could not load model")

    try:
        label, probabilities = predict_photo(loaded["model"], data)
    except Exception:
        traceback.print_exc()
        raise HTTPException(400, "could not read that image")

    result = prediction(dataset, label, probabilities)
    result["image"] = imagenet.preview_from_bytes(data)
    return result


class SaveRequest(BaseModel):
    name: str
    description: str = ""


def git_state() -> dict:
    """Commit the code was at when saving, and whether there were uncommitted changes."""
    def git(*cmd):
        result = subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
    except OSError:
        return {"commit": None, "uncommitted_changes": None}
    return {"commit": commit, "uncommitted_changes": bool(status) if status is not None else None}


def model_info(name: str, description: str, run: dict) -> dict:
    """Everything worth knowing about a saved model, written next to its weights as JSON."""
    model, args = run["model"], run["args"]
    loss, accuracy = run["loss"], run["accuracy"]

    # Every epoch has the same number of batches, so the per-batch loss splits evenly
    batches_per_epoch = len(loss) // args.epochs if args.epochs else 0
    loss_per_epoch = [
        sum(loss[e * batches_per_epoch:(e + 1) * batches_per_epoch]) / batches_per_epoch
        for e in range(args.epochs)
    ] if batches_per_epoch else []

    return {
        "name": name,
        "model": run["model_name"],
        "saved": datetime.now().isoformat(timespec="seconds"),
        "trained": run["finished"].isoformat(timespec="seconds"),
        "description": description.strip(),

        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "layers": {n: list(p.shape) for n, p in model.named_parameters()},
        },

        # What the model predicts; set by the run once other datasets are trainable
        "dataset": run.get("dataset", "MNIST"),
        "args": asdict(args),
        "batches_per_epoch": batches_per_epoch,

        "results": {
            "final_accuracy": accuracy[-1] if accuracy else None,
            "best_accuracy": max(accuracy) if accuracy else None,
            "accuracy_per_epoch": accuracy,
            "final_loss": loss_per_epoch[-1] if loss_per_epoch else None,
            "loss_per_epoch": loss_per_epoch,
            "loss_per_batch": [round(v, 5) for v in loss],
        },

        "environment": {
            "train_seconds": round(run["train_seconds"], 2),
            "device": str(device),
            "torch": t.__version__,
            **git_state(),
        },
    }


def uploaded_info(name: str, model, model_name: str, dataset: str, description: str,
                  accuracy: float | None) -> dict:
    """Info file for weights trained elsewhere: what we were told, plus what the file shows."""
    return {
        "name": name,
        "model": model_name,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "trained": None,
        "description": description.strip(),
        "source": "uploaded",

        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "layers": {n: list(p.shape) for n, p in model.named_parameters()},
        },

        "dataset": dataset,
        "args": None,
        "batches_per_epoch": 0,

        "results": {
            "final_accuracy": accuracy,
            "best_accuracy": accuracy,
            "accuracy_per_epoch": [],
            "final_loss": None,
            "loss_per_epoch": [],
            "loss_per_batch": [],
        },

        "environment": {"device": str(device), "torch": t.__version__},
    }


@app.post("/api/upload")
async def upload_model(
    request: Request,
    name: str,
    model: str,
    dataset: str = "MNIST",
    description: str = "",
    accuracy: float | None = None,
):
    """Saves a weights file trained elsewhere, with the details the wizard collected.

    The body is the raw .pt file, so no multipart parser is needed.
    """
    name = name.strip()
    if not re.fullmatch(r"[\w\- ]+", name):
        raise HTTPException(400, "invalid name")
    if model not in list_architectures():
        raise HTTPException(400, f"unknown architecture {model}")

    WEIGHTS_DIR.mkdir(exist_ok=True)
    path = WEIGHTS_DIR / f"{name}.pt"
    if path.exists():
        raise HTTPException(400, "name already used")

    # Stream the upload to disk so big files never sit in memory twice
    partial = path.with_suffix(".part")
    try:
        with partial.open("wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        if partial.stat().st_size == 0:
            raise HTTPException(400, "no file")

        # Loading it into the chosen architecture is the check that they match
        try:
            loaded_model = load_model(partial, model)
        except Exception:
            traceback.print_exc()
            raise HTTPException(400, f"weights do not fit {model}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    partial.rename(path)
    info = uploaded_info(name, loaded_model, model, dataset, description, accuracy)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))

    return {"ok": True, "path": str(path.relative_to(ROOT))}


@app.post("/api/save")
def save_model(req: SaveRequest):
    if latest_run is None:
        raise HTTPException(400, "no trained model")

    name = req.name.strip()
    if not re.fullmatch(r"[\w\- ]+", name):
        raise HTTPException(400, "invalid name")

    # Weights and info are saved as a pair: weights/<name>.pt + weights/<name>.json
    WEIGHTS_DIR.mkdir(exist_ok=True)
    path = WEIGHTS_DIR / f"{name}.pt"
    t.save(latest_run["model"].state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(model_info(name, req.description, latest_run), indent=2))

    return {"ok": True, "path": str(path.relative_to(ROOT))}


def open_browser_when_ready(url: str):
    """Waits for the server to answer, then opens the app in the default browser."""
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{url}/api/ping", timeout=1)
        except OSError:
            time.sleep(0.5)
            continue
        webbrowser.open(url)
        return


if __name__ == "__main__":
    # Only runs once here, so reloads after code changes don't open more tabs
    threading.Thread(target=open_browser_when_ready, args=("http://127.0.0.1:8000",), daemon=True).start()

    # Reload needs the app as an import string rather than the object
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
