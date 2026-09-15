# VIS10N
<p>
  <img src="screenshots/screenshot1.png" width="49%">
  <img src="screenshots/screenshot2.png" width="49%">
</p>

This project is my basic implementation of training and running hand-drawn digit classification models. The model contains 3 layers (28*28, 100, 10) and is trained on the MNIST dataset.
The model structure is based off the ARENA (https://arena.education) courses guide.

## Credits
- Model python code - myself
- App code (HTML, FastAPI) - Claude Opus 5
- predict.py, image preprocessing = Claude
- UI design - myself, background pixel art - Claude

## Setup and running
Create and activate a venv:
```
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate       # macOS / Linux
```

Install requirements to venv:
```
pip install -r requirements.txt
```

Run the app, opens on http://localhost:8000:
```
python main.py
```

## Using the app
**Train model** – set the training args, then drag the slider to start. Loss (per batch) and accuracy (per epoch) are plotted live, and training can be cancelled by dragging the slider again. When training finishes, enter a name and save the model. The MNIST dataset downloads to `data/` on the first run.

**Test model** – pick a saved model, draw a digit on the 28x28 canvas and submit. The model's prediction and its confidence are shown on the right.

## How it works
The frontend (`window.html`) talks to a FastAPI server (`main.py`). Training runs in a background thread and streams progress to the page over a websocket. Saved models are stored as PyTorch state dicts in `models/`.

Before predicting, drawings are preprocessed to look like MNIST digits: cropped, scaled to fit a 20x20 box, centred by centre of mass in a 28x28 image and lightly blurred.

## Project structure
- `model.py` – the SimpleMLP model and its layers
- `dataset.py` – MNIST loading and transforms
- `train_model.py` – training args and training loop
- `predict.py` – loading saved models, preprocessing and prediction
- `main.py` – FastAPI server
- `window.html` – the UI