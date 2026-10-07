# Load saved SimpleMLP weights and run single images through the model
# Imports
import numpy as np
import torch as t
from PIL import Image, ImageFilter

import torch.nn as nn

from dataloaders import cifar, imagenet
from dataloaders.mnist import MNIST_TRANSFORM
from models import get_model
from device import device


def load_model(path, model_name: str) -> nn.Module:
    """
    Creates a fresh model with the architecture in models/<model_name>.py
    and fills it with the weights saved at `path`.
    """
    model = get_model(model_name)().to(device)

    # map_location puts the weights on this machine's device, whatever device they were saved from
    state_dict = t.load(path, map_location=device)
    model.load_state_dict(state_dict)

    # Switch to evaluation mode (matters for layers like dropout / batchnorm if they get added later)
    model.eval()
    return model


def shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Moves the image down by dy and right by dx (negative = up / left), filling the gap with black.
    """
    h, w = img.shape
    out = np.zeros_like(img)
    src = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out


def preprocess(pixels) -> np.ndarray:
    """
    Makes a drawing look like an MNIST digit, following how MNIST itself was made:
    crop to the digit, scale its longest side to 20px, soften the edges,
    and centre it in 28x28 by its centre of mass.

    pixels: 28x28 grid of ints 0-255, white digit on black
    Returns: 28x28 uint8 array
    """
    img = np.array(pixels, dtype=np.uint8)

    # Nothing drawn: nothing to crop or centre
    ys, xs = np.nonzero(img)
    if len(xs) == 0:
        return img

    # Crop to the bounding box of the digit
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Scale so the longest side is 20px, keeping the aspect ratio
    # (bilinear resampling gives the soft grey edges MNIST digits have)
    h, w = crop.shape
    scale = 20 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = np.array(Image.fromarray(crop).resize((new_w, new_h), Image.BILINEAR))

    # Paste into the middle of a black 28x28 image
    canvas = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # Move the centre of mass (brightness-weighted average position) to the centre
    weights = canvas.astype(np.float32)
    total = weights.sum()
    cy = (weights.sum(axis=1) * np.arange(28)).sum() / total
    cx = (weights.sum(axis=0) * np.arange(28)).sum() / total
    canvas = shift(canvas, round(13.5 - cy), round(13.5 - cx))

    # Light blur to soften the hard canvas strokes, then stretch back to full brightness
    blurred = np.array(Image.fromarray(canvas).filter(ImageFilter.GaussianBlur(radius=0.5)), dtype=np.float32)
    blurred = blurred / blurred.max() * 255

    return blurred.astype(np.uint8)


def predict(model: nn.Module, pixels) -> tuple[int, list[float]]:
    """
    pixels: 28x28 grid of ints 0-255, white digit on black

    Returns:
        The predicted digit, and the probability for each digit 0-9.
    """
    # Make the drawing MNIST-like, then apply the same transform as training
    img = preprocess(pixels)                    # (28, 28) uint8
    x = MNIST_TRANSFORM(img)                    # (1, 28, 28)

    # Add a batch dimension: the model expects a batch of images, here a batch of one
    x = x.unsqueeze(0).to(device)               # (1, 1, 28, 28)

    with t.inference_mode():
        logits = model(x)                       # (1, 10)

    # Softmax turns raw logits into probabilities that sum to 1
    probabilities = t.softmax(logits, dim=1)[0] # (10,)
    digit = probabilities.argmax().item()

    return digit, probabilities.tolist()


def predict_cifar(model: nn.Module, index: int) -> tuple[int, list[float]]:
    """
    index: which image of the CIFAR-10 test set to run

    Returns:
        The predicted class index, and the probability for each of the ten classes.
    """
    # Already normalised the same way training data is, so no preprocessing here
    x = cifar.as_tensor(index).to(device)       # (1, 3, 32, 32)

    with t.inference_mode():
        logits = model(x)                       # (1, 10)

    probabilities = t.softmax(logits, dim=1)[0]
    return probabilities.argmax().item(), probabilities.tolist()


def predict_tensor(model: nn.Module, x: t.Tensor) -> tuple[int, list[float]]:
    """
    x: an already preprocessed batch of one, (1, channels, height, width)

    Returns:
        The predicted class index, and the probability of every class.
    """
    with t.inference_mode():
        logits = model(x.to(device))

    probabilities = t.softmax(logits, dim=1)[0]
    return probabilities.argmax().item(), probabilities.tolist()


def predict_imagenet(model: nn.Module, index: int) -> tuple[int, list[float]]:
    """Runs one image of the Imagenette test set through a 1000-class model."""
    return predict_tensor(model, imagenet.as_tensor(index))


def predict_photo(model: nn.Module, data: bytes) -> tuple[int, list[float]]:
    """Runs a photo someone picked themselves through a 1000-class model."""
    return predict_tensor(model, imagenet.tensor_from_bytes(data))
