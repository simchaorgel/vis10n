"""Reads the CIFAR-10 test set straight from the python batches in data/.

The batch files are plain pickles: b"data" is (N, 3072) uint8 (32x32 red, then green,
then blue) and b"labels" is the class index of each row.
"""
import base64
import io
import pickle
import random
from pathlib import Path

import numpy as np
import torch as t
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CIFAR_DIR = ROOT / "data" / "cifar-10-batches-py"
SIZE = 32

# Must match how the model was trained - see training/finetune_cifar_resnet.ipynb.
# The CIFAR models are finetuned from ImageNet weights, so images are scaled up to 224
# and normalised with ImageNet statistics rather than CIFAR ones.
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Loaded on the first request and kept for the life of the server
_test_set = None


def _load_test_set() -> tuple[np.ndarray, list[int], list[str]]:
    """(images as (N, 32, 32, 3) uint8, labels, class names)."""
    global _test_set
    if _test_set is None:
        if not CIFAR_DIR.exists():
            raise FileNotFoundError(
                f"{CIFAR_DIR} not found - extract data/cifar-10-python.tar.gz into data/"
            )
        with (CIFAR_DIR / "test_batch").open("rb") as f:
            batch = pickle.load(f, encoding="bytes")
        with (CIFAR_DIR / "batches.meta").open("rb") as f:
            meta = pickle.load(f, encoding="bytes")

        images = batch[b"data"].reshape(-1, 3, SIZE, SIZE).transpose(0, 2, 3, 1)
        names = [name.decode() for name in meta[b"label_names"]]
        _test_set = (images, list(batch[b"labels"]), names)
    return _test_set


def size() -> int:
    return len(_load_test_set()[1])


def sample(index: int | None = None) -> dict:
    """One test image as a PNG data URL, with its index and true class."""
    images, labels, names = _load_test_set()
    if index is None:
        index = random.randrange(len(labels))
    if not 0 <= index < len(labels):
        raise IndexError(f"index out of range: {index}")

    buffer = io.BytesIO()
    Image.fromarray(images[index]).save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    return {
        "dataset": "CIFAR-10",
        "index": index,
        "label": int(labels[index]),
        "class_name": names[labels[index]],
        "size": SIZE,
        "image": data_url,
    }


def classes() -> list[str]:
    """The ten class names, in label order."""
    return _load_test_set()[2]


def label(index: int) -> tuple[int, str]:
    """The true label of a test image, for checking a prediction against."""
    _, labels, names = _load_test_set()
    if not 0 <= index < len(labels):
        raise IndexError(f"index out of range: {index}")
    return int(labels[index]), names[labels[index]]


def as_tensor(index: int) -> t.Tensor:
    """A test image ready for a model: (1, 3, 224, 224), preprocessed like training data."""
    images, labels, _ = _load_test_set()
    if not 0 <= index < len(labels):
        raise IndexError(f"index out of range: {index}")
    return CIFAR_TRANSFORM(images[index]).unsqueeze(0)


def pixels(index: int) -> list[list[list[int]]]:
    """The raw 32x32x3 values, for when a model is asked to predict this image."""
    images, labels, _ = _load_test_set()
    if not 0 <= index < len(labels):
        raise IndexError(f"index out of range: {index}")
    return images[index].tolist()
