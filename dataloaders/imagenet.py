"""ImageNet-1k: the class names, the preprocessing, and Imagenette as a test set.

The full ImageNet validation set needs registration, so the shuffle button samples
Imagenette instead: 3,925 real photos from 10 of the 1000 classes, which is enough to
check a 1000-class model end to end. Any other photo can be sent in through predict().
"""
import base64
import io
import random
from pathlib import Path

import torch as t
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet34_Weights

ROOT = Path(__file__).resolve().parents[1]
IMAGENETTE_DIR = ROOT / "data" / "imagenette2-160" / "val"
IMAGE_SIZE = 224

# The standard ImageNet evaluation pipeline the pretrained weights were measured with
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Imagenette's folder names, and where each one sits in the 1000 ImageNet classes
WNID_TO_LABEL = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

_files = None


def classes() -> list[str]:
    """All 1000 ImageNet class names, in label order."""
    return ResNet34_Weights.IMAGENET1K_V1.meta["categories"]


def _test_files() -> list[tuple[Path, int]]:
    """Every Imagenette validation image with its ImageNet label, found once and kept."""
    global _files
    if _files is None:
        if not IMAGENETTE_DIR.exists():
            raise FileNotFoundError(
                f"{IMAGENETTE_DIR} not found - run dataloaders/imagenet.py to download Imagenette"
            )
        _files = sorted(
            (path, label)
            for wnid, label in WNID_TO_LABEL.items()
            for path in (IMAGENETTE_DIR / wnid).glob("*.JPEG")
        )
    return _files


def size() -> int:
    return len(_test_files())


def label(index: int) -> tuple[int, str]:
    """The true label of a test image, for checking a prediction against."""
    files = _test_files()
    if not 0 <= index < len(files):
        raise IndexError(f"index out of range: {index}")
    return files[index][1], classes()[files[index][1]]


def sample(index: int | None = None) -> dict:
    """One test image as a PNG data URL, with its index and true class."""
    files = _test_files()
    if index is None:
        index = random.randrange(len(files))
    if not 0 <= index < len(files):
        raise IndexError(f"index out of range: {index}")

    path, true_label = files[index]
    # Shown exactly as the model sees it: resized and centre cropped, before normalising
    image = transforms.CenterCrop(IMAGE_SIZE)(transforms.Resize(256)(Image.open(path).convert("RGB")))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return {
        "dataset": "ImageNet",
        "index": index,
        "label": true_label,
        "class_name": classes()[true_label],
        "size": IMAGE_SIZE,
        "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
    }


def as_tensor(index: int) -> t.Tensor:
    """A test image ready for a model: (1, 3, 224, 224), preprocessed like ImageNet eval."""
    files = _test_files()
    if not 0 <= index < len(files):
        raise IndexError(f"index out of range: {index}")
    return IMAGENET_TRANSFORM(Image.open(files[index][0]).convert("RGB")).unsqueeze(0)


def tensor_from_bytes(data: bytes) -> t.Tensor:
    """The same preprocessing for a photo someone picked themselves."""
    return IMAGENET_TRANSFORM(Image.open(io.BytesIO(data)).convert("RGB")).unsqueeze(0)


def preview_from_bytes(data: bytes) -> str:
    """That photo as a PNG data URL, cropped the way the model will see it."""
    image = transforms.CenterCrop(IMAGE_SIZE)(
        transforms.Resize(256)(Image.open(io.BytesIO(data)).convert("RGB"))
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def download() -> None:
    """Fetches Imagenette (~99MB) into data/ - only needed once."""
    import tarfile
    import urllib.request

    archive = ROOT / "data" / "imagenette2-160.tgz"
    if not archive.exists():
        print("downloading imagenette2-160.tgz ...")
        urllib.request.urlretrieve(
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", archive
        )
    with tarfile.open(archive) as f:
        f.extractall(ROOT / "data")
    print(f"ready: {size()} test images")


if __name__ == "__main__":
    download()
