"""CelebA faces for GAN training: downloaded once from Hugging Face into data/celeba,
then read with ImageFolder at 64x64, scaled to -1..1."""
from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CELEB_DIR = ROOT / "data" / "celeba"
IMAGE_DIR = CELEB_DIR / "img_align_celeba"   # ImageFolder's single "class"
SAVED = CELEB_DIR / ".saved"                 # how many images are saved, from 000000.jpg on
CELEBA_SIZE = 202_599                        # images in the full set
IMAGE_SIZE = 64


def saved_count() -> int:
    """How many images are saved so far (0 if none)."""
    try:
        return int(SAVED.read_text())
    except (FileNotFoundError, ValueError):
        return 0


def download(samples: int | None = None):
    """
    Saves the first `samples` CelebA images (all of them if None) to data/celeba/img_align_celeba.
    Only images not saved yet are written, so asking for more later just adds the rest.
    """
    wanted = CELEBA_SIZE if samples is None else min(samples, CELEBA_SIZE)
    have = saved_count()
    if have >= wanted:
        return
    from datasets import load_dataset   # Hugging Face; only needed when there's something to save

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset("nielsr/CelebA-faces")["train"].select(range(have, wanted))
    for idx, item in enumerate(tqdm(train, initial=have, total=wanted, desc="Saving imgs", ascii=True), start=have):
        item["image"].save(IMAGE_DIR / f"{idx:06}.jpg")
        # Recorded as it goes, so an interrupted download carries on from about where it stopped
        if (idx + 1) % 1000 == 0:
            SAVED.write_text(str(idx + 1))
    SAVED.write_text(str(wanted))


def get_dataset(samples: int | None = None, train: bool = True) -> Dataset:
    """The first `samples` CelebA images (all of them if None), downloading any not saved yet."""
    assert train, "CelebA dataset only has a training set"
    download(samples)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = ImageFolder(root=CELEB_DIR, transform=transform)
    # More may be saved than asked for, from an earlier bigger run; files sort by name, so these are the first ones
    return dataset if samples is None else Subset(dataset, range(min(samples, len(dataset))))
