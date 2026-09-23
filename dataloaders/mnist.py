# Load or dowload the MSIST datasets
# Imports
from torchvision import datasets, transforms
from torch.utils.data import Subset
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Transform
MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=MNIST_TRANSFORM)

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset

