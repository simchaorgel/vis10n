"""784 → 128 → 10, ReLU (the shape the NumPy train window trains)"""
# Torch twin of the MLP in _numpy_model.py: same layers under the same names,
# so its trained arrays load straight into this as a state dict
import torch.nn as nn
from torch import Tensor

from .components.linear import Flatten, Linear, ReLU


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(28**2, 128)
        self.relu = ReLU()
        self.linear2 = Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        out = self.flatten(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out
