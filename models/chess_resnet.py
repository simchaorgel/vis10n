"""Something"""
import torch.nn as nn
from torch import Tensor

from ._resnet_components import *
from ._linear_components import Flatten

class ChessResnet(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        # Input
        # (B, 12, 8, 8) (Batch, 12 pieces, 8x8 board)
        # Stem
        self.stem = Sequential(
            Conv2d(in_channels=12,
                   out_channels=64,
                   kernel_size=3,
                   padding=1,
            ),
            BatchNorm2d(64),
            ReLU(),
        )

        # Blocks
        self.blocks = Sequential(*[ResidualBlock(64, 64) for _ in range(6)])

        # End
        self.end_layers = Sequential(
            Conv2d(64, 32, 1),
            BatchNorm2d(32),
            ReLU(),
            Flatten(),
            Linear(32 * 8 * 8, 1792)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.stem(x)
        out = self.blocks(out)
        out = self.end_layers(out)
        return out
