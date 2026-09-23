"""Something"""
import torch.nn as nn
from torch import Tensor

from ._resnet_components import *

# Resnet34
##################################################################
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        out_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        # Stem
        self.stem = Sequential(
            # 7x7 conv
            Conv2d(in_channels=3,
                   out_channels=out_feats0,
                   kernel_size=7,
                   stride=2,
                   padding=3),
            BatchNorm2d(out_feats0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride= 2)
        )

        # Blocks
        blockgroups = []
        for i in range(len(n_blocks_per_group)):
            in_feats = out_features_per_group[i-1] if i > 0 else out_feats0
            blockgroups.append(
                BlockGroup(n_blocks_per_group[i], in_feats, out_features_per_group[i], first_strides_per_group[i])
            )
        
        self.blockgroups = Sequential(
            *blockgroups
        )

        # End
        self.out_layers = Sequential(
            AveragePool(),
            Linear(out_features_per_group[-1], self.n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        out = self.stem(x)
        out = self.blockgroups(out)
        out = self.out_layers(out)
        return out