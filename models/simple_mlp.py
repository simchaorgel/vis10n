"""784 → 100 → 10, ReLU"""
# Basic MSINT model try 1
# Imports
import torch.nn as nn
from torch import Tensor

from ._linear_components import Flatten, Linear, ReLU


# Model
#############################################################
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Create object instances in order of use
        self.flatten = Flatten()
        # Layer 1 (takes flattened img input, outputs 100 neuron raw activations)
        self.linear1 = Linear(28**2, 100)
        self.relu = ReLU()
        # Layer 2 (takes 100 activations from prev layer, outputs 10 (guesses for images))
        self.linear2 = Linear(100, 10)


    def forward(self, x: Tensor) -> Tensor:
        # Flatten input (needs to be 1d for the Linear module)
        flattened = self.flatten(x)
        # Pass to layer 1
        pass1 = self.linear1(flattened)
        # ReLU
        relud = self.relu(pass1)
        # Pass to layer 2
        pass2 = self.linear2(relud)

        return pass2
