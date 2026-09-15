# Basic MSINT model try 1
# Imports
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Model componenets
#############################################################

# ReLU
class ReLU(nn.Module):
    '''
    Takes a Tensor and returns the ReLU: all negatives -> 0
    '''
    def forward(self, x: Tensor) -> Tensor: 
        return t.maximum(x, t.tensor([0]))


# Linear pass
class Linear(nn.Module):
    '''
    Simple linear transformation.
    Stores weights and biases
    The fields should be named `weight` and `bias` for compatibility with PyTorch.
    If `bias` is False, set `self.bias` to None.
    '''
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        # Stats
        self.in_features = in_features
        self.out_features = out_features

        # Compute weight bounds
        high = 1 / in_features ** 0.5
        low = -high
        # Create random weights, and then shift them to be between the Kaiming bounds
        weight = t.rand(out_features, in_features) * (high - low) + low
        self.weight = nn.Parameter(weight)

        # Create bias if required
        if bias:
            # Create biases and shape to Kaiming bounds
            self.bias = t.rand(out_features) * (high - low) + low
            self.bias = nn.Parameter(self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        # Matmul the inputs to the weights (transformed to match)
        raw_activations = x @ self.weight.T

        # Add bias if there is one
        if self.bias is not None:
            raw_activations += self.bias

        return raw_activations

    def extra_repr(self) -> str:
            return f"In features: {self.in_features}, Out features: {self.out_features}, Bias: {self.bias is not None}"


# Flatten
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])



# Model
#############################################################
class SimpleMLP(nn.Module):
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
