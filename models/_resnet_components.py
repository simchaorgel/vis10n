# Imports
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int

# General Componenets
#######################################################
class ReLU(nn.Module):
    '''
    Takes a tensor and returns the ReLU: all negatives -> 0
    '''
    def forward(self, x: Tensor) -> Tensor:
        return t.clamp(x, min=0)

class Linear(nn.Module):
    '''
    Simple linear layer. Stores weights and optional biases
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

class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)  # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)  # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: Tensor) -> Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x
    
# ResNet componenets
###########################################################
class Conv2d(nn.Module):
    '''
    Same as torch.nn.Conv2d with bias=False.
    Assumes kernel is square
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Calculate Kaiming range and create random weights
        kaiming_upper = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        weight = t.rand(out_channels, in_channels, kernel_size, kernel_size) * (kaiming_upper * 2) + -kaiming_upper
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d (uses torch's conv2d)."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d."""
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])


class BatchNorm2d(nn.Module):
    # The type hints below purely documentation
    running_mean: Float[Tensor, " num_features"]
    running_var: Float[Tensor, " num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        # Reshape function to respape 1d to (1 data 1 1), to match x channels
        reshape = lambda input: einops.rearrange(input, "channels -> 1 channels 1 1")
        
        if self.training:
            # Get batch mean and var
            batch_mean = x.mean(dim=(0,2,3))
            batch_var = x.var(unbiased=False, dim=(0, 2, 3))

            # Update running buffers
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            self.num_batches_tracked += 1

            # Define variables
            mean = reshape(batch_mean)
            var = reshape(batch_var)
        else:
            mean = reshape(self.running_mean)
            var = reshape(self.running_var)

        # Norm
        x_normed = ((x - mean) / (t.sqrt(var + self.eps)))
        x_affine = x_normed * reshape(self.weight) + reshape(self.bias)     
        return x_affine

    def extra_repr(self) -> str:
        return f"Num features: {self.num_features}"

class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return einops.reduce(x, "b c h w -> b c", "mean")


# Resnet componenets assembly
#############################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a
        `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right
        branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        is_shape_preserving = (first_stride == 1) and (in_feats == out_feats)  # determines if right branch is identity

        # Left branch
        self.left_branch = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats)
        )

        # Right branch
        if is_shape_preserving:
            self.right_branch = nn.Identity()
        else:
            self.right_branch = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, padding=0),
                BatchNorm2d(out_feats)
            )

        self.ReLU = ReLU()
        

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass. If no downsampling block is present, the addition should just add
        the left branch's output to the input.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)
        """
        left_out = self.left_branch(x)
        right_out = self.right_branch(x)
        left_right_added = left_out + right_out
        out = self.ReLU(left_right_added)

        #assert
        return out

class BlockGroup(nn.Module):
    """
    An n_blocks-long sequence of ResidualBlock where only the first block uses the provided
    stride.
    """
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        blocks = []
        for i in range(n_blocks - 1):
            blocks.append(ResidualBlock(out_feats, out_feats))

        self.blocks = Sequential(
            # first block
            ResidualBlock(in_feats, out_feats, first_stride),
            # other blocks
            *blocks
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.
        x: shape (batch, in_feats, height, width)
        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        out = self.blocks(x)
        return out
