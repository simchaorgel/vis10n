"""VAE"""
import torch as t
from torch import Tensor, nn
from torch.nn import Conv2d, Linear, ReLU, Sequential
from einops.layers.torch import Rearrange
from jaxtyping import Float


class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, latent_dim_size: int = 5, hidden_dim_size: int = 128):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size
        
        self.encoder = Sequential(
            # Input is 28x28
            Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Rearrange("b c h w -> b (c h w)"),
            Linear(in_features=7 * 7 * 32, out_features=hidden_dim_size),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=2 * latent_dim_size),
            # Rearragne so 'v' - mean or logstd - is first. The logits represent two outputs
            Rearrange("b (v latent_dim) -> v b latent_dim", v=2)
        )

        self.decoder = Sequential(
            Linear(latent_dim_size, hidden_dim_size),
            ReLU(),
            Linear(hidden_dim_size, 32 * 7 * 7),
            Rearrange("b (c h w) -> b c h w", c=32, h=7, w=7),
            ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1, bias=False)
        )

    def sample_latent_vector(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> tuple[
        Float[Tensor, "batch latent"],
        Float[Tensor, "batch latent"],
        Float[Tensor, "batch latent"],
    ]:
        """
        Passes `x` through the encoder, returns tuple of (sampled latent vector, mean, log std dev).
        This function can be used in `forward`, but also used on its own to generate samples for
        evaluation.
        """
        mean, logsigma = self.encoder(x)
        std = t.exp(logsigma)
        sample = mean + std * t.randn_like(std)
        return sample, mean, logsigma


    def forward(
        self, x: Float[Tensor, "batch 1 height width"]
    ) -> tuple[
        Float[Tensor, "batch 1 height width"],
        Float[Tensor, "batch latent"],
        Float[Tensor, "batch latent"],
    ]:
        """
        Passes `x` through the encoder and decoder. Returns the reconstructed input, as well as mu
        and logsigma.
        """
        random_encoded_sample, mu, logsigma = self.sample_latent_vector(x)
        decoded_img = self.decoder(random_encoded_sample)
        return decoded_img, mu, logsigma

