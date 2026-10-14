# Main imports
import torch as t
from dataclasses import dataclass
from torch.utils.data import DataLoader
from jaxtyping import Float
from typing import Callable
from torch import Tensor, nn

# Local imports
from dataloaders import mnist
from models.mnist_vae import VAE

# Check if we have access to a GPU
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

@dataclass
class VAEArgs:
    # architecture
    latent_dim_size: int = 5   # If edited, model cannot be run in UI
    hidden_dim_size: int = 128 # If edited, model cannot be run in UI

    # data / training
    trainset_size: int = 60_000
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)
    beta_kl: float = 0.1

    # logging
    log_every_n_steps: int = 50


class VAETrainer:
    def __init__(self, args: VAEArgs, on_update: Callable = None, cancel=None):
        # Tranining args and run args
        self.args = args
        self.on_update = on_update
        self.cancel = cancel

        # Get data
        self.trainset, testset = mnist.get_mnist(trainset_size=args.trainset_size, testset_size=100)
        # One test image of each digit, 0-9, reconstructed at every log
        holdout = {}
        for img, label in testset:
            holdout.setdefault(label, img)
            if len(holdout) == 10:
                break
        self.holdout = t.stack([holdout[d] for d in range(10)]).to(device)   # (10, 1, 28, 28)
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Initialise model
        self.model = VAE(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)

        # Tracking
        self.losses = []

    def training_step(
        self, img: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, ""]:
        """
        Performs a training step on the batch of images in `img`. Returns the loss.
        if enabled.
        """
        # Forward pass (1 batch)
        decoded_img, mu, logsigma = self.model(img)

        # Loss: difference between img | output, AND KL divergence
        kl_divergence = ((t.exp(logsigma)**2 + mu**2 - 1) / 2 - logsigma).mean() * self.args.beta_kl
        reconstruct_to_img_loss = nn.MSELoss()(decoded_img, img)
        loss =  reconstruct_to_img_loss + kl_divergence

        # Back pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Logging (loss) (increment step)
        self.step += 1
        self.losses.append(loss.item())

        if self.on_update:
            self.on_update({
                "type": "batch",
                "epoch": self.epoch,
                "batch": self.batch,
                "batches": len(self.trainloader),
                "loss": loss.item(),
                "reconstruction": reconstruct_to_img_loss.item(),
                "kl": kl_divergence.item(),
            })

        return loss

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Evaluates model on holdout data, either logging to wandb or displaying output inline.
        """
        assert self.step > 0, "First call should come after a training step. Remember to increment `self.step`."
        output = self.model(self.holdout)[0]
        if self.on_update:
            pixels = ((output * 0.3081 + 0.1307).clamp(0, 1) * 255).round().byte().squeeze(1)   # back to 0-255
            self.on_update({"type": "samples", "step": self.step, "images": pixels.tolist()})    # 10 x 28 x 28


    def train(self) -> VAE:
        """Performs a full training run."""
        self.step = 0

        # Loop over epochs
        for self.epoch in range(self.args.epochs):
            # Loop over batches
            for self.batch, (images, labels) in enumerate(self.trainloader, 1):
                # Check if run has been canceled
                if self.cancel and self.cancel.is_set():
                    return self.model
                
                # Train step
                images = images.to(device)
                loss = self.training_step(images)

                # Every n steps, evaluate as well
                if self.step % self.args.log_every_n_steps == 0:
                    self.log_samples()

        return self.model


def train(args: VAEArgs, on_update=None, cancel=None) -> tuple[list[float], VAE]:
    trainer = VAETrainer(args, on_update=on_update, cancel=cancel)
    model = trainer.train()
    return trainer.losses, model
