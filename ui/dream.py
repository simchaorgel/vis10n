# Activation maximisation: freeze the model and run gradient ascent on the input image
# instead of the weights, to find the image the model thinks is the most "digit-like"
# Imports
import torch as t
import torch.nn as nn

from device import device

# Same normalisation as MNIST_TRANSFORM, applied by hand so gradients flow back to the pixels
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def total_variation(x: t.Tensor) -> t.Tensor:
    """How much neighbouring pixels differ; penalising it keeps strokes smooth instead of speckled."""
    return (x[..., 1:, :] - x[..., :-1, :]).abs().sum() + (x[..., :, 1:] - x[..., :, :-1]).abs().sum()


def dream(
    model: nn.Module,
    digit: int,
    steps: int = 300,
    lr: float = 0.05,
    l2_weight: float = 0.1,
    tv_weight: float = 0.05,
    jitter: int = 1,
    clamp: bool = True,
) -> tuple[list[list[int]], list[float]]:
    """
    Starts from a nearly black image and nudges its pixels to raise the model's logit for `digit`.

    The logit is maximised rather than the softmax probability: the probability can also be
    raised by pushing the other nine digits down, which gives messier images.
    l2_weight keeps the image sparse like a real drawing, tv_weight keeps it smooth,
    and jitter shifts the image a random pixel or two each step so it can't rely on exact positions.
    clamp keeps pixels in 0..1 like a real image. With all of them off (0 / False) it's pure
    gradient ascent on the pixels.

    Returns:
        The image as 28x28 ints 0-255 (white digit on black), and the model's probabilities for it.
        Unclamped pixels can go far outside 0..1, so then the image is stretched to fit 0-255.
    """
    # Pixels in 0..1, like an MNIST image before normalisation; the only thing being optimised
    x = (t.rand(1, 1, 28, 28, device=device) * 0.1).requires_grad_(True)
    optimizer = t.optim.Adam([x], lr=lr)

    for _ in range(steps):
        dy, dx = t.randint(-jitter, jitter + 1, (2,)).tolist()
        shifted = t.roll(x, shifts=(dy, dx), dims=(2, 3))

        logits = model((shifted - MNIST_MEAN) / MNIST_STD)
        loss = -logits[0, digit] + l2_weight * x.pow(2).sum() + tv_weight * total_variation(x)

        # Gradient for the image only, so the model's weights are never touched
        optimizer.zero_grad()
        x.grad, = t.autograd.grad(loss, x)
        optimizer.step()

        # Keep pixels in the range a real image could have
        if clamp:
            with t.no_grad():
                x.clamp_(0, 1)

    with t.inference_mode():
        probabilities = t.softmax(model((x - MNIST_MEAN) / MNIST_STD), dim=1)[0]

    image = x[0, 0].detach()
    if not clamp:
        image = (image - image.min()) / (image.max() - image.min()).clamp_min(1e-8)
    pixels = (image * 255).round().to(t.uint8).tolist()
    return pixels, probabilities.tolist()
