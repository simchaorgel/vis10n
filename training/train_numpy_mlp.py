# Runs train_loop from the from-scratch NumPy MLP (models/_pure_numpy_model.py) for the NumPy train window,
# then copies its weights into the matching torch model so it saves and loads like any other
from dataclasses import dataclass

import torch as t

from models._pure_numpy_model import MLP, train_loop
from models.numpy_mlp import Model as TorchMLP


@dataclass
class NumpyMLPTrainingArgs:
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 0.1


class Cancelled(Exception):
    """Raised from the epoch callback to stop train_loop, which has no cancel of its own."""


def to_torch(model: MLP) -> t.nn.Module:
    """A torch model holding the NumPy model's trained weights."""
    state_dict = {
        f"{layer_name}.{param_name}": t.tensor(param.array, dtype=t.float32)
        for layer_name, layer in model._modules.items()
        for param_name, param in layer._parameters.items()
    }
    torch_model = TorchMLP()
    torch_model.load_state_dict(state_dict)
    return torch_model


def train(args: NumpyMLPTrainingArgs, on_update=None, cancel=None) -> tuple[list[float], list[float], t.nn.Module | None]:
    """
    Same updates and return as training/train_mnist_mlp.py's train(), so the server streams it the same way.
    train_loop only reports at the end of each epoch, so that epoch's batch losses are all sent then,
    and a cancel takes effect at the next epoch end.

    Returns:
        Lists of loss & accuracy, and the trained model as a torch model (None if cancelled).
    """
    def on_epoch_end(epoch_num, train_losses, test_loss, test_acc):
        if on_update:
            for i, loss in enumerate(train_losses):
                on_update({"type": "batch", "epoch": epoch_num, "batch": i + 1,
                           "batches": len(train_losses), "loss": float(loss)})
            on_update({"type": "epoch", "epoch": epoch_num, "accuracy": test_acc})
        if cancel and cancel.is_set():
            raise Cancelled

    try:
        model, loss_history, accuracy_history = train_loop(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            on_epoch_end=on_epoch_end,
        )
    except Cancelled:
        return [], [], None

    loss = [float(v) for epoch_losses in loss_history for v in epoch_losses]
    return loss, accuracy_history, to_torch(model)
