# Basic MSINT model try 1
# Imports
import json
from dataclasses import dataclass

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

from dataloaders.mnist import get_mnist


# SimpleMLPTrainingArgs dataclass for model args
@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128).
    """
    trainset_size: int = 10000
    testset_size: int = 1000
    
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3


# Check if GPU is available
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


# Train model
def train(args: SimpleMLPTrainingArgs, model_class: type[nn.Module], on_update = None, cancel = None) -> tuple[list[float], list[float], nn.Module]:
    """
    Trains the model, using training parameters from the `args` object.

    Returns:
        The model, and lists of loss & accuracy.
    """
    # Create model object
    model = model_class().to(device)

    # Load datasets (train and test sets)
    mnist_trainset, test_dataset = get_mnist(args.trainset_size, args.testset_size)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    test_trainloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []

    # List of accuracy for every epoch
    accuracy_list =[]

    # Train n times
    for epoch in range(args.epochs):        
        epoch_accurate_predictions = 0

        # Train
        for i, (imgs, labels) in enumerate(mnist_trainloader):
            # Check if canceled
            if cancel and cancel.is_set():
                break

            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            
            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & send update
            loss_list.append(loss.item())
            if on_update:
                on_update({
                    "type": "batch",
                    "epoch": epoch,                        # 0-based
                    "batch": i + 1,                        # batches done this epoch
                    "batches": len(mnist_trainloader),     # batches per epoch
                    "loss": loss.item(),
                })

        # Check if canceled
        if cancel and cancel.is_set():
            break

        # Validate epoch accuracy
        for imgs, labels in test_trainloader:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            with t.inference_mode():
                logits = model(imgs)

            # Calculate accuracy
            # Get max logit, which is treated as the models guess
            predictions = t.argmax(logits, dim=1)
            # Create tensor with 1 if the prediction was correct, 0 if not, for every img
            accuracy_bools = predictions == labels
            # Number of accurate guesses
            epoch_accurate_predictions += accuracy_bools.sum().item()

            
        # Add accuracy percentage to list
        epoch_accuracy = epoch_accurate_predictions / len(test_dataset)
        accuracy_list.append(epoch_accuracy)

        # Send update
        if on_update:
            on_update({
                "type": "epoch",
                "epoch": epoch,
                "accuracy": epoch_accuracy,
            })

    return loss_list, accuracy_list, model