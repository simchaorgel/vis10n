"""The device tensors run on, picked once and shared by everything that needs it."""
import torch as t

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
