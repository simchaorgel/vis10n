"""Optimizers: every class in optimizers.py with a step() and a zero_grad()."""
import inspect

from . import optimizers


def _classes() -> dict[str, type]:
    return {
        name: obj
        for name, obj in inspect.getmembers(optimizers, inspect.isclass)
        if obj.__module__ == optimizers.__name__
        and hasattr(obj, "step")
        and hasattr(obj, "zero_grad")
    }


def list_optimizers() -> list[str]:
    """Names of the optimizers that can be trained with."""
    return sorted(_classes())


def get_optimizer(name: str) -> type:
    """The optimizer class called `name`, e.g. get_optimizer("SGD")."""
    found = _classes().get(name)
    if found is None:
        raise ValueError(f"unknown optimizer {name}, expected one of {list_optimizers()}")
    return found


# Re-export every optimizer class, so `from optimizers import SGD` and
# `from optimizers import *` work without naming the inner module
globals().update(_classes())
__all__ = list_optimizers()
