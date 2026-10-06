# Model architectures: every file in this folder (not starting with _) defines a `Model` class
import importlib
import pkgutil

import torch.nn as nn


def list_models() -> list[str]:
    """Names of all architecture files in this folder (skips files starting with _)."""
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))


def get_model(name: str):
    """The Model class defined in models/<name>.py, or the one nn.Module class it defines if it has no Model."""
    module = importlib.import_module(f"{__name__}.{name}")
    if hasattr(module, "Model"):
        return module.Model

    # Only classes written in that file, not the layers it imports
    defined = [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj.__module__ == module.__name__
    ]
    if len(defined) != 1:
        raise AttributeError(f"models/{name}.py has no Model class")
    return defined[0]
