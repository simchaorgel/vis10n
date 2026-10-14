# Model architectures: every file in this folder or in variations/ (not starting with _) that defines a
# `Model` class, or exactly one nn.Module class. An architecture's name is its file name alone, so a file
# can move between the two folders without breaking the weights saved for it. components/ holds layers.
import importlib
import pkgutil
from pathlib import Path

import torch.nn as nn

# Where architectures are looked for, as (import prefix, folder); the first folder wins a name clash
SEARCH = [("", Path(__file__).parent), ("variations.", Path(__file__).parent / "variations")]


def _files() -> dict[str, str]:
    """Architecture name -> module path, for every candidate file (skips files starting with _)."""
    found = {}
    for prefix, folder in SEARCH:
        for m in pkgutil.iter_modules([str(folder)]):
            if not m.ispkg and not m.name.startswith("_"):
                found.setdefault(m.name, f"{__name__}.{prefix}{m.name}")
    return found


def _model_class(module):
    """The module's Model class, or the one nn.Module class it defines (not the layers it imports); else None."""
    if hasattr(module, "Model"):
        return module.Model
    defined = [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj.__module__ == module.__name__
    ]
    return defined[0] if len(defined) == 1 else None


def get_module(name: str):
    """The module that defines architecture `name` (for its docstring, say)."""
    path = _files().get(name)
    if path is None:
        raise ValueError(f"no architecture called {name}")
    return importlib.import_module(path)


def list_models() -> list[str]:
    """Names of all architectures. Files with no torch model (like the pure NumPy one) are left out;
    files that fail to import are kept, so the error shows up where they're listed."""
    names = []
    for name in _files():
        try:
            if _model_class(get_module(name)) is None:
                continue
        except Exception:
            pass
        names.append(name)
    return sorted(names)


def get_model(name: str):
    """The Model class of architecture `name`, or the one nn.Module class its file defines if it has no Model."""
    model_class = _model_class(get_module(name))
    if model_class is None:
        raise AttributeError(f"models/{name}.py has no Model class")
    return model_class
