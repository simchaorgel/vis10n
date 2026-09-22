# Model architectures: every file in this folder (not starting with _) defines a `Model` class
import importlib
import pkgutil


def list_models() -> list[str]:
    """Names of all architecture files in this folder (skips files starting with _)."""
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))


def get_model(name: str):
    """The Model class defined in models/<name>.py."""
    return importlib.import_module(f"{__name__}.{name}").Model
