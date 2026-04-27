from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from .models import Step


@dataclass(frozen=True)
class LoadedStep:
    module: str
    step: Step


def load_steps() -> list[LoadedStep]:
    """
    Auto-discover steps under magrun.steps.

    Contract:
    - Each module under magrun.steps may expose a top-level variable `step`
    - Importing a step module must have **no side effects** (no streamlit calls)
    """

    loaded: list[LoadedStep] = []
    pkg = importlib.import_module("magrun.steps")
    for modinfo in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        module = importlib.import_module(modinfo.name)
        if hasattr(module, "step"):
            loaded.append(LoadedStep(module=modinfo.name, step=getattr(module, "step")))

    loaded.sort(key=lambda x: (x.step.meta.category, x.step.meta.name))
    return loaded

