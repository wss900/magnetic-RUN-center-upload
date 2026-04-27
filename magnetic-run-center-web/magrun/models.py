from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import pandas as pd


@dataclass(frozen=True)
class StepParam:
    key: str
    label: str
    kind: str  # "int" | "float" | "bool" | "str" | "select"
    default: Any
    options: list[str] | None = None
    help: str | None = None


@dataclass(frozen=True)
class StepMeta:
    id: str
    name: str
    category: str
    description: str
    file_types: list[str]
    params: list[StepParam] = field(default_factory=list)


@dataclass
class StepOutputs:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    downloads: dict[str, tuple[str, bytes, str]] = field(default_factory=dict)  # name -> (filename, payload, mime)
    images: dict[str, tuple[str, bytes, str]] = field(default_factory=dict)  # name -> (filename, payload, mime)
    notes: list[str] = field(default_factory=list)


class Step(Protocol):
    meta: StepMeta

    def run(self, *, files: list[tuple[str, bytes]], params: Mapping[str, Any]) -> StepOutputs: ...

