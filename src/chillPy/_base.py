"""Shared helpers for placeholder implementations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def placeholder_array(values: Any = None, *, cumulative: bool = False) -> np.ndarray:
    """Return a zero-valued array matching the shape of ``values`` where possible."""
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    zeros = np.zeros_like(arr, dtype=float)
    return np.cumsum(zeros) if cumulative else zeros


def ensure_1d(values: Any) -> np.ndarray:
    """Convert an input sequence to a one-dimensional NumPy float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.ravel()


def placeholder_record(object_type: str, **payload: Any) -> dict[str, Any]:
    """Build a small dictionary used by many chillR object placeholders."""
    return {"object_type": object_type, **payload}


def not_implemented(name: str) -> None:
    """Raise a consistent placeholder error for unported behavior."""
    raise NotImplementedError(
        f"{name} is a placeholder in chillPy and has not been ported from chillR yet."
    )


def as_list(value: Any) -> list[Any]:
    """Return ``value`` as a list without treating strings as iterables."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]

