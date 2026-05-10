"""General utility functions mapped from chillR."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._base import as_list, ensure_1d


def runn_mean(
    vec: Any,
    runn_mean: int,
    *,
    na_rm: bool = False,
    exclude_central_value: bool = False,
    fun: Callable[[np.ndarray], float] | None = None,
) -> np.ndarray:
    """Calculate the running mean of a numeric vector.

    Translates R ``runn_mean``. The window shrinks at both ends of the vector,
    and the supplied ``runn_mean`` is capped at ``floor(len(vec) / 2)`` as in
    the R implementation.
    """
    values = ensure_1d(vec)
    if values.size == 0:
        return values.copy()
    if runn_mean < 0:
        raise ValueError("runn_mean must be non-negative")

    window = min(int(runn_mean), values.size // 2)
    half_floor = window // 2
    half_ceiling = int(np.ceil(window / 2))
    reducer = fun or (np.nanmean if na_rm else np.mean)

    out = np.empty(values.shape, dtype=float)
    n = values.size
    for idx0 in range(n):
        idx = idx0 + 1  # R code is written in 1-based indices.
        if idx < half_ceiling:
            lo = 0
            hi = idx + half_floor
        elif idx <= n - half_ceiling:
            lo = max(0, idx - half_floor - 1)
            hi = idx + half_floor
        else:
            lo = max(0, idx - half_floor - 1)
            hi = n
        window = values[lo:hi]
        if exclude_central_value:
            center = idx0 - lo
            if 0 <= center < window.size:
                window = np.delete(window, center)
        out[idx0] = reducer(window) if window.size else np.nan
    return out


def runn_mean_pred(
    indep: Any,
    dep: Any,
    pred: Any,
    *,
    runn_mean: int = 11,
    na_rm: bool = False,
    exclude_central_value: bool = False,
    fun: Callable[[np.ndarray], float] | None = None,
) -> dict[str, np.ndarray]:
    """Predict values by interpolating a running mean.

    Translates R ``runn_mean_pred``. Predictions outside the range of
    ``indep`` return ``nan``.
    """
    x = ensure_1d(indep)
    y = ensure_1d(dep)
    targets = ensure_1d(pred)
    if x.shape != y.shape:
        raise ValueError("indep and dep must have the same shape")

    effective_window = min(int(runn_mean), x.size // 2)
    runny = globals()["runn_mean"](
        y,
        effective_window,
        na_rm=na_rm,
        exclude_central_value=exclude_central_value,
        fun=fun,
    )

    valid_x = x[~np.isnan(x)]
    if valid_x.size == 0:
        return {"x": targets, "predicted": np.full(targets.shape, np.nan, dtype=float)}
    min_x = np.min(valid_x)
    max_x = np.max(valid_x)

    out = np.full(targets.shape, np.nan, dtype=float)
    for idx, target in enumerate(targets):
        if np.isnan(target) or not (min_x <= target <= max_x):
            continue

        lower_candidates = np.flatnonzero(x <= target)
        upper_candidates = np.flatnonzero(x >= target)
        if lower_candidates.size == 0 or upper_candidates.size == 0:
            continue

        start_idx = lower_candidates[-1]
        end_idx = upper_candidates[0]
        start_x = x[start_idx]
        end_x = x[end_idx]
        start_y = runny[start_idx]
        end_y = runny[end_idx]
        if end_x == start_x:
            out[idx] = start_y
        else:
            out[idx] = start_y + (end_y - start_y) * ((target - start_x) / (end_x - start_x))

    return {"x": targets, "predicted": out}


def select_by_file_extension(strings: Any, file_extension: str) -> list[Any]:
    """Partial placeholder for R ``select_by_file_extension``."""
    return [value for value in as_list(strings) if str(value).endswith(file_extension)]


def identify_common_string(strings: Any, *, leading: bool = True) -> str:
    """Partial placeholder for R ``identify_common_string``."""
    values = [str(value) for value in as_list(strings)]
    if not values:
        return ""
    if not leading:
        reversed_values = [value[::-1] for value in values]
        return identify_common_string(reversed_values, leading=True)[::-1]
    prefix = values[0]
    for value in values[1:]:
        while not value.startswith(prefix) and prefix:
            prefix = prefix[:-1]
    return prefix


def extract_differences_between_characters(strings: Any) -> list[str]:
    """Placeholder for R ``extract_differences_between_characters``."""
    return [str(value) for value in as_list(strings)]


def test_if_equal(test_vector: Any) -> bool:
    """Partial placeholder for R ``test_if_equal``."""
    values = as_list(test_vector)
    return len(set(values)) <= 1


test_if_equal.__test__ = False


def read_tab(tab: Any) -> Any:
    """Placeholder for R ``read_tab``; currently returns the input unchanged."""
    return tab
