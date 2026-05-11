"""General utility functions mapped from chillR."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

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


def select_by_file_extension(strings: Any, file_extension: str) -> list[str] | float:
    """Select strings that end in a particular way (e.g. a certain file extension).

    Translates R ``select_by_file_extension``.

    Parameters
    ----------
    strings : Any
        Vector of character strings for elements to be extracted from.
    file_extension : str
        Character string specifying the extension or trailing string to match.

    Returns
    -------
    list or float
        Subset of the strings vector that end on file_extension. Returns ``np.nan``
        if no matches are found.
    """
    if strings is None:
        return np.nan

    vals = as_list(strings)
    if not vals:
        return np.nan

    out = [str(v) for v in vals if str(v).endswith(file_extension)]
    if not out:
        return np.nan
    return out


def identify_common_string(strings: Any, *, leading: bool = True) -> str | float:
    """Identify shared leading or trailing character strings.

    Translates R ``identify_common_string``. Returns ``np.nan`` if no common
    string exists, matching R's ``NA``.
    """
    values = [val for val in as_list(strings) if val is not None]
    if not values:
        return np.nan
    if len(values) == 1:
        return values[0] if not pd.isna(values[0]) else np.nan

    strs = [str(v) for v in values]
    if leading:
        prefix = strs[0]
        for s in strs[1:]:
            while not s.startswith(prefix) and prefix:
                prefix = prefix[:-1]
        return prefix if prefix else np.nan
    else:
        suffix = strs[0]
        for s in strs[1:]:
            while not s.endswith(suffix) and suffix:
                suffix = suffix[1:]
        return suffix if suffix else np.nan


def extract_differences_between_characters(strings: Any) -> np.ndarray | float:
    """Identify elements between shared leading and/or trailing substrings.

    Translates R ``extract_differences_between_characters``.
    """
    vals = as_list(strings)
    if not vals:
        return np.nan
    if len(vals) == 1:
        return np.array([str(vals[0])], dtype=object)

    leader = identify_common_string(vals, leading=True)
    trailer = identify_common_string(vals, leading=False)

    def remove_leader(s: str, lead: Any) -> str:
        if pd.isna(lead):
            return s
        return s[len(str(lead)) :]

    def remove_trailer(s: str, trail: Any) -> str:
        if pd.isna(trail):
            return s
        t_str = str(trail)
        return s[: -len(t_str)] if len(t_str) > 0 else s

    no_leader = [remove_leader(str(v), leader) for v in vals]
    if test_if_equal(no_leader):
        return np.nan

    result = [remove_trailer(s, trailer) for s in no_leader]
    return np.array(result, dtype=object)


def test_if_equal(test_vector: Any) -> bool:
    """Check if all elements in a vector are equal.

    Translates R ``test_if_equal``.
    """
    values = as_list(test_vector)
    if not values:
        return True
    first = values[0]
    for val in values[1:]:
        if val != first:
            return False
    return True


test_if_equal.__test__ = False


def read_tab(tab: str) -> pd.DataFrame:
    """Read CSV table regardless of whether it is comma or semicolon separated.

    Translates R ``read_tab``.
    """
    with open(tab, "r", encoding="utf-8", errors="replace") as f:
        content = f.read(10000)  # Read enough to detect separator

    freq_comma = content.count(",")
    freq_semicolon = content.count(";")

    if freq_comma > freq_semicolon:
        return pd.read_csv(tab, sep=",", decimal=".", header=0)
    else:
        return pd.read_csv(tab, sep=";", decimal=",", header=0)
