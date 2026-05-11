"""Temperature, chilling, and heat models translated from chillR."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._base import ensure_1d, placeholder_array


_UTAH_TABLE = {
    "lower": [-1000, 1.4, 2.4, 9.1, 12.4, 15.9, 18],
    "upper": [1.4, 2.4, 9.1, 12.4, 15.9, 18, 1000],
    "weight": [0, 0.5, 1, 0.5, 0, -0.5, -1],
}


def _parse_step_table(df: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if df is None:
        df = _UTAH_TABLE

    if isinstance(df, Mapping):
        lower = np.asarray(df["lower"], dtype=float)
        upper = np.asarray(df["upper"], dtype=float)
        weight = np.asarray(df["weight"], dtype=float)
    elif all(hasattr(df, attr) for attr in ("lower", "upper", "weight")):
        lower = np.asarray(df.lower, dtype=float)
        upper = np.asarray(df.upper, dtype=float)
        weight = np.asarray(df.weight, dtype=float)
    elif hasattr(df, "__getitem__"):
        try:
            lower = np.asarray(df["lower"], dtype=float)
            upper = np.asarray(df["upper"], dtype=float)
            weight = np.asarray(df["weight"], dtype=float)
        except Exception as exc:  # pragma: no cover - defensive for table-like objects
            raise TypeError("df must expose lower, upper, and weight columns") from exc
    else:
        try:
            rows = list(df)
            lower = np.asarray([row[0] for row in rows], dtype=float)
            upper = np.asarray([row[1] for row in rows], dtype=float)
            weight = np.asarray([row[2] for row in rows], dtype=float)
        except Exception as exc:  # pragma: no cover - defensive for odd iterables
            raise TypeError("df must be a mapping, table, or sequence of rows") from exc

    if not (lower.shape == upper.shape == weight.shape):
        raise ValueError("df lower, upper, and weight columns must have the same length")
    if lower.ndim != 1:
        raise ValueError("df columns must be one-dimensional")
    return lower, upper, weight


def _maybe_cumsum(values: np.ndarray, summ: bool) -> np.ndarray:
    return np.cumsum(values) if summ else values


def step_model(hour_temp: Any, df: Any = None, *, summ: bool = True) -> np.ndarray:
    """Calculate a cumulative stepwise temperature metric.

    Translates R ``step_model``. Intervals follow the R convention
    ``lower < temperature <= upper``. ``df`` may be a mapping/table with
    ``lower``, ``upper``, and ``weight`` columns or a sequence of
    ``(lower, upper, weight)`` rows.
    """
    temps = ensure_1d(hour_temp)
    lower, upper, weight = _parse_step_table(df)
    out = np.full(temps.shape, np.nan, dtype=float)

    for idx, temp in enumerate(temps):
        if np.isnan(temp):
            continue
        matches = np.flatnonzero((temp > lower) & (temp <= upper))
        if matches.size == 0:
            raise ValueError(f"temperature {temp!r} is outside all step_model intervals")
        out[idx] = weight[matches[0]]

    return _maybe_cumsum(out, summ)


def utah_model(hour_temp: Any, *, summ: bool = True) -> np.ndarray:
    """Calculate Utah Chill Units from hourly temperatures.

    Translates R ``Utah_Model`` using Richardson et al. (1974) weights.
    """
    return step_model(hour_temp, df=_UTAH_TABLE, summ=summ)


def chilling_hours(hour_temp: Any, *, summ: bool = True) -> np.ndarray:
    """Calculate Chilling Hours from hourly temperatures.

    Translates R ``Chilling_Hours``: each hour from 0 to 7.2 degrees Celsius,
    inclusive, receives weight 1; all other hours receive 0.
    """
    temps = ensure_1d(hour_temp)
    weights = np.zeros(temps.shape, dtype=float)
    weights[(temps >= 0) & (temps <= 7.2)] = 1.0
    return _maybe_cumsum(weights, summ)


def dynamic_model(
    hour_temp: Any,
    *,
    summ: bool = True,
    e0: float = 4153.5,
    e1: float = 12888.8,
    a0: float = 139500,
    a1: float = 2.567e18,
    slope: float = 1.6,
    tf: float = 277,
) -> np.ndarray:
    """Calculate Chill Portions with the Dynamic Model.

    Translates R ``Dynamic_Model``. Input temperatures are hourly values in
    degrees Celsius. The return value is either hourly chill portions
    (``summ=False``) or cumulative chill portions (``summ=True``).
    """
    temps = ensure_1d(hour_temp)
    if temps.size == 0:
        return np.array([], dtype=float)
    if np.isnan(temps).any():
        raise ValueError("dynamic_model does not accept missing temperature values")

    tk = temps + 273.0
    aa = a0 / a1
    ee = e1 - e0
    sr = np.exp(slope * tf * (tk - tf) / tk)
    xi = sr / (1.0 + sr)
    xs = aa * np.exp(ee / tk)
    eak1 = np.exp(-a1 * np.exp(-e1 / tk))

    x = np.zeros(temps.shape, dtype=float)
    for idx in range(1, temps.size):
        state = x[idx - 1]
        if state >= 1.0:
            state *= 1.0 - xi[idx - 2]
        x[idx] = xs[idx - 1] - (xs[idx - 1] - state) * eak1[idx - 1]

    delta = np.zeros(temps.shape, dtype=float)
    exceeded = np.flatnonzero(x >= 1.0)
    exceeded = exceeded[exceeded > 0]
    delta[exceeded] = x[exceeded] * xi[exceeded - 1]
    return _maybe_cumsum(delta, summ)


def dynamic_model_driver(
    temp: Any,
    times: Any | None = None,
    *,
    a0: float = 139500,
    a1: float = 2.567e18,
    e0: float = 4153.5,
    e1: float = 12888.8,
    slope: float = 1.6,
    tf: float = 4,
    deg_celsius: bool = True,
) -> dict[str, np.ndarray]:
    """Detailed Dynamic Model driver translated from R ``DynModel_driver``.

    Returns a dictionary with ``x`` (precursor state), ``y`` (cumulative chill),
    ``delta`` (per-step chill portions), and ``xs``. With hourly data this
    gives the same chill-portion totals as :func:`dynamic_model`.
    """
    temps = ensure_1d(temp)
    if temps.size == 0:
        empty = np.array([], dtype=float)
        return {"x": empty, "y": empty, "delta": empty, "xs": empty}
    if np.isnan(temps).any():
        raise ValueError("dynamic_model_driver does not accept missing temperature values")

    if times is None:
        time_values = np.arange(1, temps.size + 1, dtype=float)
    else:
        time_values = ensure_1d(times)
    if time_values.size != temps.size:
        raise ValueError("temp and times must have the same length")

    model_temp = temps + 273.0 if deg_celsius else temps.copy()
    transition_temp = tf + 273.0 if deg_celsius else tf
    xs = a0 / a1 * np.exp(-(e0 - e1) / model_temp)
    deltas_t = np.concatenate([np.diff(time_values), np.array([0.0])])
    ek1 = np.exp(-a1 * np.exp(-e1 / model_temp) * deltas_t)

    x = np.zeros(temps.shape, dtype=float)
    delta = np.zeros(temps.shape, dtype=float)
    for idx in range(temps.size - 1):
        x[idx + 1] = xs[idx] - (xs[idx] - x[idx]) * ek1[idx]
        if x[idx + 1] >= 1.0:
            xtmp = slope * transition_temp * (model_temp[idx] - transition_temp) / model_temp[idx]
            portion = x[idx + 1]
            if xtmp < 17:
                sr = np.exp(xtmp)
                portion *= sr / (1.0 + sr)
            delta[idx + 1] = portion
            x[idx + 1] -= portion

    return {"x": x, "y": np.cumsum(delta), "delta": delta, "xs": xs}


def gdh(hour_temp: Any, *, summ: bool = True) -> np.ndarray:
    """Calculate Growing Degree Hours from hourly temperatures.

    Translates R ``GDH`` using the Anderson et al. (1986) response curve.
    """
    temps = ensure_1d(hour_temp)
    tb = 4.0
    tu = 25.0
    tc = 36.0
    weights = np.zeros(temps.shape, dtype=float)

    lower = (temps >= tb) & (temps <= tu)
    weights[lower] = (tu - tb) / 2.0 * (
        1.0 + np.cos(np.pi + np.pi * (temps[lower] - tb) / (tu - tb))
    )

    upper = (temps > tu) & (temps <= tc)
    weights[upper] = (tu - tb) * (
        1.0 + np.cos(np.pi / 2.0 + np.pi / 2.0 * (temps[upper] - tu) / (tc - tu))
    )
    return _maybe_cumsum(weights, summ)


def gdh_model(hour_temp: Any, *, summ: bool = True) -> np.ndarray:
    """Alias implementation for R ``GDH_model``."""
    return gdh(hour_temp, summ=summ)


def gdd(hour_temp: Any, *, summ: bool = True, tbase: float = 5) -> np.ndarray:
    """Calculate Growing Degree Days from hourly temperatures.

    Translates R ``GDD``. Hourly temperatures are clipped to the R function's
    10 to 30 degree Celsius bounds before subtracting ``tbase`` and dividing
    by 24.
    """
    temps = ensure_1d(hour_temp)
    clipped = np.clip(temps, 10.0, 30.0)
    weights = (clipped - tbase) / 24.0
    weights[np.isnan(temps)] = 0.0
    return _maybe_cumsum(weights, summ)


def phenoflex(
    temp: Any,
    times: Any,
    *,
    a0: float = 6319.5,
    a1: float = 5.939917e13,
    e0: float = 3372.8,
    e1: float = 9900.3,
    slope: float = 1.6,
    tf: float = 4,
    s1: float = 0.5,
    tu: float = 25,
    tb: float = 4,
    tc: float = 36,
    yc: float = 40,
    delta: float = 4,
    imodel: int = 0,
    zc: float = 190,
    stop_at_zc: bool = True,
    deg_celsius: bool = True,
    basic_output: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Combined model of the dynamic model for chill accumulation and the GDH model.

    Translates R PhenoFlex (C++ implementation).

    Args:
        temp: Hourly temperatures.
        times: Time steps (e.g., hours).
        a0, a1, e0, e1: Dynamic model parameters.
        slope: Slope parameter for sigmoidal function.
        tf: Transition temperature for sigmoidal function.
        s1: Slope of transition from chill to heat accumulation.
        tu: GDH optimal temperature or GAUSS mean.
        tb: GDH base temperature.
        tc: GDH upper temperature.
        yc: Critical value defining end of chill accumulation.
        delta: Width of Gaussian heat accumulation model.
        imodel: Heat model: 0 for GDH, 1 for Gaussian.
        zc: Critical value for end of heat accumulation.
        stop_at_zc: Whether to stop once zc is reached.
        deg_celsius: Whether temperatures are in Celsius.
        basic_output: If True, returns only bloomindex.

    Returns:
        A dictionary with 'bloomindex' and optionally 'x', 'y', 'z', 'xs'.
    """
    temp = ensure_1d(temp)
    times = ensure_1d(times)
    n = len(temp)

    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    xs = np.zeros(n)

    _tf = tf
    _tu = tu
    _tc = tc
    _tb = tb

    if deg_celsius:
        _tf += 273.0
        _tu += 273.0
        _tc += 273.0
        _tb += 273.0

    bloom_index = 0

    pi = np.pi

    def p1z(t: float, tu_val: float, tb_val: float, tc_val: float) -> float:
        if tb_val <= t <= tu_val:
            return 0.5 * (1 + np.cos(pi + pi * (t - tb_val) / (tu_val - tb_val)))
        elif tu_val < t <= tc_val:
            return 1 + np.cos(pi / 2.0 + pi / 2.0 * (t - tu_val) / (tc_val - tu_val))
        return 0.0

    def p2z(t: float, tu_val: float, delta_val: float) -> float:
        return np.exp(-((t - tu_val) / 2.0 / delta_val) ** 2)

    def p_fcn(t_val: float, tf_val: float, slope_val: float) -> float:
        if t_val == 0:
            return 0.0
        val = slope_val * tf_val * (t_val - tf_val) / t_val
        if val >= 17:
            return 1.0
        elif val <= -20:
            return 0.0
        sr = np.exp(val)
        return sr / (1 + sr)

    for i in range(n - 1):
        ti = temp[i]
        if deg_celsius:
            ti += 273.0

        xs[i] = a0 / a1 * np.exp(-(e0 - e1) / ti)
        k1 = a1 * np.exp(-e1 / ti)
        x[i + 1] = xs[i] - (xs[i] - x[i]) * np.exp(-k1 * (times[i + 1] - times[i]))
        y[i + 1] = y[i]

        heat_inc = 0.0
        if imodel == 0:
            heat_inc = p1z(ti, _tu, _tb, _tc)
        else:
            heat_inc = p2z(ti, _tu, delta)

        z[i + 1] = z[i] + heat_inc * p_fcn(y[i], yc, s1) * (times[i + 1] - times[i])

        if x[i + 1] >= 1.0:
            delta_y = p_fcn(ti, _tf, slope) * x[i + 1]
            y[i + 1] += delta_y
            x[i + 1] -= delta_y

        if z[i + 1] >= zc:
            bloom_index = i + 2
            if stop_at_zc:
                break

    if basic_output:
        return {"bloomindex": bloom_index, "values": y + z}
    return {
        "x": x,
        "y": y,
        "z": z,
        "xs": xs,
        "bloomindex": bloom_index,
    }


Utah_Model = utah_model
Chilling_Hours = chilling_hours
Dynamic_Model = dynamic_model
DynModel_driver = dynamic_model_driver
GDH = gdh
GDH_model = gdh_model
GDD = gdd
PhenoFlex = phenoflex
