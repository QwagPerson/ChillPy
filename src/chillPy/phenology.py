"""Phenology, PLS, and model-fitting helpers mapped from chillR."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .date_utils import get_last_date, make_jday
from .temperature import interpolate_gaps, temp_response, temp_response_hourtable
from .temperature_models import dynamic_model, gdh, gdh_model, phenoflex
from .utils import runn_mean as runn_mean_func

_MONTH_END_JDAYS = np.array([31, 59, 89, 120, 151, 181, 212, 243, 274, 304, 335, 365])


@dataclass
class PLSModel:
    """Lightweight single-response PLS model compatible with ``VIP``."""

    method: str
    n_components: int
    coefficients: np.ndarray
    intercept: float
    scores: np.ndarray
    y_loadings: np.ndarray
    loading_weights: np.ndarray
    x_loadings: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    x_columns: list[str]
    explained_variance: np.ndarray


def _as_dataframe(data: Any, *, name: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    try:
        return pd.DataFrame(data).copy()
    except Exception as exc:  # pragma: no cover - defensive for unusual table-like inputs
        raise TypeError(f"{name} must be convertible to a pandas DataFrame") from exc


def _temperature_frame(temps: Any) -> pd.DataFrame:
    if isinstance(temps, Mapping) and "hourtemps" in temps:
        return _as_dataframe(temps["hourtemps"], name="temps['hourtemps']")
    if isinstance(temps, Mapping) and "weather" in temps:
        return _as_dataframe(temps["weather"], name="temps['weather']")
    return _as_dataframe(temps, name="temps")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _normalize_years(years: Any) -> list[int]:
    if years is None:
        raise ValueError("years must be provided")
    if np.isscalar(years):
        out = [int(years)]
    else:
        out = [int(year) for year in list(years)]
    if not out:
        raise ValueError("years must contain at least one year")
    return out


def _validate_mrange(mrange: Sequence[int], *, require_cross_year: bool = False) -> tuple[int, int]:
    if len(mrange) != 2:
        raise ValueError("mrange must contain exactly two months")
    start, end = int(mrange[0]), int(mrange[1])
    if start < 1 or start > 12 or end < 1 or end > 12:
        raise ValueError("mrange entries must be months from 1 through 12")
    if require_cross_year and start <= end:
        raise ValueError("mrange[0] must be greater than mrange[1] for cross-year seasons")
    return start, end


def _season_indices(frame: pd.DataFrame, *, start_month: int, end_month: int, year: int) -> np.ndarray:
    months = pd.to_numeric(frame["Month"], errors="coerce")
    years = pd.to_numeric(frame["Year"], errors="coerce")
    mask = ((months >= start_month) & (months <= 12) & (years == year - 1)) | (
        (months >= 1) & (months <= end_month) & (years == year)
    )
    return np.flatnonzero(mask.to_numpy(dtype=bool))


def _ensure_weather_dates(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = frame.copy()
    if {"Year", "Month", "Day"}.issubset(out.columns):
        pass
    elif "Date" in out.columns:
        parsed = pd.to_datetime(out["Date"], errors="coerce")
        if (parsed.isna() & out["Date"].notna()).any():
            raise ValueError(f"{name} Date contains invalid date values")
        out["Year"] = parsed.dt.year
        out["Month"] = parsed.dt.month
        out["Day"] = parsed.dt.day
    else:
        _require_columns(out, ["Year", "Month", "Day"], name=name)
    out = make_jday(out)
    for column in ["Year", "Month", "Day", "JDay"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["Year", "Month", "Day", "JDay"]].isna().any(axis=None):
        raise ValueError(f"{name} contains missing or invalid date fields")
    out = out.sort_values(["Year", "Month", "Day"], kind="mergesort").reset_index(drop=True)
    out["Season"] = np.nan
    out["Date"] = out["Month"].astype(int) * 100 + out["Day"].astype(int)
    return out


def _season_label_dates(frame: pd.DataFrame, split_month: int) -> pd.DataFrame:
    if int(split_month) < 1 or int(split_month) > 12:
        raise ValueError("split_month must be a month from 1 through 12")
    out = frame.copy()
    months = out["Month"].astype(int)
    years = out["Year"].astype(int)
    out["Season"] = np.where(months <= int(split_month), years, years + 1)
    out["Date"] = months * 100 + out["Day"].astype(int)
    return out


def _as_numeric_vector(values: Any, *, name: str) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _validate_pls_inputs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("independent variables must be a two-dimensional matrix")
    if x.shape[0] != y.size:
        raise ValueError("independent and dependent variables must have the same number of rows")
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[mask]
    y = y[mask]
    if x.shape[0] < 3:
        raise ValueError("at least three complete seasons are required for PLS")
    if x.shape[1] < 1:
        raise ValueError("at least one predictor is required for PLS")
    if np.nanstd(y, ddof=1) == 0:
        raise ValueError("dependent variable must vary across seasons")
    return x, y


def _max_pls_components(x: np.ndarray) -> int:
    return int(min(x.shape[0] - 1, x.shape[1]))


def _fit_pls1(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_components: int,
    x_columns: Sequence[str] | None = None,
    x_scale: np.ndarray | None = None,
) -> PLSModel:
    """Fit a deterministic single-response orthogonal-score PLS model."""
    x, y = _validate_pls_inputs(x, y)
    max_components = _max_pls_components(x)
    if n_components < 1 or n_components > max_components:
        raise ValueError(f"n_components must be between 1 and {max_components}")

    x_mean = np.nanmean(x, axis=0)
    if x_scale is None:
        scale = np.nanstd(x, axis=0, ddof=1)
    else:
        scale = np.asarray(x_scale, dtype=float).reshape(-1)
        if scale.size != x.shape[1]:
            raise ValueError("x_scale length must match the number of predictors")
    scale = np.where(~np.isfinite(scale) | (scale == 0), 1.0, scale)
    y_mean = float(np.nanmean(y))

    x_res = (x - x_mean) / scale
    y_res = y - y_mean
    total_x_ss = float(np.sum(x_res**2))
    if total_x_ss == 0:
        raise ValueError("independent variables must vary across seasons")

    n_samples, n_predictors = x.shape
    scores = np.zeros((n_samples, n_components), dtype=float)
    weights = np.zeros((n_predictors, n_components), dtype=float)
    loadings = np.zeros((n_predictors, n_components), dtype=float)
    y_loadings = np.zeros(n_components, dtype=float)
    explained = np.zeros(n_components, dtype=float)

    for comp in range(n_components):
        weight = x_res.T @ y_res
        norm = float(np.linalg.norm(weight))
        if norm <= np.finfo(float).eps:
            raise ValueError("requested PLS component has no remaining covariance")
        weight = weight / norm
        score = x_res @ weight
        score_ss = float(score @ score)
        if score_ss <= np.finfo(float).eps:
            raise ValueError("requested PLS component has zero score variance")
        loading = x_res.T @ score / score_ss
        y_loading = float(y_res @ score / score_ss)
        x_hat = np.outer(score, loading)

        scores[:, comp] = score
        weights[:, comp] = weight
        loadings[:, comp] = loading
        y_loadings[comp] = y_loading
        explained[comp] = np.sum(x_hat**2) / total_x_ss * 100.0

        x_res = x_res - x_hat
        y_res = y_res - y_loading * score

    coef_scaled = weights @ np.linalg.pinv(loadings.T @ weights) @ y_loadings
    coefficients = coef_scaled / scale
    intercept = y_mean - float(x_mean @ coefficients)
    columns = list(x_columns) if x_columns is not None else [f"x{idx + 1}" for idx in range(n_predictors)]
    return PLSModel(
        method="oscorespls",
        n_components=int(n_components),
        coefficients=coefficients.astype(float),
        intercept=intercept,
        scores=scores,
        y_loadings=y_loadings.reshape(1, -1),
        loading_weights=weights,
        x_loadings=loadings,
        x_mean=x_mean,
        x_scale=scale,
        y_mean=y_mean,
        x_columns=columns,
        explained_variance=explained,
    )


def _select_n_components(x: np.ndarray, y: np.ndarray, *, threshold: float, ncomp_fix: int | None) -> int:
    x, y = _validate_pls_inputs(x, y)
    max_components = _max_pls_components(x)
    if ncomp_fix is not None:
        ncomp = int(ncomp_fix)
        if ncomp < 1 or ncomp > max_components:
            raise ValueError(f"ncomp_fix must be between 1 and {max_components}")
        return ncomp
    if y.size <= 15:
        return min(2, max_components)

    trial_components = min(10, max_components)
    trial = _fit_pls1(x, y, n_components=trial_components)
    cumulative = np.cumsum(trial.explained_variance)
    matches = np.flatnonzero(cumulative > float(threshold))
    return int(matches[0] + 1) if matches.size else trial_components


def _fit_pls_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_columns: Sequence[str],
    expl_var: float,
    ncomp_fix: int | None,
) -> PLSModel:
    n_components = _select_n_components(x, y, threshold=float(expl_var), ncomp_fix=ncomp_fix)
    scale = np.nanstd(np.asarray(x, dtype=float), axis=0, ddof=1)
    scale = np.where(~np.isfinite(scale) | (scale == 0), 1.0, scale)
    return _fit_pls1(x, y, n_components=n_components, x_columns=x_columns, x_scale=scale)


def _coerce_pheno(bio_data: Any, *, name: str = "bio_data") -> pd.DataFrame:
    bio = _as_dataframe(bio_data, name=name)
    _require_columns(bio, ["Year", "pheno"], name=name)
    bio = bio.copy()
    bio["Year"] = pd.to_numeric(bio["Year"], errors="coerce")
    bio["pheno"] = pd.to_numeric(bio["pheno"], errors="coerce")
    bio = bio.loc[bio["Year"].notna() & bio["pheno"].notna()].copy()
    if bio.empty:
        raise ValueError(f"{name} must contain at least one complete Year/pheno observation")
    bio["Year"] = bio["Year"].astype(int)
    return bio.sort_values("Year", kind="mergesort").reset_index(drop=True)


def _season_matrix(
    weather_file: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    prefix: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    seasons = [int(season) for season in pd.unique(weather_file["Season"])]
    rows: list[np.ndarray] = []
    keep_seasons: list[int] = []
    labdates: np.ndarray | None = None
    labjdates: np.ndarray | None = None
    for season in seasons:
        yearweather = weather_file.loc[weather_file["Season"] == season].sort_values(
            ["Year", "Month", "Day"], kind="mergesort"
        )
        if len(yearweather) < 365:
            continue
        parts: list[np.ndarray] = []
        for column in value_columns:
            parts.append(yearweather[column].to_numpy(dtype=float)[:365])
        vector = np.concatenate(parts)
        if np.isfinite(vector).all():
            rows.append(vector)
            keep_seasons.append(season)
        if labdates is None:
            first_year = yearweather.iloc[:365]
            labdates = first_year["Date"].to_numpy(dtype=int)
            labjdates = first_year["JDay"].to_numpy(dtype=int)
    if not rows or labdates is None or labjdates is None:
        raise ValueError("no complete 365-day seasons are available for PLS")
    columns: list[str] = []
    for label in prefix:
        columns.extend([f"{label}_{day}" for day in range(1, 366)])
    return np.vstack(rows), np.asarray(keep_seasons, dtype=int), labdates, labjdates, columns


def _filter_to_pheno(
    x: np.ndarray,
    seasons: np.ndarray,
    bio_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    bio = bio_data.loc[bio_data["Year"].isin(seasons)].copy()
    if bio.empty:
        raise ValueError("no phenology observations match complete weather seasons")
    season_to_pos = {int(season): idx for idx, season in enumerate(seasons)}
    row_positions = [season_to_pos[int(year)] for year in bio["Year"]]
    return x[row_positions, :], bio["pheno"].to_numpy(dtype=float), bio.reset_index(drop=True)


def _clip_at_pheno_end(
    x: np.ndarray,
    labdates: np.ndarray,
    labjdates: np.ndarray,
    bio_data: pd.DataFrame,
    end_at_pheno_end: bool | int | float,
    *,
    paired: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(end_at_pheno_end, (int, float)) and not isinstance(end_at_pheno_end, bool):
        pheno_end = int(end_at_pheno_end)
    elif end_at_pheno_end:
        inferred = get_last_date(bio_data["pheno"])
        pheno_end = int(inferred) if inferred is not None else None
    else:
        pheno_end = None
    if pheno_end is None or pheno_end not in set(labjdates.tolist()):
        return x, labdates, labjdates

    last = int(np.flatnonzero(labjdates == pheno_end)[0]) + 1
    keep = np.arange(last)
    if paired:
        x = x[:, np.concatenate([keep, keep + 365])]
    else:
        x = x[:, keep]
    return x, labdates[keep], labjdates[keep]


def _adjust_split_jdays(labjdates: np.ndarray, split_month: int) -> np.ndarray:
    cutoff = int(_MONTH_END_JDAYS[int(split_month) - 1])
    adjusted = labjdates.astype(int).copy()
    adjusted[adjusted > cutoff] = adjusted[adjusted > cutoff] - 365
    return adjusted


def _chuine_cf(x: Any, a: float, b: float, c: float) -> np.ndarray:
    """Chuine unified-model chill/force response function."""
    temps = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    return 1.0 / (1.0 + np.exp(float(a) * (temps - float(c)) ** 2 + float(b) * (temps - float(c))))


def _chuine_fstar(ctot: float, w: float, k: float) -> float:
    """Critical forcing value from Chuine's unified model."""
    return float(w) * np.exp(float(k) * float(ctot))


def _prepare_wrapper_inputs(x: Any, par: Any, *, par_length: int, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = _as_dataframe(x, name="x")
    _require_columns(frame, ["Temp", "JDay"], name="x")
    temps = pd.to_numeric(frame["Temp"], errors="coerce").to_numpy(dtype=float)
    jdays = pd.to_numeric(frame["JDay"], errors="coerce").to_numpy(dtype=float)
    params = pd.to_numeric(pd.Series(par), errors="coerce").to_numpy(dtype=float)
    if params.size != par_length:
        raise ValueError(f"{name} requires a parameter vector of length {par_length}")
    if temps.size == 0:
        raise ValueError("x must contain at least one temperature")
    if np.isnan(temps).any() or np.isnan(jdays).any():
        raise ValueError("x Temp and JDay must be numeric and complete")
    if np.isnan(params).any():
        raise ValueError(f"{name} parameters must be numeric and complete")
    return temps, jdays, params


def _jday_from_r_relative_crossing(jdays: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Return JDay using the relative-index lookup used in R wrappers."""
    crossing = np.flatnonzero(values >= float(threshold))
    if crossing.size == 0:
        return np.nan
    idx = int(crossing[0])
    return float(jdays[idx]) if idx < jdays.size else np.nan


def _resolve_phenology_model(modelfn: Any) -> Callable[..., Any]:
    """Resolve R-style model wrapper names to Python callables."""
    if modelfn is None:
        return uni_force_wrapper
    if callable(modelfn):
        return modelfn
    if isinstance(modelfn, str):
        key = modelfn.strip().lower().replace(".", "_").replace("-", "_")
        mapping: dict[str, Callable[..., Any]] = {
            "unifiedmodel_wrapper": unified_model_wrapper,
            "unified_model_wrapper": unified_model_wrapper,
            "unified": unified_model_wrapper,
            "unichill_wrapper": uni_chill_wrapper,
            "uni_chill_wrapper": uni_chill_wrapper,
            "uni_chill": uni_chill_wrapper,
            "uniforce_wrapper": uni_force_wrapper,
            "uni_force_wrapper": uni_force_wrapper,
            "uni_force": uni_force_wrapper,
            "stepchill_wrapper": step_chill_wrapper,
            "step_chill_wrapper": step_chill_wrapper,
            "step_chill": step_chill_wrapper,
        }
        if key in mapping:
            return mapping[key]
    raise ValueError("modelfn must be a callable or a supported model wrapper name")


def _coerce_parameter_vector(values: Any, *, name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if np.isnan(arr).any():
        raise ValueError(f"{name} must be numeric and complete")
    return arr


def _coerce_control(control: Mapping[str, Any] | None, *, seed: int) -> dict[str, Any]:
    out = {
        "smooth": False,
        "verbose": False,
        "maxit": 1000,
        "nb.stop.improvement": 250,
        "tol": 1e-6,
    }
    if control is not None:
        out.update(dict(control))
    out["seed"] = int(seed)
    out["maxit"] = int(out.get("maxit", 1000))
    if out["maxit"] < 0:
        raise ValueError("control['maxit'] must be non-negative")
    out["tol"] = float(out.get("tol", 1e-6))
    if out["tol"] <= 0:
        raise ValueError("control['tol'] must be positive")
    return out


def _coerce_season_list(season_list: Any) -> list[pd.DataFrame]:
    if season_list is None:
        raise ValueError("SeasonList must be provided")
    if isinstance(season_list, Mapping) and "hourtemps" in season_list:
        raise ValueError("SeasonList must be a list of season DataFrames, not a full hourly table")
    if not isinstance(season_list, Sequence) or isinstance(season_list, (str, bytes, pd.DataFrame)):
        raise ValueError("SeasonList must be a list of season DataFrames")
    seasons = [_as_dataframe(season, name=f"SeasonList[{idx}]") for idx, season in enumerate(season_list)]
    if not seasons:
        raise ValueError("SeasonList must contain at least one season")
    for idx, season in enumerate(seasons):
        _require_columns(season, ["Temp", "JDay"], name=f"SeasonList[{idx}]")
        temps = pd.to_numeric(season["Temp"], errors="coerce")
        jdays = pd.to_numeric(season["JDay"], errors="coerce")
        if temps.isna().any() or jdays.isna().any():
            raise ValueError(f"SeasonList[{idx}] Temp and JDay must be numeric and complete")
    return [season.reset_index(drop=True) for season in seasons]


def _unwrap_cross_year_seasons(
    season_list: Sequence[pd.DataFrame],
    bloom_jdays: np.ndarray,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], np.ndarray]:
    """Apply the JDay unwrapping used by R ``phenologyFitter``."""
    fit_seasons: list[pd.DataFrame] = []
    stored_seasons: list[pd.DataFrame] = []
    unwrapped_bloom = bloom_jdays.astype(float).copy()
    for idx, season in enumerate(season_list):
        fit_season = season.copy()
        stored = season.copy()
        jdays = pd.to_numeric(fit_season["JDay"], errors="coerce").to_numpy(dtype=float)
        min_jday = float(jdays[0])
        max_jday = float(jdays[-1])
        if jdays.size > 1 and max_jday > min_jday:
            raise ValueError(f"Season {idx + 1} is overlapping with the previous or following one")
        bloom = unwrapped_bloom[idx]
        if np.isfinite(bloom) and bloom > max_jday and bloom < min_jday:
            raise ValueError(f"In season {idx + 1} the bloomJDay is outside the provided JDay vector")
        if jdays.size > 1:
            diffs = np.diff(jdays)
            min_diff = float(np.min(diffs))
            jump_idx = int(np.flatnonzero(diffs == min_diff)[0]) + 1
            jdays[:jump_idx] = jdays[:jump_idx] + min_diff - 1.0
            if np.isfinite(bloom) and bloom > min_jday:
                unwrapped_bloom[idx] = bloom + min_diff - 1.0
        fit_season["JDay"] = jdays
        stored["JDayunwrapped"] = jdays
        fit_seasons.append(fit_season)
        stored_seasons.append(stored)
    return fit_seasons, stored_seasons, unwrapped_bloom


def predict_bloom_days(
    par: Any,
    season_list: Any,
    modelfn: Any = None,
    **kwargs: Any,
) -> np.ndarray:
    """Predict bloom Julian days for each season using a model wrapper."""
    params = _coerce_parameter_vector(par, name="par")
    if params is None:  # pragma: no cover - guarded by helper contract
        raise ValueError("par must be provided")
    seasons = _coerce_season_list(season_list)
    model = _resolve_phenology_model(modelfn)
    predictions: list[float] = []
    for season in seasons:
        value = model(season, params, **kwargs)
        try:
            predictions.append(float(value))
        except (TypeError, ValueError):
            predictions.append(np.nan)
    return np.asarray(predictions, dtype=float)


def _phenology_rss(
    par: np.ndarray,
    *,
    modelfn: Callable[..., Any],
    bloom_jdays: np.ndarray,
    season_list: Sequence[pd.DataFrame],
    na_penalty: float = 365.0,
    kwargs: Mapping[str, Any] | None = None,
) -> float:
    predictions = predict_bloom_days(par, season_list, modelfn=modelfn, **dict(kwargs or {}))
    residuals = predictions - bloom_jdays
    residuals[np.isnan(predictions)] = float(na_penalty)
    complete = np.isfinite(residuals)
    if not complete.any():
        raise ValueError("at least one complete observed bloom JDay is required")
    return float(np.sum(residuals[complete] ** 2))


def _bounded_coordinate_search(
    objective: Callable[[np.ndarray], float],
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Small deterministic bounded coordinate search used in place of GenSA."""
    x = np.clip(np.asarray(start, dtype=float), lower, upper)
    fixed = np.isclose(lower, upper)
    free = ~fixed
    if np.isinf(lower[free]).any() or np.isinf(upper[free]).any():
        raise ValueError("finite lower and upper bounds are required for free parameters")
    step = np.zeros_like(x, dtype=float)
    step[free] = (upper[free] - lower[free]) / 4.0
    if "step" in control:
        custom_step = _coerce_parameter_vector(control["step"], name="control['step']")
        if custom_step is not None:
            if custom_step.size != x.size:
                raise ValueError("control['step'] must match the parameter vector length")
            step[free] = np.abs(custom_step[free])
    step[free & (step == 0)] = np.maximum(np.abs(x[free & (step == 0)]) * 0.1, 1.0)

    best_value = float(objective(x))
    evaluations = 1
    maxit = int(control.get("maxit", 1000))
    tol = float(control.get("tol", 1e-6))
    convergence = 1
    message = "maximum iterations reached"

    while evaluations < maxit:
        improved = False
        for dim in np.flatnonzero(free):
            if evaluations >= maxit:
                break
            candidates = []
            for direction in (-1.0, 1.0):
                candidate = x.copy()
                candidate[dim] = np.clip(candidate[dim] + direction * step[dim], lower[dim], upper[dim])
                candidates.append(candidate)
            for candidate in candidates:
                if evaluations >= maxit:
                    break
                value = float(objective(candidate))
                evaluations += 1
                if value < best_value:
                    x = candidate
                    best_value = value
                    improved = True
        if not improved:
            step[free] *= 0.5
            converged = bool(np.nanmax(step[free]) <= tol) if free.any() else True
            if converged:
                convergence = 0
                message = "converged"
                break
    if maxit == 0:
        convergence = 0
        message = "evaluated starting parameters only"
    return {
        "par": x.astype(float),
        "value": best_value,
        "counts": int(evaluations),
        "convergence": int(convergence),
        "message": message,
        "optimizer": "bounded_coordinate_search",
    }


def _pop_alias(kwargs: dict[str, Any], name: str, current: Any) -> Any:
    if name in kwargs:
        if current is not None:
            raise TypeError(f"received both Python and R-style values for {name}")
        return kwargs.pop(name)
    return current


def _numeric_requirements(values: Any, *, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} must be provided")
    arr = pd.to_numeric(pd.Series(values if not np.isscalar(values) else [values]), errors="coerce").to_numpy(
        dtype=float
    )
    if arr.size == 0 or np.isnan(arr).any():
        raise ValueError(f"{name} must contain numeric values")
    return arr


def _validate_start_jday(start_jday: Any) -> int:
    if start_jday is None:
        raise ValueError("start_jday must be provided")
    try:
        value = int(round(float(start_jday)))
    except (TypeError, ValueError) as exc:
        raise ValueError("start_jday must be numeric") from exc
    if value < 1:
        raise ValueError("start_jday can't be less than 1")
    if value > 366:
        raise ValueError("start_jday can't be greater than 366")
    return value


def _prepare_chill_table(
    hour_chill_table: Any,
    *,
    chill_model: str,
    heat_model: str,
    start_jday: int,
) -> pd.DataFrame:
    frame = _as_dataframe(hour_chill_table, name="hour_chill_table")
    _require_columns(frame, ["Year", "Month", "Day", chill_model, heat_model], name="hour_chill_table")
    if "JDay" not in frame.columns:
        frame = make_jday(frame)
    for column in ["Year", "Month", "Day", "JDay", chill_model, heat_model]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["Year", "Month", "Day", "JDay", chill_model, heat_model]].isna().any(axis=None):
        raise ValueError("hour_chill_table contains missing or non-numeric required values")
    sort_columns = [column for column in ["Year", "JDay", "Hour"] if column in frame.columns]
    frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    if "Season" not in frame.columns:
        frame["Season"] = np.where(frame["JDay"] >= start_jday, frame["Year"], frame["Year"] - 1)
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    if frame["Season"].isna().any():
        raise ValueError("Season values must be numeric")
    frame["Season"] = frame["Season"].astype(int)
    return frame


def _reset_metric_by_season(frame: pd.DataFrame, column: str, *, start_jday: int) -> np.ndarray:
    values = frame[column].to_numpy(dtype=float).copy()
    for season in pd.unique(frame["Season"]):
        mask = frame["Season"].to_numpy() == season
        start_mask = mask & (frame["JDay"].to_numpy(dtype=float) == float(start_jday))
        if start_mask.any():
            offset = values[np.flatnonzero(start_mask)[0]]
        else:
            offset = values[np.flatnonzero(mask)[0]]
        values[mask] = values[mask] - offset
    return values


def _next_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    return np.concatenate([values[1:], values[:1]])


def _first_crossing_index(
    values: np.ndarray,
    threshold: float,
    *,
    seasons: np.ndarray | None = None,
    season: int | None = None,
    inclusive_next: bool = True,
) -> int | None:
    next_values = _next_values(values)
    if inclusive_next:
        crossing = (values < threshold) & (next_values >= threshold)
    else:
        crossing = (values <= threshold) & (next_values > threshold)
    if seasons is not None and season is not None:
        crossing &= seasons == season
    matches = np.flatnonzero(crossing)
    return int(matches[0]) if matches.size else None


def _yearmoda(row: pd.Series) -> int:
    return int(row["Year"]) * 10000 + int(row["Month"]) * 100 + int(row["Day"])


def _build_requirement_table(
    seasons: Sequence[int],
    chill_req: np.ndarray,
    heat_req: np.ndarray,
    *,
    permutations: bool,
    infocol: Any,
) -> pd.DataFrame:
    season_values = list(dict.fromkeys(int(season) for season in seasons))
    if permutations:
        rows = [
            {"Season": season, "Creq": float(creq), "Hreq": float(hreq)}
            for season in season_values
            for creq in chill_req
            for hreq in heat_req
        ]
        return pd.DataFrame(rows)
    if chill_req.size != heat_req.size:
        raise ValueError("chill_req and heat_req are of different length")
    base = pd.DataFrame({"Creq": chill_req.astype(float), "Hreq": heat_req.astype(float)})
    if infocol is not None:
        info = list(infocol)
        if len(info) == chill_req.size:
            base.insert(0, "infocol", info)
    rows = []
    for season in season_values:
        season_frame = base.copy()
        season_frame.insert(0 if "infocol" not in season_frame.columns else 1, "Season", season)
        rows.append(season_frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Season", "Creq", "Hreq"])


def _prediction2_from_table(
    frame: pd.DataFrame,
    chill_req: np.ndarray,
    heat_req: np.ndarray,
    *,
    permutations: bool,
    chill_model: str,
    heat_model: str,
    start_jday: int,
    infocol: Any = None,
    include_yearmoda: bool = False,
) -> pd.DataFrame:
    chill = _reset_metric_by_season(frame, chill_model, start_jday=start_jday)
    heat = _reset_metric_by_season(frame, heat_model, start_jday=start_jday)
    seasons = frame["Season"].to_numpy(dtype=int)
    reqs = _build_requirement_table(pd.unique(frame["Season"]), chill_req, heat_req, permutations=permutations, infocol=infocol)
    if reqs.empty:
        return reqs

    chill_rows: dict[tuple[int, float], dict[str, Any]] = {}
    for season in pd.unique(reqs["Season"]):
        for creq in pd.unique(reqs.loc[reqs["Season"] == season, "Creq"]):
            idx = _first_crossing_index(chill, float(creq), seasons=seasons, season=int(season), inclusive_next=True)
            if idx is None:
                chill_rows[(int(season), float(creq))] = {
                    "Chill_comp": np.nan,
                    "Heat_on_CR": np.nan,
                    "Chill_comp_YEARMODA": np.nan,
                }
            else:
                row = frame.iloc[idx]
                chill_rows[(int(season), float(creq))] = {
                    "Chill_comp": float(row["JDay"]),
                    "Heat_on_CR": float(heat[idx]),
                    "Chill_comp_YEARMODA": _yearmoda(row),
                }

    rows: list[dict[str, Any]] = []
    for _, req in reqs.iterrows():
        season = int(req["Season"])
        creq = float(req["Creq"])
        hreq = float(req["Hreq"])
        chill_info = chill_rows[(season, creq)]
        heat_target = hreq + chill_info["Heat_on_CR"] if np.isfinite(chill_info["Heat_on_CR"]) else np.nan
        heat_idx = None
        if np.isfinite(heat_target):
            heat_idx = _first_crossing_index(heat, float(heat_target), seasons=seasons, season=season, inclusive_next=True)
        pheno_date = np.nan
        pheno_yearmoda = np.nan
        if heat_idx is not None:
            heat_row = frame.iloc[heat_idx]
            if int(heat_row["Season"]) == season:
                pheno_date = float(heat_row["JDay"])
                pheno_yearmoda = _yearmoda(heat_row)
        out = req.to_dict()
        out.update(
            {
                "Chill_comp": chill_info["Chill_comp"],
                "Pheno_date": pheno_date,
            }
        )
        if include_yearmoda:
            out.update(
                {
                    "Chill_comp_YEARMODA": chill_info["Chill_comp_YEARMODA"],
                    "Pheno_YEARMODA": pheno_yearmoda,
                }
            )
        rows.append(out)

    result = pd.DataFrame(rows)
    if "Season" in result.columns and not result.empty:
        result["Season"] = result["Season"].astype(int)
    order = ["Season", "Creq", "Hreq", "Chill_comp", "Pheno_date"]
    if "infocol" in result.columns:
        order = ["infocol", *order]
    if include_yearmoda:
        order = [*order, "Chill_comp_YEARMODA", "Pheno_YEARMODA"]
    return result[order].reset_index(drop=True)


def bloom_prediction(
    hour_chill_table: Any = None,
    chill_req: Any = None,
    heat_req: Any = None,
    chill_model: str = "Chill_Portions",
    heat_model: str = "GDH",
    start_jday: int | float = 305,
    **kwargs: Any,
) -> pd.DataFrame:
    """Predict bloom dates from scalar chill and heat requirements.

    Translates R ``bloom_prediction``. The input is an hourly chill/heat table
    such as the output of ``chilling_hourtable``.
    """
    hour_chill_table = _pop_alias(kwargs, "HourChillTable", hour_chill_table)
    chill_req = _pop_alias(kwargs, "Chill_req", chill_req)
    heat_req = _pop_alias(kwargs, "Heat_req", heat_req)
    chill_model = _pop_alias(kwargs, "Chill_model", chill_model)
    heat_model = _pop_alias(kwargs, "Heat_model", heat_model)
    start_jday = _pop_alias(kwargs, "Start_JDay", start_jday)
    if kwargs:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(kwargs)}")

    creq = float(_numeric_requirements(chill_req, name="chill_req")[0])
    hreq = float(_numeric_requirements(heat_req, name="heat_req")[0])
    start = _validate_start_jday(start_jday)
    frame = _prepare_chill_table(
        hour_chill_table,
        chill_model=str(chill_model),
        heat_model=str(heat_model),
        start_jday=start,
    )
    chill = _reset_metric_by_season(frame, str(chill_model), start_jday=start)
    seasons = frame["Season"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []

    next_chill = _next_values(chill)
    crossing_indices = np.flatnonzero((chill <= creq) & (next_chill > creq)) + 1
    crossing_indices = crossing_indices[crossing_indices < len(frame)]
    crossing_indices = crossing_indices[chill[crossing_indices] != 0]

    for pos, creq_idx in enumerate(crossing_indices):
        next_creq = int(crossing_indices[pos + 1]) if pos + 1 < len(crossing_indices) else len(frame) - 1
        creq_row = frame.iloc[int(creq_idx)]
        temp = frame.iloc[int(creq_idx) : next_creq + 1].copy().reset_index(drop=True)
        heat = temp[str(heat_model)].to_numpy(dtype=float) - float(temp[str(heat_model)].iloc[0])
        heat_crossing = _first_crossing_index(heat, hreq, inclusive_next=False)
        heat_idx = heat_crossing + 1 if heat_crossing is not None else None
        row: dict[str, Any] = {
            "Season": int(seasons[int(creq_idx)]),
            "Creqfull": int(creq_idx + 1),
            "Creq_year": int(creq_row["Year"]),
            "Creq_month": int(creq_row["Month"]),
            "Creq_day": int(creq_row["Day"]),
            "Creq_JDay": int(creq_row["JDay"]),
            "Hreqfull": np.nan,
            "Hreq_year": np.nan,
            "Hreq_month": np.nan,
            "Hreq_day": np.nan,
            "Hreq_JDay": np.nan,
        }
        if heat_idx is not None and heat_idx < len(temp):
            heat_row = temp.iloc[heat_idx]
            row.update(
                {
                    "Hreqfull": int(heat_idx + 1),
                    "Hreq_year": int(heat_row["Year"]),
                    "Hreq_month": int(heat_row["Month"]),
                    "Hreq_day": int(heat_row["Day"]),
                    "Hreq_JDay": int(heat_row["JDay"]),
                }
            )
        rows.append(row)

    result = pd.DataFrame(
        rows,
        columns=[
            "Season",
            "Creqfull",
            "Creq_year",
            "Creq_month",
            "Creq_day",
            "Creq_JDay",
            "Hreqfull",
            "Hreq_year",
            "Hreq_month",
            "Hreq_day",
            "Hreq_JDay",
        ],
    )
    if result.empty:
        return result
    keep = result["Hreq_year"].notna() & ((result["Hreq_year"] - result["Creq_year"]) < 2)
    return result.loc[keep].reset_index(drop=True)


def bloom_prediction2(
    hour_chill_table: Any = None,
    chill_req: Any = None,
    heat_req: Any = None,
    permutations: bool = False,
    chill_model: str = "Chill_Portions",
    heat_model: str = "GDH",
    start_jday: int | float = 305,
    infocol: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Vectorized bloom prediction from precomputed chill and heat metrics.

    Translates R ``bloom_prediction2``. Requirement vectors are paired by
    position unless ``permutations=True``.
    """
    hour_chill_table = _pop_alias(kwargs, "HourChillTable", hour_chill_table)
    chill_req = _pop_alias(kwargs, "Chill_req", chill_req)
    heat_req = _pop_alias(kwargs, "Heat_req", heat_req)
    chill_model = _pop_alias(kwargs, "Chill_model", chill_model)
    heat_model = _pop_alias(kwargs, "Heat_model", heat_model)
    start_jday = _pop_alias(kwargs, "Start_JDay", start_jday)
    if kwargs:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(kwargs)}")

    creq = _numeric_requirements(chill_req, name="chill_req")
    hreq = _numeric_requirements(heat_req, name="heat_req")
    start = _validate_start_jday(start_jday)
    frame = _prepare_chill_table(
        hour_chill_table,
        chill_model=str(chill_model),
        heat_model=str(heat_model),
        start_jday=start,
    )
    return _prediction2_from_table(
        frame,
        creq,
        hreq,
        permutations=bool(permutations),
        chill_model=str(chill_model),
        heat_model=str(heat_model),
        start_jday=start,
        infocol=infocol,
        include_yearmoda=False,
    )


def bloom_prediction3(
    hourtemps: Any = None,
    chill_req: Any = None,
    heat_req: Any = None,
    models: Mapping[str, Callable[..., Any]] | None = None,
    permutations: bool = False,
    chill_model: str = "Chill_Portions",
    heat_model: str = "GDH",
    start_jday: int | float = 305,
    infocol: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Predict bloom dates from hourly temperatures and model requirements.

    Translates R ``bloom_prediction3`` by first calculating a
    ``tempResponse_hourtable`` and then applying the vectorized prediction
    logic from ``bloom_prediction2``.
    """
    hourtemps = _pop_alias(kwargs, "hourtemps", hourtemps)
    chill_req = _pop_alias(kwargs, "Chill_req", chill_req)
    heat_req = _pop_alias(kwargs, "Heat_req", heat_req)
    chill_model = _pop_alias(kwargs, "Chill_model", chill_model)
    heat_model = _pop_alias(kwargs, "Heat_model", heat_model)
    start_jday = _pop_alias(kwargs, "Start_JDay", start_jday)
    if kwargs:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(kwargs)}")

    creq = _numeric_requirements(chill_req, name="chill_req")
    hreq = _numeric_requirements(heat_req, name="heat_req")
    start = _validate_start_jday(start_jday)
    if hourtemps is None:
        raise ValueError("hourtemps must be provided")
    hour_frame = _temperature_frame(hourtemps)
    selected_models = (
        {"Chill_Portions": dynamic_model, "GDH": gdh_model}
        if models is None
        else dict(models)
    )
    hour_chill_table = temp_response_hourtable(hour_frame, start_jday=start, models=selected_models)
    if str(chill_model) not in hour_chill_table.columns:
        raise ValueError(f"{chill_model} metric not calculated")
    if str(heat_model) not in hour_chill_table.columns:
        raise ValueError(f"{heat_model} metric not calculated")
    frame = _prepare_chill_table(
        hour_chill_table,
        chill_model=str(chill_model),
        heat_model=str(heat_model),
        start_jday=start,
    )
    return _prediction2_from_table(
        frame,
        creq,
        hreq,
        permutations=bool(permutations),
        chill_model=str(chill_model),
        heat_model=str(heat_model),
        start_jday=start,
        infocol=infocol,
        include_yearmoda=True,
    )


def _model_attr(object_: Any, *names: str) -> Any:
    for name in names:
        if hasattr(object_, name):
            return getattr(object_, name)
        if isinstance(object_, Mapping) and name in object_:
            return object_[name]
    raise ValueError(f"PLS model is missing required field '{names[0]}'")


def vip(object_: Any) -> pd.DataFrame:
    """Calculate variable-importance-in-projection scores for a PLS model.

    Translates R ``VIP`` for single-response orthogonal-score PLS models. The
    Python return value is a DataFrame with one row per component and one
    column per predictor.
    """
    method = _model_attr(object_, "method")
    if method != "oscorespls":
        raise ValueError('Only implemented for orthogonal scores algorithm. Refit with method="oscorespls"')

    y_loadings = np.asarray(_model_attr(object_, "y_loadings", "Yloadings"), dtype=float)
    if y_loadings.ndim == 1:
        y_loadings = y_loadings.reshape(1, -1)
    if y_loadings.shape[0] > 1:
        raise ValueError("Only implemented for single-response models")

    scores = np.asarray(_model_attr(object_, "scores"), dtype=float)
    weights = np.asarray(_model_attr(object_, "loading_weights", "loading.weights"), dtype=float)
    if scores.ndim != 2 or weights.ndim != 2:
        raise ValueError("PLS scores and loading weights must be two-dimensional")
    if scores.shape[1] != weights.shape[1] or y_loadings.shape[1] != scores.shape[1]:
        raise ValueError("PLS scores, loadings, and weights have incompatible component counts")

    ss = y_loadings.reshape(-1) ** 2 * np.sum(scores**2, axis=0)
    weight_norm = np.sum(weights**2, axis=0)
    if np.any(weight_norm == 0) or np.any(np.cumsum(ss) == 0):
        raise ValueError("PLS model contains zero-weight or zero-response components")
    ssw = weights**2 * (ss / weight_norm)
    vip_values = np.sqrt(weights.shape[0] * np.cumsum(ssw, axis=1) / np.cumsum(ss))

    columns = list(getattr(object_, "x_columns", []))
    if isinstance(object_, Mapping) and "x_columns" in object_:
        columns = list(object_["x_columns"])
    if not columns:
        columns = [f"x{idx + 1}" for idx in range(weights.shape[0])]
    index = [f"Component_{idx + 1}" for idx in range(weights.shape[1])]
    return pd.DataFrame(vip_values.T, index=index, columns=columns)


def pls_pheno(
    weather_data: Any,
    bio_data: Any,
    split_month: int = 7,
    runn_mean: int = 11,
    expl_var: float = 30,
    ncomp_fix: int | None = None,
    use_Tmean: bool = False,
    return_all: bool = False,
    crossvalidate: str = "none",
    end_at_pheno_end: bool | int | float = True,
) -> dict[str, Any]:
    """Run PLS analysis of phenological dates against daily mean temperature.

    Translates the non-plotting core of R ``PLS_pheno``. Cross-validation is
    accepted for API parity but not implemented in this lightweight PLS port.
    """
    if crossvalidate != "none":
        raise NotImplementedError("PLS cross-validation is not implemented")
    weather = _ensure_weather_dates(_as_dataframe(weather_data, name="weather_data"), name="weather_data")

    if use_Tmean:
        _require_columns(weather, ["Tmean"], name="weather_data")
        observed = pd.to_numeric(weather["Tmean"], errors="coerce").notna()
        if not observed.any():
            raise ValueError("weather_data Tmean contains no observed values")
        weather = weather.loc[observed.idxmax() : observed[::-1].idxmax()].copy()
        weather["Tmean"] = interpolate_gaps(weather["Tmean"])["interp"]
    else:
        _require_columns(weather, ["Tmin", "Tmax"], name="weather_data")
        weather["Tmin"] = pd.to_numeric(weather["Tmin"], errors="coerce")
        weather["Tmax"] = pd.to_numeric(weather["Tmax"], errors="coerce")
        observed = weather[["Tmin", "Tmax"]].notna().any(axis=1)
        if not observed.any():
            raise ValueError("weather_data Tmin/Tmax contain no observed values")
        weather = weather.loc[observed.idxmax() : observed[::-1].idxmax()].copy()
        swapped = weather["Tmin"] > weather["Tmax"]
        weather.loc[swapped, ["Tmin", "Tmax"]] = np.nan
        weather["Tmin"] = interpolate_gaps(weather["Tmin"])["interp"]
        weather["Tmax"] = interpolate_gaps(weather["Tmax"])["interp"]
        weather["Tmean"] = (weather["Tmin"] + weather["Tmax"]) / 2.0

    weather = _season_label_dates(weather, int(split_month))
    weather["runn"] = runn_mean_func(weather["Tmean"].to_numpy(dtype=float), int(runn_mean))
    x_all, seasons, labdates, labjdates, columns = _season_matrix(
        weather, value_columns=["runn"], prefix=["runn"]
    )
    bio = _coerce_pheno(bio_data)
    x, y, pheno = _filter_to_pheno(x_all, seasons, bio)
    x, labdates, labjdates = _clip_at_pheno_end(
        x, labdates, labjdates, pheno, end_at_pheno_end, paired=False
    )
    columns = columns[: x.shape[1]]
    model = _fit_pls_summary(x, y, x_columns=columns, expl_var=float(expl_var), ncomp_fix=ncomp_fix)
    vip_scores = vip(model).iloc[model.n_components - 1].to_numpy(dtype=float)
    adjusted_jdays = _adjust_split_jdays(labjdates, int(split_month))

    summary = pd.DataFrame(
        {
            "Date": labdates,
            "JDay": adjusted_jdays,
            "Coef": model.coefficients,
            "VIP": vip_scores,
            "Tmean": np.nanmean(x, axis=0),
            "Tstdev": np.nanstd(x, axis=0, ddof=1),
        }
    )
    output: dict[str, Any] = {
        "object_type": "PLS_Temp_pheno",
        "pheno": pheno,
        "PLS_summary": summary,
    }
    if return_all:
        output["PLS_output"] = model
    return output


def pls_chill_force(
    daily_chill_obj: Any,
    bio_data_frame: Any,
    split_month: int,
    expl_var: float = 30,
    ncomp_fix: int | None = None,
    return_all: bool = False,
    crossvalidate: str = "none",
    end_at_pheno_end: bool | int | float = True,
    chill_models: Sequence[str] = ("Chilling_Hours", "Utah_Chill_Units", "Chill_Portions"),
    heat_models: Sequence[str] = ("GDH",),
    runn_means: int | Sequence[int] = 1,
    metric_categories: Sequence[str] = ("Chill", "Heat"),
) -> dict[str, Any]:
    """Run PLS analysis of phenology against daily chill and heat metrics.

    Translates the non-plotting core of R ``PLS_chill_force`` for daily chill
    objects produced by :func:`chillPy.temperature.daily_chill`.
    """
    if crossvalidate != "none":
        raise NotImplementedError("PLS cross-validation is not implemented")
    if not isinstance(daily_chill_obj, Mapping) or daily_chill_obj.get("object_type") != "daily_chill":
        raise ValueError("daily_chill_obj must be a daily_chill object")
    if len(metric_categories) != 2:
        raise ValueError("metric_categories must contain two labels")

    weather = _ensure_weather_dates(_as_dataframe(daily_chill_obj["daily_chill"], name="daily_chill"), name="daily_chill")
    weather = _season_label_dates(weather, int(split_month))
    all_models = [*map(str, chill_models), *map(str, heat_models)]
    _require_columns(weather, all_models, name="daily_chill")

    if np.isscalar(runn_means):
        runners = [int(runn_means)] * len(all_models)
    else:
        runners = [int(value) for value in list(runn_means)]
        if len(runners) != len(all_models):
            raise ValueError("runn_means must contain one value or one value per selected metric")
    for model_name, runner in zip(all_models, runners, strict=True):
        weather[model_name] = runn_mean_func(pd.to_numeric(weather[model_name], errors="coerce"), runner)

    bio = _coerce_pheno(bio_data_frame, name="bio_data_frame")
    output: dict[str, Any] = {"object_type": "PLS_chillforce_pheno", "pheno": bio}
    for chill_model in map(str, chill_models):
        output.setdefault(chill_model, {})
        for heat_model in map(str, heat_models):
            x_all, seasons, labdates, labjdates, columns = _season_matrix(
                weather,
                value_columns=[chill_model, heat_model],
                prefix=[str(metric_categories[0]), str(metric_categories[1])],
            )
            x, y, pheno = _filter_to_pheno(x_all, seasons, bio)
            x, clipped_dates, clipped_jdates = _clip_at_pheno_end(
                x, labdates, labjdates, pheno, end_at_pheno_end, paired=True
            )
            keep = len(clipped_dates)
            columns = columns[:keep] + columns[365 : 365 + keep]
            model = _fit_pls_summary(x, y, x_columns=columns, expl_var=float(expl_var), ncomp_fix=ncomp_fix)
            vip_scores = vip(model).iloc[model.n_components - 1].to_numpy(dtype=float)
            adjusted_jdays = _adjust_split_jdays(clipped_jdates, int(split_month))
            summary = pd.DataFrame(
                {
                    "Date": np.concatenate([clipped_dates, clipped_dates]),
                    "Type": [metric_categories[0]] * keep + [metric_categories[1]] * keep,
                    "JDay": np.concatenate([adjusted_jdays, adjusted_jdays]),
                    "Coef": model.coefficients,
                    "VIP": vip_scores,
                    "MetricMean": np.nanmean(x, axis=0),
                    "MetricStdev": np.nanstd(x, axis=0, ddof=1),
                }
            )
            item: dict[str, Any] = {"PLS_summary": summary}
            if return_all:
                item["PLS_output"] = model
            output[chill_model][heat_model] = item
            output["pheno"] = pheno
    return output


def phenology_fit() -> dict[str, Any]:
    """Return an empty Python representation of R class ``phenologyFit``."""
    return {
        "object_type": "phenologyFit",
        "model_fit": None,
        "par": None,
        "pbloomJDays": None,
        "bloomJDays": None,
        "bloomJDaysunwrapped": None,
        "par.guess": None,
        "modelfn": None,
        "SeasonList": None,
        "lower": None,
        "upper": None,
        "control": None,
        "residuals": None,
        "rmse": None,
        "fitted": False,
    }


def phenology_fitter(
    par_guess: Any = None,
    modelfn: Any = None,
    bloom_jdays: Any = None,
    season_list: Any = None,
    control: Mapping[str, Any] | None = None,
    lower: Any = None,
    upper: Any = None,
    seed: int = 1235433,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit deterministic phenology wrapper parameters.

    This translates the non-plotting core of R ``phenologyFitter``: seasons are
    unwrapped across New Year, predictions are produced through a model wrapper,
    and parameters are chosen by minimizing the R-style residual sum of squares.
    Python uses a deterministic bounded coordinate search instead of R's
    stochastic ``GenSA`` optimizer.
    """
    kwargs = dict(kwargs)
    par_guess = _pop_alias(kwargs, "par.guess", par_guess)
    bloom_jdays = _pop_alias(kwargs, "bloomJDays", bloom_jdays)
    season_list = _pop_alias(kwargs, "SeasonList", season_list)
    na_penalty = float(kwargs.pop("na_penalty", 365.0))
    if kwargs:
        model_kwargs = kwargs
    else:
        model_kwargs = {}

    seasons = _coerce_season_list(season_list)
    bloom = _as_numeric_vector(bloom_jdays, name="bloomJDays")
    if bloom.size != len(seasons):
        raise ValueError("SeasonList and bloomJDays must have the same length")
    if not np.isfinite(bloom).any():
        raise ValueError("at least one observed bloom JDay must be finite")

    model = _resolve_phenology_model(modelfn)
    lower_arr = _coerce_parameter_vector(lower, name="lower")
    upper_arr = _coerce_parameter_vector(upper, name="upper")
    par_arr = _coerce_parameter_vector(par_guess, name="par.guess")
    if par_arr is None:
        if lower_arr is None or upper_arr is None:
            raise ValueError("par.guess is required unless finite lower and upper bounds are provided")
        if lower_arr.size != upper_arr.size:
            raise ValueError("lower and upper must have the same length")
        if np.isinf(lower_arr).any() or np.isinf(upper_arr).any():
            raise ValueError("finite lower and upper bounds are required when par.guess is omitted")
        par_arr = (lower_arr + upper_arr) / 2.0
    if lower_arr is None:
        lower_arr = par_arr.copy()
    if upper_arr is None:
        upper_arr = par_arr.copy()
    if lower_arr.size != par_arr.size or upper_arr.size != par_arr.size:
        raise ValueError("par.guess, lower, and upper must have the same length")
    if np.any(lower_arr > upper_arr):
        raise ValueError("lower bounds must be less than or equal to upper bounds")
    if np.any(par_arr < lower_arr) or np.any(par_arr > upper_arr):
        raise ValueError("par.guess must be within lower and upper bounds")

    fit_seasons, stored_seasons, unwrapped_bloom = _unwrap_cross_year_seasons(seasons, bloom)
    fit_control = _coerce_control(control, seed=int(seed))

    def objective(params: np.ndarray) -> float:
        return _phenology_rss(
            params,
            modelfn=model,
            bloom_jdays=unwrapped_bloom,
            season_list=fit_seasons,
            na_penalty=na_penalty,
            kwargs=model_kwargs,
        )

    model_fit = _bounded_coordinate_search(
        objective,
        par_arr,
        lower_arr,
        upper_arr,
        control=fit_control,
    )
    fitted_par = np.asarray(model_fit["par"], dtype=float)
    predicted = predict_bloom_days(fitted_par, stored_seasons, modelfn=model, **model_kwargs)
    residuals = bloom - predicted
    finite_residuals = residuals[np.isfinite(residuals)]
    rmse = float(np.sqrt(np.mean(finite_residuals**2))) if finite_residuals.size else np.nan
    residual_frame = pd.DataFrame(
        {
            "Season": np.arange(1, len(stored_seasons) + 1, dtype=int),
            "observed": bloom,
            "predicted": predicted,
            "residual": residuals,
        }
    )

    res = phenology_fit()
    res.update(
        {
            "model_fit": model_fit,
            "par": fitted_par,
            "pbloomJDays": predicted,
            "bloomJDays": bloom,
            "bloomJDaysunwrapped": unwrapped_bloom,
            "par.guess": par_arr,
            "modelfn": model,
            "modelfn_name": getattr(model, "__name__", str(model)),
            "SeasonList": stored_seasons,
            "fit_SeasonList": fit_seasons,
            "lower": lower_arr,
            "upper": upper_arr,
            "control": fit_control,
            "residuals": residual_frame,
            "rmse": rmse,
            "objective": float(model_fit["value"]),
            "na_penalty": na_penalty,
            "fitted": True,
        }
    )
    return res


def bootstrap_phenology_fit(
    object_: Any,
    boot_r: int = 99,
    control: Mapping[str, Any] | None = None,
    lower: Any = None,
    upper: Any = None,
    seed: int = 1766588,
) -> dict[str, Any]:
    """Bootstrap a ``phenologyFit`` object.

    Translates R ``bootstrap.phenologyFit``.
    Internally calls ``phenology_fitter`` on each bootstrap replicate.
    """
    if object_ is None:
        raise ValueError("object must be provided")
    if boot_r <= 1:
        raise ValueError("boot_r must be greater than 1")

    if lower is None:
        lower = object_.get("lower")
    if upper is None:
        upper = object_.get("upper")

    bloom_jdays = np.asarray(object_["bloomJDays"], dtype=float)
    pbloom_jdays = np.asarray(object_["pbloomJDays"], dtype=float)
    residuals = bloom_jdays - pbloom_jdays

    if np.all(residuals == 0):
        raise ValueError("All residuals equal to zero, no variation. Aborting")

    boot_res: dict[str, Any] = {
        "res": [],
        "boot_R": boot_r,
        "object": object_,
        "seed": seed,
        "lower": lower,
        "upper": upper,
        "object_type": "bootstrap_phenologyFit",
    }

    rng = np.random.default_rng(seed)

    for _ in range(boot_r):
        resampled_bloom = pbloom_jdays + rng.choice(residuals, size=len(residuals), replace=True)
        tmp = phenology_fitter(
            par_guess=object_["par"],
            modelfn=object_["modelfn"],
            bloom_jdays=resampled_bloom,
            season_list=object_["SeasonList"],
            control=control,
            lower=lower,
            upper=upper,
            seed=seed,
        )
        boot_res["res"].append(
            {
                "par": tmp["par"],
                "value": tmp["model_fit"]["value"] if tmp["model_fit"] else np.nan,
                "bloomJDays": resampled_bloom,
                "pbloomJDays": tmp["pbloomJDays"],
            }
        )

    return boot_res


def gen_season(temps: Any, mrange: tuple[int, int] = (8, 6), years: Any = None) -> list[np.ndarray]:
    """Return row indices belonging to each requested dormancy season.

    Translates R ``genSeason``. The R function returns 1-based row numbers;
    Python returns zero-based NumPy integer arrays suitable for ``iloc``.
    """
    start_month, end_month = _validate_mrange(mrange)
    selected_years = _normalize_years(years)
    frame = _temperature_frame(temps)
    _require_columns(frame, ["Year", "Month"], name="temps")

    return [
        _season_indices(frame, start_month=start_month, end_month=end_month, year=year)
        for year in selected_years
    ]


def gen_season_list(temps: Any, mrange: tuple[int, int] = (8, 6), years: Any = None) -> list[pd.DataFrame]:
    """Generate per-season temperature tables.

    Translates R ``genSeasonList``. Each returned frame has columns ``Temp``,
    ``JDay``, and ``Year`` and corresponds to one end-year in ``years``.
    """
    start_month, end_month = _validate_mrange(mrange, require_cross_year=True)
    selected_years = _normalize_years(years)
    frame = _temperature_frame(temps)
    _require_columns(frame, ["Year", "Month", "JDay", "Temp"], name="temps")

    season_list: list[pd.DataFrame] = []
    for year in selected_years:
        indices = _season_indices(frame, start_month=start_month, end_month=end_month, year=year)
        season = frame.iloc[indices][["Temp", "JDay", "Year"]].copy().reset_index(drop=True)
        season_list.append(season)
    return season_list


def stage_transitions(
    observations: Any,
    hourtemps: Any,
    stages: Sequence[str],
    models: Mapping[str, Callable[..., Any]] | None = None,
    max_steps: int | None = None,
) -> pd.DataFrame:
    """Compute temperature metrics accrued between observed phenology stages.

    Translates R ``stage_transitions``. Missing or duplicate stage observations
    leave transition metrics as ``NaN``.
    """
    stage_order = [str(stage) for stage in stages]
    if not stage_order:
        raise ValueError("stages must contain at least one stage")
    if len(set(stage_order)) != len(stage_order):
        raise ValueError("stages must not contain duplicates")
    if max_steps is None:
        max_steps = len(stage_order)
    if int(max_steps) < 1 or int(max_steps) > len(stage_order):
        raise ValueError("max_steps must be between 1 and len(stages)")

    obs = _as_dataframe(observations, name="observations")
    _require_columns(obs, ["Stage", "Season", "Year", "JDay"], name="observations")
    unknown_stages = sorted(set(obs["Stage"].dropna().astype(str)) - set(stage_order))
    if unknown_stages:
        raise ValueError(f"observations contain stages not listed in stages: {', '.join(unknown_stages)}")
    for column in ["Season", "Year", "JDay"]:
        obs[column] = pd.to_numeric(obs[column], errors="coerce")
    if obs[["Season", "Year", "JDay"]].isna().any(axis=None):
        raise ValueError("observations Season, Year, and JDay must be numeric")
    obs["Stage"] = obs["Stage"].astype(str)

    temps = _temperature_frame(hourtemps)
    _require_columns(temps, ["Year", "Temp"], name="hourtemps")
    if "JDay" not in temps.columns:
        temps["JDay"] = make_jday(temps)["JDay"]
    _require_columns(temps, ["JDay", "Hour", "Temp"], name="hourtemps")
    for column in ["Year", "JDay", "Hour", "Temp"]:
        temps[column] = pd.to_numeric(temps[column], errors="coerce")
    if temps[["Year", "JDay", "Hour"]].isna().any(axis=None):
        raise ValueError("hourtemps Year, JDay, and Hour must be numeric")
    temps = temps.sort_values(["Year", "JDay", "Hour"], kind="mergesort").reset_index(drop=True)

    selected_models: dict[str, Callable[..., Any]] = dict(
        {"Chill_Portions": dynamic_model, "GDH": gdh} if models is None else models
    )
    seasons = list(pd.unique(obs["Season"].dropna().astype(int)))
    stage_position = {stage: idx for idx, stage in enumerate(stage_order)}

    rows: list[dict[str, Any]] = []
    for season in seasons:
        for start_stage in stage_order:
            for end_stage in stage_order:
                steps = stage_position[end_stage] - stage_position[start_stage]
                if steps < 1:
                    steps += len(stage_order)
                if steps > int(max_steps):
                    continue
                row: dict[str, Any] = {
                    "Season": season,
                    "Stage": start_stage,
                    "to_Stage": end_stage,
                    "stage_steps": steps,
                }
                row.update({model_name: np.nan for model_name in selected_models})

                to_season = season + 1 if stage_position[end_stage] <= stage_position[start_stage] else season
                start_obs = obs[(obs["Season"] == season) & (obs["Stage"] == start_stage)]
                end_obs = obs[(obs["Season"] == to_season) & (obs["Stage"] == end_stage)]
                if len(start_obs) == 1 and len(end_obs) == 1:
                    start = start_obs.iloc[0]
                    end = end_obs.iloc[0]
                    start_matches = temps.index[
                        (temps["Year"] == start["Year"]) & (temps["JDay"] == start["JDay"])
                    ].to_numpy()
                    end_matches = temps.index[
                        (temps["Year"] == end["Year"]) & (temps["JDay"] == end["JDay"])
                    ].to_numpy()
                    if start_matches.size and end_matches.size and start_matches[0] <= end_matches[-1]:
                        weather_slice = temps.iloc[start_matches[0] : end_matches[-1] + 1]
                        values = temp_response(
                            weather_slice,
                            start_jday=int(start["JDay"]),
                            end_jday=int(end["JDay"]),
                            models=selected_models,
                            whole_record=True,
                        )
                        for model_name in selected_models:
                            row[model_name] = float(values[model_name])
                rows.append(row)
    return pd.DataFrame(rows, columns=["Season", "Stage", "to_Stage", "stage_steps", *selected_models.keys()])


def unified_model_wrapper(x: Any, par: Any) -> float:
    """Evaluate R ``UnifiedModel_Wrapper`` for one season.

    ``par`` has length 9: ``ac, bc, cc, bf, cf, w, k, Cstar, tc``. The
    returned value is the predicted Julian day or ``nan`` when requirements are
    not met.
    """
    temps, jdays, params = _prepare_wrapper_inputs(x, par, par_length=9, name="UnifiedModel_Wrapper")
    tend = temps.size
    if tend < params[8] or params[5] <= 0 or params[6] >= 0:
        return np.nan
    chilling = np.cumsum(_chuine_cf(temps, params[0], params[1], params[2]))
    if chilling[-1] < params[7]:
        return np.nan
    t1 = int(np.flatnonzero(chilling >= params[7])[0])
    tc = int(round(params[8]))
    if tc < 1:
        return np.nan
    ctot = float(np.sum(_chuine_cf(temps[:tc], params[0], params[1], params[2])))
    fstar = _chuine_fstar(ctot, params[5], params[6])
    forcing = _chuine_cf(temps[t1:], 0.0, params[3], params[4])
    cumulative = np.cumsum(forcing)
    if cumulative[-1] < fstar:
        return np.nan
    return _jday_from_r_relative_crossing(jdays, cumulative, fstar)


def uni_chill_wrapper(x: Any, par: Any) -> float:
    """Evaluate R ``UniChill_Wrapper`` for one season.

    ``par`` has length 7: ``ac, bc, cc, bf, cf, Cstar, Fstar``.
    """
    temps, jdays, params = _prepare_wrapper_inputs(x, par, par_length=7, name="UniChill_Wrapper")
    chilling = np.cumsum(_chuine_cf(temps, params[0], params[1], params[2]))
    if chilling[-1] < params[5]:
        return np.nan
    t1 = int(np.flatnonzero(chilling >= params[5])[0])
    forcing = _chuine_cf(temps[t1:], 0.0, params[3], params[4])
    cumulative = np.cumsum(forcing)
    if cumulative[-1] < params[6]:
        return np.nan
    return _jday_from_r_relative_crossing(jdays, cumulative, params[6])


def step_chill_wrapper(x: Any, par: Any) -> float:
    """Evaluate R ``StepChill_Wrapper`` for one season.

    ``par`` has length 5 in the R implementation: ``Tc, bf, cf, Cstar,
    Fstar``. The Rd file says length 7, but only five parameters are used.
    """
    temps, jdays, params = _prepare_wrapper_inputs(x, par, par_length=5, name="StepChill_Wrapper")
    chilling = np.cumsum((temps <= params[0]).astype(float))
    if chilling[-1] < params[3]:
        return np.nan
    t1 = int(np.flatnonzero(chilling >= params[3])[0])
    forcing = _chuine_cf(temps[t1:], 0.0, params[1], params[2])
    cumulative = np.cumsum(forcing)
    if cumulative[-1] < params[4]:
        return np.nan
    return _jday_from_r_relative_crossing(jdays, cumulative, params[4])


def uni_force_wrapper(x: Any, par: Any) -> float:
    """Evaluate R ``UniForce_Wrapper`` for one season.

    ``par`` has length 4: ``bf, cf, Fstar, t1``.
    """
    temps, jdays, params = _prepare_wrapper_inputs(x, par, par_length=4, name="UniForce_Wrapper")
    tend = temps.size
    t1 = int(round(params[3]))
    if t1 >= tend or t1 < 1:
        return np.nan
    forcing = _chuine_cf(temps[t1 - 1 :], 0.0, params[0], params[1])
    cumulative = np.cumsum(forcing)
    if cumulative[-1] < params[2]:
        return np.nan
    return _jday_from_r_relative_crossing(jdays, cumulative, params[2])


def _phenoflex_smooth(jdays: np.ndarray, bloom_index: int) -> float:
    """Apply the smoothing logic from R PhenoFlex wrappers."""
    if bloom_index == 0:
        return np.nan
    # bloom_index is 1-based index from C++ (i+2)
    # in Python it corresponds to index bloom_index - 1
    idx = bloom_index - 1
    if idx >= len(jdays):
        return np.nan

    jday_val = jdays[idx]
    jday_list = np.flatnonzero(jdays == jday_val)
    n = len(jday_list)
    if n == 1:
        return float(jday_val)

    # which(JDaylist == bloomindex)
    pos = np.flatnonzero(jday_list == idx)[0] + 1
    return float(jday_val + pos / n - 1.0 / (n / np.ceil(n / 2.0)))


def phenoflex_gdh_wrapper(x: Any, par: Any) -> float:
    """Evaluate R ``PhenoFlex_GDHwrapper`` for one season.

    ``par`` has length 12: yc, zc, s1, Tu, E0, E1, A0, A1, Tf, Tc, Tb, slope.
    """
    frame = _as_dataframe(x, name="x")
    _require_columns(frame, ["Temp", "JDay"], name="x")
    temps = frame["Temp"].values
    jdays = frame["JDay"].values
    params = _coerce_parameter_vector(par, name="par")

    if len(params) < 12:
        raise ValueError("par must have length 12 for PhenoFlex_GDHwrapper")

    # par[4] <= par[11] (Tu <= Tb)
    if params[3] <= params[10]:
        return np.nan
    # par[10] <= par[4] (Tc <= Tu)
    if params[9] <= params[3]:
        return np.nan

    res = phenoflex(
        temp=temps,
        times=np.arange(1, len(temps) + 1),
        yc=params[0],
        zc=params[1],
        s1=params[2],
        tu=params[3],
        e0=params[4],
        e1=params[5],
        a0=params[6],
        a1=params[7],
        tf=params[8],
        tc=params[9],
        tb=params[10],
        slope=params[11],
        imodel=0,
        basic_output=False,
    )
    return _phenoflex_smooth(jdays, res["bloomindex"])


def phenoflex_fixed_dynamic_model_wrapper(
    x: Any,
    par: Any,
    a0: float = 139500,
    a1: float = 2567000000000000000,
    e0: float = 4153.5,
    e1: float = 12888.8,
    slope: float = 1.6,
    tf: float = 4,
) -> float:
    """Evaluate R ``PhenoFlex_fixedDynModelwrapper`` for one season.

    ``par`` has length 6: yc, zc, s1, Tu, Tc, Tb.
    """
    frame = _as_dataframe(x, name="x")
    _require_columns(frame, ["Temp", "JDay"], name="x")
    temps = frame["Temp"].values
    jdays = frame["JDay"].values
    params = _coerce_parameter_vector(par, name="par")

    if len(params) < 6:
        raise ValueError("par must have length 6 for PhenoFlex_fixedDynModelwrapper")

    # par[4] <= par[6] (Tu <= Tb)
    if params[3] <= params[5]:
        return np.nan
    # par[5] <= par[4] (Tc <= Tu)
    if params[4] <= params[3]:
        return np.nan

    res = phenoflex(
        temp=temps,
        times=np.arange(1, len(temps) + 1),
        yc=params[0],
        zc=params[1],
        s1=params[2],
        tu=params[3],
        tc=params[4],
        tb=params[5],
        e0=e0,
        e1=e1,
        a0=a0,
        a1=a1,
        tf=tf,
        slope=slope,
        imodel=0,
        basic_output=False,
    )
    return _phenoflex_smooth(jdays, res["bloomindex"])


def phenoflex_gauss_wrapper(x: Any, par: Any) -> float:
    """Evaluate R ``PhenoFlex_GAUSSwrapper`` for one season.

    ``par`` has length 11: yc, zc, s1, Tu, E0, E1, A0, A1, Tf, Delta, slope.
    """
    frame = _as_dataframe(x, name="x")
    _require_columns(frame, ["Temp", "JDay"], name="x")
    temps = frame["Temp"].values
    jdays = frame["JDay"].values
    params = _coerce_parameter_vector(par, name="par")

    if len(params) < 11:
        raise ValueError("par must have length 11 for PhenoFlex_GAUSSwrapper")

    res = phenoflex(
        temp=temps,
        times=np.arange(1, len(temps) + 1),
        yc=params[0],
        zc=params[1],
        s1=params[2],
        tu=params[3],
        e0=params[4],
        e1=params[5],
        a0=params[6],
        a1=params[7],
        tf=params[8],
        delta=params[9],
        slope=params[10],
        imodel=1,
        basic_output=False,
    )
    return _phenoflex_smooth(jdays, res["bloomindex"])


def phenoflex_fixed_dynamic_model_gauss_wrapper(
    x: Any,
    par: Any,
    a0: float = 139500,
    a1: float = 2567000000000000000,
    e0: float = 4153.5,
    e1: float = 12888.8,
    slope: float = 1.6,
    tf: float = 4,
) -> float:
    """Evaluate R ``PhenoFlex_fixedDynModelGAUSSwrapper`` for one season.

    ``par`` has length 5: yc, zc, s1, Tu, Delta.
    """
    frame = _as_dataframe(x, name="x")
    _require_columns(frame, ["Temp", "JDay"], name="x")
    temps = frame["Temp"].values
    jdays = frame["JDay"].values
    params = _coerce_parameter_vector(par, name="par")

    if len(params) < 5:
        raise ValueError("par must have length 5 for PhenoFlex_fixedDynModelGAUSSwrapper")

    res = phenoflex(
        temp=temps,
        times=np.arange(1, len(temps) + 1),
        yc=params[0],
        zc=params[1],
        s1=params[2],
        tu=params[3],
        delta=params[4],
        e0=e0,
        e1=e1,
        a0=a0,
        a1=a1,
        tf=tf,
        slope=slope,
        imodel=1,
        basic_output=False,
    )
    return _phenoflex_smooth(jdays, res["bloomindex"])


PLS_chill_force = pls_chill_force
PLS_pheno = pls_pheno
VIP = vip
phenologyFit = phenology_fit
phenologyFitter = phenology_fitter
genSeason = gen_season
genSeasonList = gen_season_list
UnifiedModel_Wrapper = unified_model_wrapper
UniChill_Wrapper = uni_chill_wrapper
StepChill_Wrapper = step_chill_wrapper
UniForce_Wrapper = uni_force_wrapper
PhenoFlex_GDHwrapper = phenoflex_gdh_wrapper
PhenoFlex_fixedDynModelwrapper = phenoflex_fixed_dynamic_model_wrapper
PhenoFlex_GAUSSwrapper = phenoflex_gauss_wrapper
PhenoFlex_fixedDynModelGAUSSwrapper = phenoflex_fixed_dynamic_model_gauss_wrapper
bootstrap_phenology_fit = bootstrap_phenology_fit
globals()["bootstrap.phenologyFit"] = bootstrap_phenology_fit
