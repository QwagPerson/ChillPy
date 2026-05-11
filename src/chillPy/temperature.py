"""Temperature record processing functions mapped from chillR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd

from ._base import placeholder_record
from .date_utils import daylength
from .temperature_models import chilling_hours, dynamic_model, gdh, gdh_model, utah_model
from .utils import runn_mean


_DEFAULT_RESPONSE_MODELS: dict[str, Callable[..., Any]] = {
    "Chilling_Hours": chilling_hours,
    "Utah_Chill_Units": utah_model,
    "Chill_Portions": dynamic_model,
    "GDH": gdh,
}

_DEFAULT_HOURTABLE_MODELS: dict[str, Callable[..., Any]] = {
    "Chill_Portions": dynamic_model,
    "GDH": gdh_model,
}


def _as_dataframe(data: Any, *, name: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    try:
        return pd.DataFrame(data).copy()
    except Exception as exc:  # pragma: no cover - defensive for unusual table-likes
        raise TypeError(f"{name} must be convertible to a pandas DataFrame") from exc


def _require_columns(frame: pd.DataFrame, columns: list[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{name} is missing required column(s): {joined}")


def _extract_hourtemps(data: Any, *, name: str = "hourtemps") -> tuple[pd.DataFrame, Any]:
    """Return the hourly DataFrame and optional QC payload from a chillR-like object."""
    if isinstance(data, Mapping) and "hourtemps" in data:
        qc = data.get("QC", np.nan)
        return _as_dataframe(data["hourtemps"], name=name), qc
    return _as_dataframe(data, name=name), np.nan


def _prepare_hourtemps(data: Any, *, name: str = "hourtemps") -> tuple[pd.DataFrame, Any]:
    frame, qc = _extract_hourtemps(data, name=name)
    _require_columns(frame, ["Year", "JDay", "Hour", "Temp"], name=name)
    frame = frame.copy()
    for column in ["Year", "JDay", "Hour", "Temp"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, qc


def _default_models(models: dict[str, Callable[..., Any]] | None) -> dict[str, Callable[..., Any]]:
    if models is None:
        return dict(_DEFAULT_RESPONSE_MODELS)
    return dict(models)


def _call_model(model: Callable[..., Any], temps: np.ndarray, *, summ: bool) -> np.ndarray:
    """Call a chillR-style model while tolerating simple one-argument callables."""
    try:
        values = model(temps, summ=summ)
    except TypeError:
        values = model(temps)
    out = np.asarray(values, dtype=float)
    if out.ndim == 0:
        out = out.reshape(1)
    if out.size != temps.size:
        raise ValueError("model output length must match hourly temperature input length")
    return out.ravel()


def _legacy_dynamic_chill_portions(temps: np.ndarray, *, summ: bool = False) -> np.ndarray:
    """Dynamic Model recurrence as in R ``chilling`` and ``chilling_hourtable``."""
    if temps.size == 0:
        return np.array([], dtype=float)
    if np.isnan(temps).any():
        raise ValueError("legacy dynamic chill calculation does not accept missing temperatures")

    tk = temps + 273.0
    e0 = 4153.5
    e1 = 12888.8
    a0 = 139500.0
    a1 = 2.567e18
    slope = 1.6
    tf = 277.0
    aa = a0 / a1
    ee = e1 - e0
    xi = np.exp(slope * tf * (tk - tf) / tk)
    xi = xi / (1.0 + xi)
    xs = aa * np.exp(ee / tk)
    ak1 = a1 * np.exp(-e1 / tk)

    state = ak1.copy()
    previous = ak1.copy()
    state[0] = 0.0
    previous[0] = 0.0
    for idx in range(1, temps.size):
        if state[idx - 1] < 1.0:
            previous[idx] = state[idx - 1]
        else:
            previous[idx] = state[idx - 1] - state[idx - 1] * xi[idx - 1]
        state[idx] = xs[idx] - (xs[idx] - previous[idx]) * np.exp(-ak1[idx])

    delta = np.zeros(temps.shape, dtype=float)
    exceeded = state >= 1.0
    delta[exceeded] = state[exceeded] * xi[exceeded]
    return np.cumsum(delta) if summ else delta


_DEFAULT_CHILLING_MODELS: dict[str, Callable[..., Any]] = {
    "Chilling_Hours": chilling_hours,
    "Utah_Model": utah_model,
    "Chill_portions": _legacy_dynamic_chill_portions,
    "GDH": gdh,
}


def _assign_seasons(frame: pd.DataFrame, start_jday: int, end_jday: int) -> pd.Series:
    sea = pd.Series(np.nan, index=frame.index, dtype=float)
    if start_jday < end_jday:
        mask = (frame["JDay"] >= start_jday) & (frame["JDay"] <= end_jday)
        sea.loc[mask] = frame.loc[mask, "Year"]
    else:
        after_start = frame["JDay"] >= start_jday
        before_end = frame["JDay"] <= end_jday
        sea.loc[after_start] = frame.loc[after_start, "Year"] + 1
        sea.loc[before_end] = frame.loc[before_end, "Year"]
    return sea


def _date_from_jday(year: int, jday: int) -> date:
    return date(int(year), 1, 1) + timedelta(days=int(jday) - 1)


def _season_days(end_year: int, start_jday: int, end_jday: int) -> int:
    if end_jday >= start_jday:
        start = _date_from_jday(end_year, start_jday)
        end = _date_from_jday(end_year, end_jday)
    else:
        start = _date_from_jday(end_year - 1, start_jday)
        end = _date_from_jday(end_year, end_jday)
    return (end - start).days + 1


def _season_label(end_year: int) -> str:
    return f"{end_year - 1}/{end_year}"


def _ensure_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if {"Year", "Month", "Day"}.issubset(frame.columns):
        return frame
    _require_columns(frame, ["Year", "JDay"], name="hourtemps")
    out = frame.copy()
    dates = [
        _date_from_jday(int(year), int(jday)) if not (np.isnan(year) or np.isnan(jday)) else pd.NaT
        for year, jday in zip(out["Year"], out["JDay"], strict=False)
    ]
    dt = pd.to_datetime(pd.Series(dates, index=out.index), errors="coerce")
    out["Month"] = dt.dt.month.astype(float)
    out["Day"] = dt.dt.day.astype(float)
    return out


def _reset_cumulative_from_start(weights: np.ndarray, start_day_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(weights, dtype=float).copy()
    if out.size == 0:
        return out
    out[0] = 0.0
    for idx in range(1, out.size):
        if start_day_mask[idx]:
            out[idx] = 0.0
        else:
            out[idx] = out[idx - 1] + weights[idx]
    return out


def _add_jday_if_missing(frame: pd.DataFrame) -> pd.DataFrame:
    if "JDay" in frame.columns:
        return frame
    _require_columns(frame, ["Year", "Month", "Day"], name="year_file")
    dates = pd.to_datetime(
        {
            "year": pd.to_numeric(frame["Year"], errors="coerce"),
            "month": pd.to_numeric(frame["Month"], errors="coerce"),
            "day": pd.to_numeric(frame["Day"], errors="coerce"),
        },
        errors="coerce",
    )
    frame = frame.copy()
    frame["JDay"] = dates.dt.dayofyear.astype(float)
    return frame


def chilling(
    hourtemps: Any = None,
    start_jday: int = 1,
    end_jday: int = 366,
    *,
    thourly: Any = None,
    misstolerance: float = 50,
) -> pd.DataFrame:
    """Calculate standard chill and heat totals by season.

    Translates R ``chilling``. The input is a stacked hourly table, or a
    dictionary with ``hourtemps`` and optional ``QC`` as returned by
    :func:`stack_hourly_temps`.
    """
    source = hourtemps if hourtemps is not None else thourly
    if source is None:
        raise ValueError("hourtemps must be provided")
    return temp_response(
        source,
        start_jday=start_jday,
        end_jday=end_jday,
        models=dict(_DEFAULT_CHILLING_MODELS),
        misstolerance=misstolerance,
    )


def daily_chill(
    hourtemps: Any = None,
    running_mean: int = 1,
    models: dict[str, Callable[..., Any]] | None = None,
    *,
    thourly: Any = None,
) -> dict[str, Any]:
    """Calculate daily chill and heat accumulation from hourly temperatures.

    Translates R ``daily_chill``. Model functions are evaluated on hourly
    temperatures with ``summ=False`` and then summed by calendar day.
    """
    source = hourtemps if hourtemps is not None else thourly
    if source is None:
        raise ValueError("hourtemps must be provided")
    frame, qc = _prepare_hourtemps(source)
    frame = _ensure_calendar_columns(frame)
    _require_columns(frame, ["Year", "Month", "Day", "Temp"], name="hourtemps")
    if frame["Temp"].isna().any():
        raise ValueError("daily_chill does not accept missing hourly temperatures")

    selected_models = _default_models(models)
    temps = frame["Temp"].to_numpy(dtype=float)
    for model_name, model in selected_models.items():
        frame[model_name] = _call_model(model, temps, summ=False)

    frame["YYMMDD"] = (
        frame["Year"].astype(int) * 10000
        + frame["Month"].astype(int) * 100
        + frame["Day"].astype(int)
    )
    grouped = frame.groupby("YYMMDD", sort=True, dropna=False)
    daily = grouped[list(selected_models)].sum().reset_index()
    daily["Year"] = daily["YYMMDD"].astype(int) // 10000
    daily["Month"] = (daily["YYMMDD"].astype(int) - daily["Year"] * 10000) // 100
    daily["Day"] = daily["YYMMDD"].astype(int) - daily["Year"] * 10000 - daily["Month"] * 100
    daily = daily[["YYMMDD", "Year", "Month", "Day", *selected_models.keys()]]
    daily["Tmean"] = grouped["Temp"].mean().to_numpy(dtype=float)

    for model_name in selected_models:
        daily[model_name] = runn_mean(daily[model_name].to_numpy(dtype=float), int(running_mean))

    for column in ["no_Tmin", "no_Tmax"]:
        if column in frame.columns:
            daily[column] = grouped[column].sum().to_numpy(dtype=float) > 0

    return {
        "object_type": "daily_chill",
        "daily_chill": daily,
        "QC": qc if isinstance(qc, pd.DataFrame) else np.nan,
    }


def chilling_hourtable(hourtemps: Any, start_jday: int = 1) -> pd.DataFrame:
    """Add standard cumulative chill and heat metrics to an hourly table.

    Translates R ``chilling_hourtable``. Accumulation restarts on every
    ``start_jday``; as in R, all hours on the restart day are set to zero.
    """
    frame, _ = _prepare_hourtemps(hourtemps)
    original_columns = list(frame.columns)
    frame = frame.loc[frame["Temp"].notna()].copy()
    temps = frame["Temp"].to_numpy(dtype=float)
    start_mask = (frame["JDay"].to_numpy(dtype=float) == float(start_jday))

    weights = {
        "Chilling_Hours": _call_model(chilling_hours, temps, summ=False),
        "Chill_Portions": _call_model(_legacy_dynamic_chill_portions, temps, summ=False),
        "Chill_Units": _call_model(utah_model, temps, summ=False),
        "GDH": _call_model(gdh, temps, summ=False),
    }
    for column, values in weights.items():
        frame[column] = _reset_cumulative_from_start(values, start_mask)

    return frame[original_columns + ["Chilling_Hours", "Chill_Portions", "Chill_Units", "GDH"]]


def temp_response(
    hourtemps: Any,
    start_jday: int = 1,
    end_jday: int = 366,
    models: dict[str, Callable[..., Any]] | None = None,
    *,
    misstolerance: float = 50,
    whole_record: bool = False,
    mean_out: bool = False,
) -> pd.DataFrame | pd.Series:
    """Calculate seasonal totals for temperature-response models.

    Translates R ``tempResponse``. Model functions must return one value per
    hourly temperature; chillPy calls them as ``model(temps, summ=True)`` when
    possible, matching the cumulative model convention used by chillR.
    """
    frame, qc = _prepare_hourtemps(hourtemps)
    selected_models = _default_models(models)
    frame["sea"] = _assign_seasons(frame, int(start_jday), int(end_jday))
    frame = frame.loc[frame["Temp"].notna()].copy()
    temps = frame["Temp"].to_numpy(dtype=float)

    for model_name, model in selected_models.items():
        frame[model_name] = _call_model(model, temps, summ=True)

    if whole_record:
        if frame.empty:
            return pd.Series({model_name: np.nan for model_name in selected_models}, dtype=float)
        return frame.iloc[-1][list(selected_models)].astype(float)

    seasons = [int(season) for season in pd.unique(frame["sea"].dropna())]
    rows: list[dict[str, Any]] = []
    for season in seasons:
        season_mask = frame["sea"] == season
        season_frame = frame.loc[season_mask]
        if season_frame.empty:
            continue

        first_pos = frame.index.get_loc(season_frame.index[0])
        last_end_pos = max(0, first_pos - 1)
        last_end = frame.iloc[last_end_pos].copy()
        if season == seasons[0] and (start_jday > end_jday or start_jday == int(frame["JDay"].iloc[0])):
            for model_name in selected_models:
                last_end[model_name] = 0.0

        row: dict[str, Any] = {
            "Season": _season_label(season),
            "End_year": season,
            "Season_days": _season_days(season, int(start_jday), int(end_jday)),
            "Data_days": len(season_frame) / 24,
        }
        if mean_out:
            row["Input_mean"] = float(season_frame["Temp"].mean())
        if {"no_Tmin", "no_Tmax"}.issubset(frame.columns):
            row["Interpolated_days"] = (
                season_frame["no_Tmin"].astype(bool) | season_frame["no_Tmax"].astype(bool)
            ).sum() / 24
        for model_name in selected_models:
            row[model_name] = float(season_frame[model_name].iloc[-1] - float(last_end[model_name]))
        if "Interpolated_days" in row:
            row["Perc_complete"] = (
                (row["Data_days"] - row["Interpolated_days"]) / row["Season_days"] * 100
            )
        else:
            row["Perc_complete"] = row["Data_days"] / row["Season_days"] * 100
        rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output = output.loc[output["Perc_complete"] >= 100 - float(misstolerance)].reset_index(drop=True)
    if isinstance(qc, pd.DataFrame) and "End_year" in qc.columns and not output.empty:
        qc_extra = qc.loc[qc["End_year"].isin(output["End_year"])]
        if not qc_extra.empty:
            output = output.merge(qc_extra, on="End_year", how="left", suffixes=("", "_QC"))
    return output


def temp_response_daily_list(
    temperature_list: Any,
    latitude: float,
    start_jday: int = 1,
    end_jday: int = 366,
    models: dict[str, Callable[..., Any]] | None = None,
    *,
    misstolerance: float = 50,
    whole_record: bool = False,
    empirical: Any = None,
    mean_out: bool = False,
) -> list[pd.DataFrame | pd.Series] | dict[str, pd.DataFrame | pd.Series]:
    """Apply :func:`temp_response` to one or more daily temperature records.

    Translates R ``tempResponse_daily_list``. Daily records are converted to
    hourly using either idealized curves (default) or empirical coefficients
    if provided via `empirical`.

    Parameters
    ----------
    temperature_list : Any
        One or more daily temperature records (DataFrame, list of DataFrames,
        or dict of DataFrames).
    latitude : float
        Latitude for idealized hourly temperature generation.
    start_jday : int, optional
        Start day of the period.
    end_jday : int, optional
        End day of the period.
    models : dict, optional
        Models to apply.
    misstolerance : float, optional
        Missing value tolerance (percentage).
    whole_record : bool, optional
        Whether to sum over the entire record.
    empirical : Any, optional
        Empirical coefficients from ``empirical_daily_temperature_curve``.
    mean_out : bool, optional
        Whether to include input mean in output.

    Returns
    -------
    list or dict
        List or dictionary of DataFrames showing model totals for each season.
    """
    if isinstance(temperature_list, pd.DataFrame):
        records: Mapping[str, Any] = {"record_1": temperature_list}
        return_list = True
    elif isinstance(temperature_list, Mapping):
        records = temperature_list
        return_list = False
    else:
        records = {f"record_{i + 1}": rec for i, rec in enumerate(temperature_list)}
        return_list = True

    output: dict[str, pd.DataFrame | pd.Series] = {}
    for name, record in records.items():
        if empirical is None:
            hourtemps = stack_hourly_temps(record, latitude=latitude)
        else:
            hourtemps = empirical_hourly_temperatures(record, empirical)

        output[name] = temp_response(
            hourtemps,
            start_jday=start_jday,
            end_jday=end_jday,
            models=models,
            misstolerance=misstolerance,
            whole_record=whole_record,
            mean_out=mean_out,
        )

    if return_list:
        return list(output.values())
    return output


def temp_response_hourtable(
    hourtemps: Any,
    start_jday: int | float | None = None,
    models: dict[str, Callable[..., Any]] | None = None,
) -> pd.DataFrame:
    """Add cumulative model metrics to an hourly table.

    Translates R ``tempResponse_hourtable``. When ``start_jday`` is provided,
    cumulative model values are centered to zero at the first matching restart
    day in each season.
    """
    frame, _ = _prepare_hourtemps(hourtemps)
    original_columns = list(frame.columns)
    frame = frame.loc[frame["Temp"].notna()].copy()
    selected_models = dict(_DEFAULT_HOURTABLE_MODELS if models is None else models)
    temps = frame["Temp"].to_numpy(dtype=float)

    for model_name, model in selected_models.items():
        frame[model_name] = _call_model(model, temps, summ=True)

    if start_jday is not None and not pd.isna(start_jday):
        if "Season" not in frame.columns:
            frame["Season"] = np.where(
                frame["JDay"] >= float(start_jday),
                frame["Year"],
                frame["Year"] - 1,
            )
        for model_name in selected_models:
            for season in pd.unique(frame["Season"]):
                season_mask = frame["Season"] == season
                start_mask = season_mask & (frame["JDay"] == round(float(start_jday)))
                if start_mask.any():
                    offset = float(frame.loc[start_mask, model_name].iloc[0])
                else:
                    offset = float(frame.loc[season_mask, model_name].iloc[0])
                frame.loc[season_mask, model_name] = frame.loc[season_mask, model_name] - offset

    return frame[original_columns + list(selected_models)]


def make_hourly_temps(
    latitude: float,
    year_file: Any,
    *,
    keep_sunrise_sunset: bool = False,
) -> pd.DataFrame:
    """Generate hourly temperatures from daily minima and maxima.

    Translates R ``make_hourly_temps``. The input must contain ``Tmin`` and
    ``Tmax`` plus either ``JDay`` or ``Year``, ``Month``, and ``Day``. The
    returned DataFrame preserves input columns, adds ``JDay`` when needed, and
    appends ``Hour_0`` through ``Hour_23``.
    """
    latitude_array = np.asarray(latitude, dtype=float)
    if latitude_array.ndim > 0 and latitude_array.size != 1:
        raise ValueError("'latitude' has more than one element")
    latitude_value = float(latitude_array.reshape(1)[0])
    if latitude_value > 90 or latitude_value < -90:
        warnings.warn("'latitude' is usually between -90 and 90", RuntimeWarning, stacklevel=2)

    frame = _as_dataframe(year_file, name="year_file")
    _require_columns(frame, ["Tmin", "Tmax"], name="year_file")
    frame = frame.loc[frame["Tmin"].notna() & frame["Tmax"].notna()].copy()
    frame = _add_jday_if_missing(frame)
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return frame

    preserve_columns = list(frame.columns)
    jdays = pd.to_numeric(frame["JDay"], errors="coerce").to_numpy(dtype=float)
    day_times = daylength(latitude_value, np.concatenate([[jdays[0] - 1], jdays, [jdays[-1] + 1]]))
    sunrise = day_times["Sunrise"].copy()
    sunset = day_times["Sunset"].copy()
    day_length = day_times["Daylength"].copy()
    sunrise[sunrise == 99] = 0
    sunrise[sunrise == -99] = 12
    sunset[sunset == 99] = 24
    sunset[sunset == -99] = 12

    frame["Sunrise"] = sunrise[1:-1]
    frame["Sunset"] = sunset[1:-1]
    frame["Daylength"] = day_length[1:-1]
    frame["prev_Sunset"] = sunset[:-2]
    frame["next_Sunrise"] = sunrise[2:]

    tmin = pd.to_numeric(frame["Tmin"], errors="coerce").to_numpy(dtype=float)
    tmax = pd.to_numeric(frame["Tmax"], errors="coerce").to_numpy(dtype=float)
    prev_max = np.concatenate([[np.nan], tmax[:-1]])
    next_min = np.concatenate([tmin[1:], [np.nan]])
    prev_min = np.concatenate([[np.nan], tmin[:-1]])
    frame["prev_max"] = prev_max
    frame["next_min"] = next_min
    frame["prev_min"] = prev_min

    sunrise_day = frame["Sunrise"].to_numpy(dtype=float)
    sunset_day = frame["Sunset"].to_numpy(dtype=float)
    day_length_day = frame["Daylength"].to_numpy(dtype=float)
    prev_sunset = frame["prev_Sunset"].to_numpy(dtype=float)
    next_sunrise = frame["next_Sunrise"].to_numpy(dtype=float)

    tsunset = tmin + (tmax - tmin) * np.sin(np.pi * (sunset_day - sunrise_day) / (day_length_day + 4))
    prev_tsunset = prev_min + (prev_max - prev_min) * np.sin(np.pi * day_length_day / (day_length_day + 4))
    frame["Tsunset"] = tsunset
    frame["prev_Tsunset"] = prev_tsunset

    n_rows = len(frame)
    no_riseset = np.isin(day_length_day, [0, 24, -99])
    for hour in range(24):
        values = np.full(n_rows, np.nan, dtype=float)
        values[no_riseset] = ((tmax + tmin) / 2)[no_riseset]

        morning = hour <= sunrise_day
        if n_rows > 0:
            morning[0] = False
        day = (hour > sunrise_day) & (hour <= sunset_day)
        evening = hour >= sunset_day
        if n_rows > 0:
            evening[-1] = False

        morning_denom = np.log(np.maximum(1, 24 - (prev_sunset[morning] - sunrise_day[morning])))
        values[morning] = prev_tsunset[morning] - (
            (prev_tsunset[morning] - tmin[morning])
            / morning_denom
            * np.log(hour + 24 - prev_sunset[morning] + 1)
        )

        values[day] = tmin[day] + (tmax[day] - tmin[day]) * np.sin(
            np.pi * (hour - sunrise_day[day]) / (day_length_day[day] + 4)
        )

        evening_denom = np.log(24 - (sunset_day[evening] - next_sunrise[evening]) + 1)
        values[evening] = tsunset[evening] - (
            (tsunset[evening] - next_min[evening])
            / evening_denom
            * np.log(hour - sunset_day[evening] + 1)
        )
        frame[f"Hour_{hour}"] = values

    hour_columns = [f"Hour_{hour}" for hour in range(24)]
    frame.loc[0, hour_columns] = frame.loc[0, hour_columns].fillna(frame.loc[0, "Tmin"])
    frame.loc[n_rows - 1, hour_columns] = frame.loc[n_rows - 1, hour_columns].fillna(frame.loc[n_rows - 1, "Tmin"])

    if keep_sunrise_sunset:
        return frame[preserve_columns + ["Sunrise", "Sunset", "Daylength"] + hour_columns]
    return frame[preserve_columns + hour_columns]


def stack_hourly_temps(
    weather: Any = None,
    *,
    latitude: float = 50,
    hour_file: Any = None,
    keep_sunrise_sunset: bool = False,
) -> dict[str, Any]:
    """Stack wide hourly temperature columns into a long hourly table.

    Translates R ``stack_hourly_temps``. Daily records with ``Tmin`` and
    ``Tmax`` are first passed through :func:`make_hourly_temps`; records already
    containing ``Hour_0`` through ``Hour_23`` are stacked directly.
    """
    if weather is None and hour_file is not None:
        weather = hour_file
    if weather is None:
        return {"hourtemps": pd.DataFrame(), "QC": np.nan}

    qc: Any = np.nan
    if isinstance(weather, dict) and "weather" in weather and "QC" in weather:
        qc = weather["QC"]
        source = weather["weather"]
        wide = make_hourly_temps(latitude, source, keep_sunrise_sunset=keep_sunrise_sunset)
    else:
        source_frame = _as_dataframe(weather, name="weather")
        hour_columns = [f"Hour_{hour}" for hour in range(24)]
        if all(column in source_frame.columns for column in hour_columns):
            wide = source_frame.copy()
        elif "Tmax" in source_frame.columns:
            wide = make_hourly_temps(latitude, source_frame, keep_sunrise_sunset=keep_sunrise_sunset)
        else:
            raise ValueError("weather must contain daily Tmin/Tmax columns or Hour_0 through Hour_23")

    hour_columns = [f"Hour_{hour}" for hour in range(24)]
    _require_columns(wide, hour_columns, name="weather")
    preserve_columns = [column for column in wide.columns if column not in hour_columns]
    pieces = []
    for hour in range(24):
        piece = wide[preserve_columns].copy()
        piece["Hour"] = float(hour)
        piece["Temp"] = pd.to_numeric(wide[f"Hour_{hour}"], errors="coerce")
        pieces.append(piece)

    long = pd.concat(pieces, ignore_index=True)
    sort_columns = [column for column in ["Year", "JDay", "Hour"] if column in long.columns]
    if sort_columns:
        long = long.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return {"hourtemps": long[preserve_columns + ["Hour", "Temp"]], "QC": qc}


def empirical_daily_temperature_curve(thourly: Any) -> pd.DataFrame:
    """Derive an empirical daily temperature curve from observed hourly data.

    The mean temperature during each hour of the day is expressed as a fraction
    of the daily temperature range (Tmax - Tmin), separately for each month.
    Translates R ``Empirical_daily_temperature_curve``.

    Parameters
    ----------
    thourly : Any
        Hourly temperatures as a DataFrame or table-like. Must contain
        'Year', 'Month', 'Day', 'Hour', and 'Temp'.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Month', 'Hour', and 'Prediction_coefficient'.
    """
    frame, _ = _prepare_hourtemps(thourly, name="thourly")
    if "Month" not in frame.columns:
        # _prepare_hourtemps ensures Year, JDay, Hour, Temp.
        # If Month is missing, we need to derive it from JDay if possible.
        # But R version explicitly requires Month column.
        _require_columns(frame, ["Month"], name="thourly")

    # Summarize sub-hourly data if present
    thours = frame.groupby(["Year", "Month", "Day", "Hour"])["Temp"].mean().reset_index()

    # Daily extremes
    tday = thours.groupby(["Year", "Month", "Day"])["Temp"].agg(Tmin="min", Tmax="max").reset_index()

    # Merge hourly and daily
    merged = pd.merge(thours, tday, on=["Year", "Month", "Day"])

    # Scale temperatures by daily range
    # Tscaled = (Temp - Tmin) / (Tmax - Tmin)
    # Avoid division by zero
    range_val = merged["Tmax"] - merged["Tmin"]
    merged["Tscaled"] = np.where(range_val != 0, (merged["Temp"] - merged["Tmin"]) / range_val, np.nan)

    # Average scaled values by Month and Hour
    scaled_summ = merged.groupby(["Month", "Hour"], as_index=False)["Tscaled"].mean()

    # Final adjustment per month
    def adjust_month(group: pd.DataFrame) -> pd.DataFrame:
        tmin = group["Tscaled"].min()
        res = group.copy()
        res["Tscale_adj"] = res["Tscaled"] - tmin
        tmax = res["Tscale_adj"].max()
        if tmax != 0:
            res["Tscale_adj"] /= tmax
        return res

    scaled_summ = scaled_summ.groupby("Month", group_keys=True).apply(adjust_month).reset_index()
    if "level_0" in scaled_summ.columns: # Sometimes reset_index adds this if it was already indexed
         scaled_summ = scaled_summ.drop(columns=["level_0"], errors="ignore")
    if "level_1" in scaled_summ.columns:
         scaled_summ = scaled_summ.drop(columns=["level_1"], errors="ignore")

    # Return as in R: Month, Hour, Prediction_coefficient
    out = scaled_summ[["Month", "Hour", "Tscale_adj"]].copy()
    out.columns = ["Month", "Hour", "Prediction_coefficient"]
    return out


def empirical_hourly_temperatures(tdaily: Any, empi_coeffs: Any) -> pd.DataFrame:
    """Generate hourly temperatures from daily extremes using empirical coefficients.

    Translates R ``Empirical_hourly_temperatures``.

    Parameters
    ----------
    tdaily : Any
        Daily temperatures as a DataFrame or table-like. Must contain
        'Year', 'Month', 'Day', 'Tmin', and 'Tmax'.
    empi_coeffs : Any
        Coefficients from ``empirical_daily_temperature_curve``.

    Returns
    -------
    pd.DataFrame
        Hourly temperatures including all columns from `tdaily`.
    """
    daily = _as_dataframe(tdaily, name="tdaily")
    _require_columns(daily, ["Year", "Month", "Day", "Tmin", "Tmax"], name="tdaily")

    coeffs = _as_dataframe(empi_coeffs, name="empi_coeffs")
    _require_columns(coeffs, ["Month", "Hour", "Prediction_coefficient"], name="empi_coeffs")

    # R uses stack_hourly_temps(tdaily, latitude=0)$hourtemps to get a template
    # latitude=0 is used in R source to get 24 hours per day regardless of actual latitude
    # because it just wants the structure.
    template = stack_hourly_temps(daily, latitude=0)["hourtemps"]

    # R: frame[,"YEARMODAHO"]<-frame$Year*1000000+frame$Month*10000+frame$Day*100+frame$Hour
    # merged<-merge(frame,empi_coeffs[,c("MonthHour","Prediction_coefficient")],by="MonthHour")
    # merged<-merged[order(merged$YEARMODAHO),]

    # Merge on Month and Hour
    merged = pd.merge(template, coeffs, on=["Month", "Hour"])

    # Calculate empirical temperature
    merged["Temp_empirical"] = merged["Tmin"] + (merged["Tmax"] - merged["Tmin"]) * merged["Prediction_coefficient"]

    # Rename Temp (which is idealized) to Temp_idealized and Temp_empirical to Temp
    merged = merged.rename(columns={"Temp": "Temp_idealized", "Temp_empirical": "Temp"})

    # Clean up and sort
    sort_cols = ["Year", "Month", "Day", "Hour"]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # R excludes MonthHour, YEARMODAHO, Temp_idealized, Temp_empirical, Prediction_coefficient
    # we just need to return the expected columns.
    # template had all daily columns + Hour + Temp
    daily_cols = [c for c in daily.columns if c not in ["Hour", "Temp"]]
    out_cols = daily_cols + ["Hour", "Temp"]
    out = merged[out_cols].copy()
    out = _add_jday_if_missing(out)
    return out


def interpolate_gaps(x: Any) -> dict[str, np.ndarray]:
    """Linearly interpolate missing values in a numeric sequence.

    Translates R ``interpolate_gaps``. Missing values are ``NaN`` or values
    that cannot be coerced to numeric. Leading and trailing gaps are filled
    with the nearest observed value and still marked as interpolated.
    """
    values = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    missing = np.isnan(values)
    if values.size == 0:
        return {"interp": values, "missing": missing}
    if missing.all():
        warnings.warn("no data in dataset! Nothing to interpolate", RuntimeWarning, stacklevel=2)
        return {"interp": values.copy(), "missing": np.ones(values.shape, dtype=bool)}

    indices = np.arange(values.size)
    observed = ~missing
    interp = values.copy()
    interp[missing] = np.interp(indices[missing], indices[observed], values[observed])
    return {"interp": interp, "missing": missing}


def _yearmoda_from_datetime(values: pd.Series) -> pd.Series:
    return values.dt.year * 10000 + values.dt.month * 100 + values.dt.day


def _solve_daily_extreme(
    frame: pd.DataFrame,
    *,
    group_col: str,
    output_col: str,
    matrix_builder: Callable[[pd.DataFrame], np.ndarray],
    minimum_values_for_solving: int,
) -> pd.Series:
    solved = pd.Series(np.nan, index=frame.index, dtype=float)
    for day_value, group in frame.groupby(group_col, dropna=True, sort=False):
        observed = group["Temp"].notna()
        if int(observed.sum()) < minimum_values_for_solving:
            continue
        design = matrix_builder(group)
        design = design[observed.to_numpy(dtype=bool)]
        keep_columns = np.nansum(design, axis=0) > 0
        design = design[:, keep_columns]
        if design.size == 0:
            continue
        response = group.loc[observed, "Temp"].to_numpy(dtype=float)
        try:
            solution = np.linalg.lstsq(design, response, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if solution.size:
            solved.loc[group.index] = float(solution[0])
    return solved.rename(output_col)


def _source_series(length: int, value: Any = None) -> pd.Series:
    return pd.Series([value] * length, dtype=object)


def _patch_report_from_legacy_stats(
    variable: str,
    stats: dict[str, pd.DataFrame] | list[pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if isinstance(stats, Mapping):
        iterable = stats.items()
    else:
        iterable = [(f"station_{idx + 1}", frame) for idx, frame in enumerate(stats)]
    for proxy, frame in iterable:
        if variable not in frame.index:
            continue
        row = frame.loc[variable]
        rows.append(
            {
                "Var": variable,
                "Proxy": proxy,
                "mean_bias": row.get("mean_bias", np.nan),
                "stdev_bias": row.get("stdev_bias", np.nan),
                "filled": row.get("filled", 0),
                "gaps_remain": row.get("gaps_remain", np.nan),
            }
        )
    return pd.DataFrame(rows)


def interpolate_gaps_hourly(
    hourtemps: Any,
    latitude: float = 50,
    daily_temps: Any = None,
    *,
    interpolate_remaining: bool = True,
    return_extremes: bool = False,
    minimum_values_for_solving: int = 5,
    runn_mean_test_length: int = 5,
    runn_mean_test_diff: float = 5,
    daily_patch_max_mean_bias: float | None = None,
    daily_patch_max_stdev_bias: float | None = None,
) -> dict[str, Any]:
    """Interpolate hourly temperature gaps using idealized daily curves.

    Translates R ``interpolate_gaps_hourly``. The method first solves daily
    temperature extremes from observed hourly temperatures and an idealized
    Linvill curve, optionally patches unresolved daily extremes from proxy
    daily data, then interpolates deviations from the ideal curve.
    """
    if minimum_values_for_solving < 2:
        raise ValueError("minimum_values_for_solving must be at least 2")

    if isinstance(hourtemps, Mapping) and "hourtemps" in hourtemps:
        hs = _as_dataframe(hourtemps["hourtemps"], name="hourtemps")
    else:
        hs = _as_dataframe(hourtemps, name="hourtemps")
    _require_columns(hs, ["Year", "Month", "Day", "Hour", "Temp"], name="hourtemps")

    hs = _coerce_weather_dates(hs, hourly=True)
    hs["Temp"] = pd.to_numeric(hs["Temp"], errors="coerce")
    if hs.empty:
        empty = pd.DataFrame()
        return {"weather": empty, "daily_patch_report": pd.DataFrame()}
    if hs[["Year", "Month", "Day", "Hour"]].isna().any(axis=None):
        raise ValueError("hourtemps contains invalid Year, Month, Day, or Hour values")
    if ((hs["Hour"] < 0) | (hs["Hour"] > 23) | (hs["Hour"] % 1 != 0)).any():
        raise ValueError("Hour values must be integers from 0 through 23")
    input_dates = pd.to_datetime({"year": hs["Year"], "month": hs["Month"], "day": hs["Day"]}, errors="coerce")
    if input_dates.isna().any():
        raise ValueError("hourtemps contains invalid Year, Month, or Day values")

    hs = make_all_day_table(hs, timestep="hour", input_timestep="hour")
    hs["Temp"] = pd.to_numeric(hs["Temp"], errors="coerce")
    hs["YEARMODA"] = hs["Year"] * 10000 + hs["Month"] * 100 + hs["Day"]
    hs["YEARMODAHO"] = hs["YEARMODA"] * 100 + hs["Hour"]

    daily_extremes = make_all_day_table(hs[["Year", "Month", "Day"]].iloc[[0, -1]], no_variable_check=True)
    daily_extremes["Tmin"] = 0.0
    daily_extremes["Tmax"] = 1.0
    ideal_temps = stack_hourly_temps(
        daily_extremes,
        latitude=latitude,
        keep_sunrise_sunset=True,
    )["hourtemps"]
    ideal_temps["YEARMODAHO"] = (
        ideal_temps["Year"] * 1_000_000
        + ideal_temps["Month"] * 10_000
        + ideal_temps["Day"] * 100
        + ideal_temps["Hour"]
    )
    ideal_temps = ideal_temps[["YEARMODAHO", "Sunrise", "Sunset", "Daylength", "Temp"]].rename(
        columns={"Temp": "ideal_temp"}
    )

    hs = hs.merge(ideal_temps, on="YEARMODAHO", how="inner").sort_values("YEARMODAHO").reset_index(drop=True)
    date_values = pd.to_datetime({"year": hs["Year"], "month": hs["Month"], "day": hs["Day"]}, errors="coerce")
    hs["BeforeDay"] = _yearmoda_from_datetime(date_values - pd.Timedelta(days=1)).astype(float)
    hs["AfterDay"] = _yearmoda_from_datetime(date_values + pd.Timedelta(days=1)).astype(float)
    hs["TminDay"] = np.nan
    hs["TmaxDay"] = np.nan

    before_sunrise = hs["Hour"] < hs["Sunrise"]
    after_sunset = hs["Hour"] > hs["Sunset"]
    hs.loc[before_sunrise, "TminDay"] = hs.loc[before_sunrise, "YEARMODA"]
    hs.loc[before_sunrise, "TmaxDay"] = hs.loc[before_sunrise, "BeforeDay"]
    hs.loc[after_sunset, "TminDay"] = hs.loc[after_sunset, "AfterDay"]
    hs.loc[after_sunset, "TmaxDay"] = hs.loc[after_sunset, "YEARMODA"]
    hs["TminDay"] = hs["TminDay"].fillna(hs["YEARMODA"])
    hs["TmaxDay"] = hs["TmaxDay"].fillna(hs["YEARMODA"])

    hs["Tmin_solved"] = _solve_daily_extreme(
        hs,
        group_col="TminDay",
        output_col="Tmin_solved",
        minimum_values_for_solving=int(minimum_values_for_solving),
        matrix_builder=lambda group: np.column_stack(
            [
                1.0 - group["ideal_temp"].to_numpy(dtype=float),
                (group["TmaxDay"].to_numpy(dtype=float) < float(group["TminDay"].iloc[0]))
                * group["ideal_temp"].to_numpy(dtype=float),
                (group["TmaxDay"].to_numpy(dtype=float) >= float(group["TminDay"].iloc[0]))
                * group["ideal_temp"].to_numpy(dtype=float),
            ]
        ),
    )
    hs["Tmax_solved"] = _solve_daily_extreme(
        hs,
        group_col="TmaxDay",
        output_col="Tmax_solved",
        minimum_values_for_solving=int(minimum_values_for_solving),
        matrix_builder=lambda group: np.column_stack(
            [
                group["ideal_temp"].to_numpy(dtype=float),
                (group["TminDay"].to_numpy(dtype=float) == float(group["TmaxDay"].iloc[0]))
                * (1.0 - group["ideal_temp"].to_numpy(dtype=float)),
                (group["TminDay"].to_numpy(dtype=float) > float(group["TmaxDay"].iloc[0]))
                * (1.0 - group["ideal_temp"].to_numpy(dtype=float)),
            ]
        ),
    )

    allmins = make_all_day_table(
        hs[["TminDay", "Tmin_solved"]].rename(columns={"TminDay": "YEARMODA", "Tmin_solved": "Tmin"}),
        no_variable_check=True,
    )
    allmaxs = make_all_day_table(
        hs[["TmaxDay", "Tmax_solved"]].rename(columns={"TmaxDay": "YEARMODA", "Tmax_solved": "Tmax"}),
        no_variable_check=True,
    )

    if "Tmin" in allmins:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            running = runn_mean(
                allmins["Tmin"].to_numpy(dtype=float),
                int(runn_mean_test_length),
                na_rm=True,
                exclude_central_value=True,
            )
        runn_diff = np.abs(allmins["Tmin"].to_numpy(dtype=float) - running)
        allmins.loc[runn_diff > float(runn_mean_test_diff), "Tmin"] = np.nan
    if "Tmax" in allmaxs:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            running = runn_mean(
                allmaxs["Tmax"].to_numpy(dtype=float),
                int(runn_mean_test_length),
                na_rm=True,
                exclude_central_value=True,
            )
        runn_diff = np.abs(allmaxs["Tmax"].to_numpy(dtype=float) - running)
        allmaxs.loc[runn_diff > float(runn_mean_test_diff), "Tmax"] = np.nan

    allmins["Tmin_source"] = _source_series(len(allmins))
    allmins.loc[allmins["Tmin"].notna(), "Tmin_source"] = "solved"
    allmaxs["Tmax_source"] = _source_series(len(allmaxs))
    allmaxs.loc[allmaxs["Tmax"].notna(), "Tmax_source"] = "solved"

    patch_report = pd.DataFrame(
        [
            {
                "Var": "Tmin",
                "Proxy": "solved",
                "mean_bias": np.nan,
                "stdev_bias": np.nan,
                "filled": int(allmins["Tmin"].notna().sum()),
                "gaps_remain": int(allmins["Tmin"].isna().sum()),
            },
            {
                "Var": "Tmax",
                "Proxy": "solved",
                "mean_bias": np.nan,
                "stdev_bias": np.nan,
                "filled": int(allmaxs["Tmax"].notna().sum()),
                "gaps_remain": int(allmaxs["Tmax"].isna().sum()),
            },
        ]
    )

    if daily_temps is not None:
        min_patched = patch_daily_temperatures(
            allmins.drop(columns=["Tmin_source"]),
            daily_temps,
            vars=("Tmin",),
            max_mean_bias=daily_patch_max_mean_bias,
            max_stdev_bias=daily_patch_max_stdev_bias,
        )
        allmins = min_patched["weather"]
        if "Tmin_source" not in allmins:
            allmins["Tmin_source"] = _source_series(len(allmins))
        allmins.loc[allmins["Tmin"].notna() & allmins["Tmin_source"].isna(), "Tmin_source"] = "solved"
        patch_report = pd.concat(
            [patch_report.iloc[[0]], _patch_report_from_legacy_stats("Tmin", min_patched["statistics"]), patch_report.iloc[[1]]],
            ignore_index=True,
        )

        max_patched = patch_daily_temperatures(
            allmaxs.drop(columns=["Tmax_source"]),
            daily_temps,
            vars=("Tmax",),
            max_mean_bias=daily_patch_max_mean_bias,
            max_stdev_bias=daily_patch_max_stdev_bias,
        )
        allmaxs = max_patched["weather"]
        if "Tmax_source" not in allmaxs:
            allmaxs["Tmax_source"] = _source_series(len(allmaxs))
        allmaxs.loc[allmaxs["Tmax"].notna() & allmaxs["Tmax_source"].isna(), "Tmax_source"] = "solved"
        patch_report = pd.concat(
            [patch_report, _patch_report_from_legacy_stats("Tmax", max_patched["statistics"])],
            ignore_index=True,
        )

    if interpolate_remaining:
        min_interp = interpolate_gaps(allmins["Tmin"])
        allmins["Tmin"] = min_interp["interp"]
        min_source_missing = allmins["Tmin"].notna() & allmins["Tmin_source"].isna()
        allmins.loc[min_source_missing, "Tmin_source"] = "interpolated"
        min_interpolated = int((allmins["Tmin_source"] == "interpolated").sum())
        patch_report = pd.concat(
            [
                patch_report.loc[patch_report["Var"] == "Tmin"],
                pd.DataFrame(
                    [
                        {
                            "Var": "Tmin",
                            "Proxy": "interpolated",
                            "mean_bias": np.nan,
                            "stdev_bias": np.nan,
                            "filled": min_interpolated,
                            "gaps_remain": int(allmins["Tmin"].isna().sum()),
                        }
                    ]
                ),
                patch_report.loc[patch_report["Var"] == "Tmax"],
            ],
            ignore_index=True,
        )

        max_interp = interpolate_gaps(allmaxs["Tmax"])
        allmaxs["Tmax"] = max_interp["interp"]
        max_source_missing = allmaxs["Tmax"].notna() & allmaxs["Tmax_source"].isna()
        allmaxs.loc[max_source_missing, "Tmax_source"] = "interpolated"
        max_interpolated = int((allmaxs["Tmax_source"] == "interpolated").sum())
        patch_report = pd.concat(
            [
                patch_report,
                pd.DataFrame(
                    [
                        {
                            "Var": "Tmax",
                            "Proxy": "interpolated",
                            "mean_bias": np.nan,
                            "stdev_bias": np.nan,
                            "filled": max_interpolated,
                            "gaps_remain": int(allmaxs["Tmax"].isna().sum()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    allmins = allmins.rename(columns={"YEARMODA": "TminDay"})
    allmaxs = allmaxs.rename(columns={"YEARMODA": "TmaxDay"})
    hs_all = hs.drop(columns=[column for column in ["Tmin", "Tmax"] if column in hs.columns])
    hs_all = hs_all.merge(allmins[["TminDay", "Tmin", "Tmin_source"]], on="TminDay", how="inner")
    hs_all = hs_all.merge(allmaxs[["TmaxDay", "Tmax", "Tmax_source"]], on="TmaxDay", how="inner")
    hs_all = hs_all.drop(columns=[column for column in ["Tmin_solved", "Tmax_solved", "BeforeDay", "AfterDay"] if column in hs_all])

    hs_all["Temp_idealized"] = hs_all["Tmin"] + hs_all["ideal_temp"] * (hs_all["Tmax"] - hs_all["Tmin"])
    hs_all["deviation"] = hs_all["Temp"] - hs_all["Temp_idealized"]
    hs_all["deviation"] = interpolate_gaps(hs_all["deviation"])["interp"]
    hs_all["Temp_interp"] = hs_all["Temp_idealized"] + hs_all["deviation"]
    hs_all = hs_all.rename(columns={"Temp": "Temp_measured", "Temp_interp": "Temp"})
    measured = hs_all["Temp_measured"].notna()
    hs_all.loc[measured, "Tmin_source"] = None
    hs_all.loc[measured, "Tmax_source"] = None

    drop_columns = [
        "deviation",
        "Temp_idealized",
        "TmaxDay",
        "TminDay",
        "ideal_temp",
        "Sunrise",
        "Sunset",
        "Daylength",
        "YEARMODA",
        "YEARMODAHO",
        "DATE",
    ]
    if not return_extremes:
        drop_columns.extend(["Tmin", "Tmax"])
    hs_all = hs_all.drop(columns=[column for column in drop_columns if column in hs_all.columns])
    hs_all = hs_all.sort_values(["Year", "Month", "Day", "Hour"], kind="mergesort").reset_index(drop=True)
    patch_report["filled"] = pd.to_numeric(patch_report["filled"], errors="coerce").fillna(0)
    return {"weather": hs_all, "daily_patch_report": patch_report.reset_index(drop=True)}


def _coerce_weather_dates(frame: pd.DataFrame, *, hourly: bool) -> pd.DataFrame:
    out = frame.copy()
    rename = {column: column.title() for column in out.columns if column in {"YEAR", "MONTH", "DAY"}}
    out = out.rename(columns=rename)

    if hourly and "YEARMODAHO" in out.columns and not {"Year", "Month", "Day", "Hour"}.issubset(out.columns):
        value = pd.to_numeric(out["YEARMODAHO"], errors="coerce")
        out["Year"] = np.floor(value / 1_000_000)
        out["Month"] = np.floor((value % 1_000_000) / 10_000)
        out["Day"] = np.floor((value % 10_000) / 100)
        out["Hour"] = value % 100
    elif "YEARMODA" in out.columns and not {"Year", "Month", "Day"}.issubset(out.columns):
        value = pd.to_numeric(out["YEARMODA"], errors="coerce")
        out["Year"] = np.floor(value / 10_000)
        out["Month"] = np.floor((value % 10_000) / 100)
        out["Day"] = value % 100

    required = ["Year", "Month", "Day"] + (["Hour"] if hourly else [])
    _require_columns(out, required, name="tab")
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _mean_numeric_first(values: pd.Series) -> Any:
    if pd.api.types.is_numeric_dtype(values):
        return values.mean()
    non_missing = values.dropna()
    return non_missing.iloc[0] if not non_missing.empty else np.nan


def _complete_time_table(
    frame: pd.DataFrame,
    *,
    hourly: bool,
    add_date: bool,
    tz: str,
    outcols: list[str],
) -> pd.DataFrame:
    time_col = "DATE"
    if hourly:
        times = pd.to_datetime(
            {
                "year": frame["Year"],
                "month": frame["Month"],
                "day": frame["Day"],
                "hour": frame["Hour"],
            },
            errors="coerce",
        )
        freq = "h"
    else:
        times = pd.to_datetime(
            {"year": frame["Year"], "month": frame["Month"], "day": frame["Day"]},
            errors="coerce",
        )
        freq = "D"

    work = frame.loc[times.notna()].copy()
    work["_alltime"] = times[times.notna()].to_numpy()
    if work.empty:
        return pd.DataFrame(columns=(["DATE"] if add_date else []) + outcols)

    grouped = work.groupby("_alltime", sort=True, dropna=False).agg(_mean_numeric_first)
    datevec = pd.date_range(grouped.index.min(), grouped.index.max(), freq=freq)
    complete = pd.DataFrame({time_col: datevec})
    complete["Year"] = complete[time_col].dt.year.astype(float)
    complete["Month"] = complete[time_col].dt.month.astype(float)
    complete["Day"] = complete[time_col].dt.day.astype(float)
    if hourly:
        complete["Hour"] = complete[time_col].dt.hour.astype(float)

    payload_columns = [column for column in grouped.columns if column not in ["DATE", "Year", "Month", "Day", "Hour"]]
    merged = complete.merge(grouped[payload_columns], left_on=time_col, right_index=True, how="left")

    if "YEARMODA" in outcols:
        merged["YEARMODA"] = merged["Year"] * 10000 + merged["Month"] * 100 + merged["Day"]
    if hourly and "YEARMODAHO" in outcols:
        merged["YEARMODAHO"] = (
            merged["Year"] * 1_000_000 + merged["Month"] * 10_000 + merged["Day"] * 100 + merged["Hour"]
        )

    final_columns = (["DATE"] if add_date else []) + [column for column in outcols if column in merged.columns]
    result = merged[final_columns].copy()
    if add_date and tz.upper() != "GMT":
        # pandas stores timezone-naive dates here; `tz` is accepted for API parity.
        pass
    return result


def _aggregate_hourly_to_daily(
    frame: pd.DataFrame,
    *,
    aggregation_hours: Sequence[int] | Mapping[str, int] | None,
) -> pd.DataFrame:
    work = frame.copy()
    work["DATE"] = pd.to_datetime({"year": work["Year"], "month": work["Month"], "day": work["Day"]}, errors="coerce")
    work = work.loc[work["DATE"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["Year", "Month", "Day", "Tmin", "Tmean", "Tmax"])

    numeric_cols = [column for column in work.columns if pd.api.types.is_numeric_dtype(work[column])]
    grouped = work.groupby("DATE", sort=True)
    means = grouped[numeric_cols].mean().reset_index()
    if "Temp" in means.columns:
        means = means.rename(columns={"Temp": "Tmean"})
    else:
        means["Tmean"] = np.nan

    if aggregation_hours is None:
        extremes = grouped["Temp"].agg(Tmin="min", Tmax="max").reset_index()
    else:
        if isinstance(aggregation_hours, Mapping):
            min_hours = int(aggregation_hours["min_hours"])
            max_hours = int(aggregation_hours["max_hours"])
            hours_needed = int(aggregation_hours["hours_needed"])
        else:
            if len(aggregation_hours) != 3:
                raise ValueError("aggregation_hours must contain min_hours, max_hours, and hours_needed")
            min_hours, max_hours, hours_needed = [int(value) for value in aggregation_hours]

        mean_hour_temps = work.groupby(["Month", "Hour"], sort=True)["Temp"].mean().reset_index()
        month_min_hours: dict[int, set[float]] = {}
        month_max_hours: dict[int, set[float]] = {}
        for month in range(1, 13):
            month_hours = mean_hour_temps.loc[mean_hour_temps["Month"] == month]
            month_min_hours[month] = set(month_hours.nsmallest(min_hours, "Temp")["Hour"])
            month_max_hours[month] = set(month_hours.nlargest(max_hours, "Temp")["Hour"])

        rows: list[dict[str, Any]] = []
        for day, chunk in grouped:
            month = int(chunk["Month"].mean())
            observed = chunk.loc[chunk["Temp"].notna()]
            min_observed = observed.loc[observed["Hour"].isin(month_min_hours.get(month, set()))]
            max_observed = observed.loc[observed["Hour"].isin(month_max_hours.get(month, set()))]
            rows.append(
                {
                    "DATE": day,
                    "Tmin": min_observed["Temp"].min() if len(min_observed) >= hours_needed else np.nan,
                    "Tmax": max_observed["Temp"].max() if len(max_observed) >= hours_needed else np.nan,
                }
            )
        extremes = pd.DataFrame(rows)

    daily = means.merge(extremes, on="DATE", how="left")
    daily["Year"] = daily["DATE"].dt.year.astype(float)
    daily["Month"] = daily["DATE"].dt.month.astype(float)
    daily["Day"] = daily["DATE"].dt.day.astype(float)
    return daily


def make_all_day_table(
    tab: Any,
    timestep: str = "day",
    input_timestep: str | None = None,
    *,
    tz: str = "GMT",
    add_date: bool = True,
    no_variable_check: bool = False,
    aggregation_hours: Sequence[int] | Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Fill missing daily or hourly rows in a chillR-style time series.

    Translates the deterministic table-completion behavior of R
    ``make_all_day_table``. Hourly input can also be aggregated to daily
    ``Tmin``, ``Tmean``, and ``Tmax`` records.
    """
    del no_variable_check
    if input_timestep is None:
        input_timestep = timestep
    if timestep not in {"day", "hour"}:
        raise ValueError("timestep must be 'day' or 'hour'")
    if input_timestep not in {"day", "hour"}:
        raise ValueError("input_timestep must be 'day' or 'hour'")
    if timestep == "hour" and input_timestep == "day":
        raise ValueError("converting daily to hourly data is not supported by make_all_day_table")

    source = _as_dataframe(tab, name="tab")
    hourly_input = input_timestep == "hour"
    source = _coerce_weather_dates(source, hourly=hourly_input)
    source = source.loc[source[["Year", "Month", "Day"]].notna().all(axis=1)].copy()
    if hourly_input:
        source = source.loc[source["Hour"].notna()].copy()

    if timestep == "day" and input_timestep == "hour":
        daily = _aggregate_hourly_to_daily(source, aggregation_hours=aggregation_hours)
        outcols = [
            column
            for column in source.columns
            if column not in {"Tmin", "Tmean", "Tmax", "Temp", "Hour", "DATE"}
        ] + ["Tmin", "Tmean", "Tmax"]
        return _complete_time_table(daily, hourly=False, add_date=add_date, tz=tz, outcols=outcols)

    outcols = list(source.columns)
    return _complete_time_table(source, hourly=timestep == "hour", add_date=add_date, tz=tz, outcols=outcols)


def filter_temperatures(
    temp_file: Any,
    remove_value: float | None = None,
    running_mean_filter: float | None = None,
    running_mean_length: int = 3,
    min_extreme: float | None = None,
    max_extreme: float | None = None,
    max_missing_in_window: float = 1,
    missing_window_size: int = 9,
) -> pd.DataFrame:
    """Remove implausible or isolated temperature readings.

    Translates R ``filter_temperatures``. Missing rows are first made explicit
    with :func:`make_all_day_table`; filters then set suspect ``Temp`` values to
    ``nan`` without dropping rows.
    """
    frame = _as_dataframe(temp_file, name="temp_file")
    _require_columns(frame, ["Year", "Month", "Day", "Temp"], name="temp_file")
    original_columns = set(frame.columns)
    timestep = "hour" if "Hour" in frame.columns else "day"
    alldays = make_all_day_table(frame, timestep=timestep, input_timestep=timestep)
    if "Minutes" in frame.columns:
        alldays["Minutes"] = 0

    alldays["Temp"] = pd.to_numeric(alldays["Temp"], errors="coerce")
    if running_mean_filter is not None and not pd.isna(running_mean_filter):
        smoothed = runn_mean(
            alldays["Temp"].to_numpy(dtype=float),
            int(running_mean_length),
            na_rm=True,
            exclude_central_value=True,
        )
        runn_diff = np.abs(alldays["Temp"].to_numpy(dtype=float) - smoothed)
        alldays.loc[runn_diff > float(running_mean_filter), "Temp"] = np.nan

    if remove_value is not None and not pd.isna(remove_value):
        alldays.loc[alldays["Temp"] == float(remove_value), "Temp"] = np.nan
    if min_extreme is not None and not pd.isna(min_extreme):
        alldays.loc[alldays["Temp"] < float(min_extreme), "Temp"] = np.nan
    if max_extreme is not None and not pd.isna(max_extreme):
        alldays.loc[alldays["Temp"] > float(max_extreme), "Temp"] = np.nan

    if max_missing_in_window < 1:
        missing_share = runn_mean(
            alldays["Temp"].to_numpy(dtype=float),
            int(missing_window_size),
            exclude_central_value=True,
            fun=lambda values: float(np.isnan(values).sum() / len(values)) if len(values) else np.nan,
        )
        alldays.loc[missing_share > float(max_missing_in_window), "Temp"] = np.nan

    drop_if_generated = {"YEARMODA", "YEARMODAHO", "DATE"} - original_columns
    return alldays.drop(columns=[column for column in drop_if_generated if column in alldays.columns])


def _as_named_weather_list(patch_weather: Any) -> list[tuple[str | None, Any]]:
    if isinstance(patch_weather, pd.DataFrame):
        return [(None, patch_weather)]
    if isinstance(patch_weather, Mapping):
        return list(patch_weather.items())
    return [(None, value) for value in list(patch_weather)]


def patch_daily_temperatures(
    weather: Any,
    patch_weather: Any,
    vars: Sequence[str] = ("Tmin", "Tmax"),
    max_mean_bias: float | None = None,
    max_stdev_bias: float | None = None,
) -> dict[str, Any]:
    """Patch gaps in daily weather records from auxiliary daily records.

    Translates deprecated R ``patch_daily_temperatures``. Bias is computed as
    ``weather - auxiliary`` over overlapping records and added to auxiliary
    values used to fill missing values.
    """
    variable_names = list(vars)
    base = make_all_day_table(weather, no_variable_check=True)
    for variable in variable_names:
        if variable not in base.columns:
            base[variable] = np.nan
    if "YEARMODA" not in base.columns:
        base["YEARMODA"] = base["Year"] * 10000 + base["Month"] * 100 + base["Day"]

    daily = _as_named_weather_list(patch_weather)
    statistics: list[pd.DataFrame] = []
    for _name, _record in daily:
        statistics.append(
            pd.DataFrame(
                {
                    "mean_bias": np.nan,
                    "stdev_bias": np.nan,
                    "filled": np.nan,
                    "gaps_remain": np.nan,
                },
                index=variable_names,
            )
        )

    aux_weather = base.copy()
    gaps = int(aux_weather[variable_names].isna().sum().sum())
    for idx, (name, record) in enumerate(daily):
        if gaps == 0:
            break
        auxiliary = make_all_day_table(record, no_variable_check=True)
        for variable in variable_names:
            if variable not in auxiliary.columns:
                auxiliary[variable] = np.nan
        auxiliary["YEARMODA"] = auxiliary["Year"] * 10000 + auxiliary["Month"] * 100 + auxiliary["Day"]
        temp_names = [f"{variable}temp" for variable in variable_names]
        auxiliary = auxiliary[["YEARMODA", *variable_names]].rename(
            columns=dict(zip(variable_names, temp_names, strict=False))
        )
        aux_weather = aux_weather.drop(columns=[column for column in temp_names if column in aux_weather.columns])
        aux_weather = aux_weather.merge(auxiliary, on="YEARMODA", how="left")

        for variable, temp_name in zip(variable_names, temp_names, strict=False):
            diff = pd.to_numeric(aux_weather[variable], errors="coerce") - pd.to_numeric(
                aux_weather[temp_name], errors="coerce"
            )
            bias = float(diff.mean(skipna=True)) if diff.notna().any() else np.nan
            stdev = float(diff.std(skipna=True, ddof=1)) if diff.notna().sum() > 1 else np.nan
            statistics[idx].loc[variable, "mean_bias"] = round(bias, 3) if not np.isnan(bias) else np.nan
            statistics[idx].loc[variable, "stdev_bias"] = round(stdev, 3) if not np.isnan(stdev) else np.nan

            dont_use = np.isnan(bias)
            if not dont_use:
                if max_mean_bias is not None and not pd.isna(max_mean_bias) and abs(bias) > float(max_mean_bias):
                    dont_use = True
                if max_stdev_bias is not None and not pd.isna(max_stdev_bias):
                    if np.isnan(stdev) or stdev > float(max_stdev_bias):
                        dont_use = True
                if idx > 0 and statistics[idx - 1].loc[variable, "gaps_remain"] == 0:
                    dont_use = True

            fill_mask = aux_weather[variable].isna() & aux_weather[temp_name].notna()
            if dont_use:
                statistics[idx].loc[variable, "filled"] = 0
            else:
                source_col = f"{variable}_source"
                if source_col not in aux_weather.columns:
                    aux_weather[source_col] = pd.Series([None] * len(aux_weather), dtype=object)
                elif aux_weather[source_col].dtype != object:
                    aux_weather[source_col] = aux_weather[source_col].astype(object)
                source = f"daily_{name}" if name is not None else f"daily_records{idx + 1}"
                aux_weather.loc[fill_mask, source_col] = source
                statistics[idx].loc[variable, "filled"] = int(fill_mask.sum())
                aux_weather.loc[fill_mask, variable] = aux_weather.loc[fill_mask, temp_name] + bias

            statistics[idx].loc[variable, "gaps_remain"] = int(aux_weather[variable].isna().sum())

        aux_weather = aux_weather.drop(columns=temp_names)
        gaps = int(aux_weather[variable_names].isna().sum().sum())

    named_stats: dict[str, pd.DataFrame] | list[pd.DataFrame]
    names = [name for name, _record in daily]
    if all(name is not None for name in names):
        named_stats = {str(name): stat for name, stat in zip(names, statistics, strict=False)}
    else:
        named_stats = statistics
    return {"weather": aux_weather, "statistics": named_stats}


def _parse_patch_interval(time_interval: str) -> pd.DateOffset:
    normalized = str(time_interval).strip().lower()
    parts = normalized.split()
    if len(parts) == 1:
        count = 1
        unit = parts[0]
    elif len(parts) == 2 and parts[0].isdigit():
        count = int(parts[0])
        unit = parts[1]
    else:
        raise ValueError("time_interval must look like 'month', 'week', or '2 weeks'")

    unit = unit.rstrip("s")
    if count < 1:
        raise ValueError("time_interval count must be positive")
    if unit == "month":
        return pd.offsets.MonthEnd(count)
    if unit == "week":
        return pd.DateOffset(weeks=count)
    if unit == "day":
        return pd.DateOffset(days=count)
    raise ValueError("time_interval must use day(s), week(s), or month(s)")


def _assign_patch_intervals(dates: pd.Series, time_interval: str) -> pd.Series:
    offset = _parse_patch_interval(time_interval)
    intervals = pd.Series(np.nan, index=dates.index, dtype=float)
    for year in sorted(dates.dt.year.dropna().astype(int).unique()):
        year_mask = dates.dt.year == year
        start = pd.Timestamp(year=year - 1, month=12, day=31)
        end = pd.Timestamp(year=year, month=12, day=31) + offset
        boundaries: list[pd.Timestamp] = []
        current = start
        while current < end:
            current = current + offset
            boundaries.append(current)
        if not boundaries:
            continue

        boundary_values = pd.Series(boundaries)
        intervals.loc[year_mask] = dates.loc[year_mask].apply(
            lambda value: float(np.searchsorted(boundary_values.to_numpy(), value.to_datetime64(), side="left") + 1)
        )
    return intervals


def patch_daily_temps(
    weather: Any,
    patch_weather: Any,
    vars: Sequence[str] = ("Tmin", "Tmax"),
    max_mean_bias: float | None = None,
    max_stdev_bias: float | None = None,
    time_interval: str = "month",
) -> dict[str, Any]:
    """Patch daily weather gaps with interval-specific proxy bias correction.

    Translates R ``patch_daily_temps``. Bias and standard deviation are
    evaluated independently for each variable, proxy record, and calendar
    interval. Missing values are filled as ``auxiliary - mean_bias`` when the
    proxy passes the configured thresholds.
    """
    variable_names = list(vars)
    if not variable_names:
        raise ValueError("vars must contain at least one column name")

    base = make_all_day_table(weather, no_variable_check=True)
    for variable in variable_names:
        if variable not in base.columns:
            base[variable] = np.nan
    if "YEARMODA" not in base.columns:
        base["YEARMODA"] = base["Year"] * 10000 + base["Month"] * 100 + base["Day"]

    daily = _as_named_weather_list(patch_weather)
    if not daily:
        return {"weather": base, "statistics": {variable: {} for variable in variable_names}}
    if any(name is None for name, _record in daily):
        daily = [(name or f"station_{idx + 1}", record) for idx, (name, record) in enumerate(daily)]

    source_columns = [f"{variable}_source" for variable in variable_names]
    for variable, source_col in zip(variable_names, source_columns, strict=False):
        if source_col not in base.columns:
            base[source_col] = pd.Series([None] * len(base), dtype=object)
        else:
            base[source_col] = base[source_col].astype(object)
        base.loc[base[variable].notna(), source_col] = "original"

    weather_columns = list(base.columns)
    statistics: dict[str, dict[str, pd.DataFrame]] = {variable: {} for variable in variable_names}
    aux_weather = base.copy()
    gaps = int(aux_weather[variable_names].isna().sum().sum())

    for station_name, record in daily:
        if gaps == 0:
            break

        auxiliary = make_all_day_table(record, no_variable_check=True)
        for variable in variable_names:
            if variable not in auxiliary.columns:
                auxiliary[variable] = np.nan
        auxiliary["YEARMODA"] = auxiliary["Year"] * 10000 + auxiliary["Month"] * 100 + auxiliary["Day"]
        temp_names = [f"{variable}temp" for variable in variable_names]
        auxiliary = auxiliary[["YEARMODA", *variable_names]].rename(
            columns=dict(zip(variable_names, temp_names, strict=False))
        )

        aux_weather = aux_weather[weather_columns].merge(auxiliary, on="YEARMODA", how="left")
        dates = pd.to_datetime(
            {"year": aux_weather["Year"], "month": aux_weather["Month"], "day": aux_weather["Day"]},
            errors="coerce",
        )
        aux_weather["Date"] = dates
        aux_weather["interval"] = _assign_patch_intervals(dates, time_interval)

        interval_values = [int(value) for value in pd.unique(aux_weather["interval"].dropna())]
        for variable in variable_names:
            statistics[variable][str(station_name)] = pd.DataFrame(
                {
                    "Interval": interval_values,
                    "Total_days": np.nan,
                    "Overlap_days": np.nan,
                    "Mean_bias": np.nan,
                    "Stdev_bias": np.nan,
                    "Gaps_before": np.nan,
                    "Filled": np.nan,
                    "Gaps_remain": np.nan,
                }
            )

        if len(interval_values) > 2:
            counts = aux_weather.groupby(["Year", "interval"], dropna=True).size().unstack(fill_value=0)
            if counts.shape[1] > 2:
                last_median = float(np.median(counts.iloc[:, -1]))
                middle_median = float(np.median(counts.iloc[:, 1:-1].to_numpy()))
                if middle_median and last_median < 0.5 * middle_median:
                    warnings.warn(
                        "The number of days in the last interval is often a lot smaller than in other intervals. "
                        "Consider changing the time interval.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        for variable, temp_name in zip(variable_names, temp_names, strict=False):
            stats = statistics[variable][str(station_name)]
            for row_idx, interval in enumerate(interval_values):
                interval_mask = aux_weather["interval"] == interval
                interval_frame = aux_weather.loc[interval_mask]
                overlap = interval_frame[variable].notna() & interval_frame[temp_name].notna()
                diff = pd.to_numeric(interval_frame[temp_name], errors="coerce") - pd.to_numeric(
                    interval_frame[variable], errors="coerce"
                )
                bias = float(diff[overlap].mean(skipna=True)) if overlap.any() else np.nan
                stdev = float(diff[overlap].std(skipna=True, ddof=1)) if overlap.sum() > 1 else np.nan
                gaps_before = int(interval_frame[variable].isna().sum())

                stats.loc[row_idx, "Total_days"] = int(interval_mask.sum())
                stats.loc[row_idx, "Overlap_days"] = int(overlap.sum())
                stats.loc[row_idx, "Mean_bias"] = round(bias, 3) if not np.isnan(bias) else np.nan
                stats.loc[row_idx, "Stdev_bias"] = round(stdev, 3) if not np.isnan(stdev) else np.nan
                stats.loc[row_idx, "Gaps_before"] = gaps_before

                use_data = not (np.isnan(bias) or np.isnan(stdev))
                if use_data and max_mean_bias is not None and not pd.isna(max_mean_bias):
                    use_data = abs(bias) <= float(max_mean_bias)
                if use_data and max_stdev_bias is not None and not pd.isna(max_stdev_bias):
                    use_data = stdev <= float(max_stdev_bias)
                if use_data and gaps_before == 0:
                    use_data = False

                if not use_data:
                    stats.loc[row_idx, "Filled"] = 0
                else:
                    replace_mask = interval_mask & aux_weather[variable].isna() & aux_weather[temp_name].notna()
                    aux_weather.loc[replace_mask, f"{variable}_source"] = str(station_name)
                    stats.loc[row_idx, "Filled"] = int(replace_mask.sum())
                    aux_weather.loc[replace_mask, variable] = aux_weather.loc[replace_mask, temp_name] - bias

                stats.loc[row_idx, "Gaps_remain"] = int(aux_weather.loc[interval_mask, variable].isna().sum())

        aux_weather = aux_weather[weather_columns]
        gaps = int(aux_weather[variable_names].isna().sum().sum())

    out_weather = aux_weather.sort_values("YEARMODA", kind="mergesort").reset_index(drop=True)
    return {"weather": out_weather, "statistics": statistics}


tempResponse = temp_response
tempResponse_daily_list = temp_response_daily_list
tempResponse_hourtable = temp_response_hourtable
Empirical_daily_temperature_curve = empirical_daily_temperature_curve
Empirical_hourly_temperatures = empirical_hourly_temperatures
