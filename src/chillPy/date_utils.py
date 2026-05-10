"""Date and day-of-year helpers mapped from exported chillR date functions."""

from __future__ import annotations

from datetime import date, datetime
from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
import pandas as pd


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes, date, datetime, pd.Timestamp)) or np.isscalar(value)


def _as_series(value: Any, *, name: str | None = None) -> tuple[pd.Series, bool]:
    if _is_scalar(value):
        return pd.Series([value], name=name), True
    return pd.Series(value, name=name), False


def _finish_array(values: pd.Series | np.ndarray, *, scalar: bool) -> Any:
    array = np.asarray(values)
    if scalar:
        item = array[0]
        if pd.isna(item):
            return np.nan
        if isinstance(item, np.generic):
            return item.item()
        return item
    return array


def _validate_jday(value: Any, *, name: str) -> np.ndarray:
    values, _ = _as_series(value, name=name)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or ((numeric < 1) | (numeric > 366) | (numeric % 1 != 0)).any():
        raise ValueError(f"{name} is not a Julian date")
    return numeric.astype(int).to_numpy()


def _season_days(season: Sequence[int]) -> np.ndarray:
    if len(season) != 2:
        raise ValueError("season must contain start and end Julian dates")
    start, end = _validate_jday([season[0], season[1]], name="season")
    if end > start:
        return np.arange(start, end + 1, dtype=int)
    if end < start:
        return np.concatenate([np.arange(start, 367, dtype=int), np.arange(1, end + 1, dtype=int)])
    return np.array([start], dtype=int)


def leap_year(year: Any) -> bool | np.ndarray:
    """Determine Gregorian leap-year status.

    Translates R ``leap_year``. Scalars return ``bool``; array-like input
    returns a NumPy boolean array.
    """
    years, scalar = _as_series(year, name="year")
    numeric = pd.to_numeric(years, errors="coerce")
    if numeric.isna().any() or (numeric % 1 != 0).any():
        raise ValueError("year must contain integer years")
    values = numeric.astype(int)
    result = (values % 4 == 0) & ((values % 100 != 0) | (values % 400 == 0))
    return bool(result.iloc[0]) if scalar else result.to_numpy(dtype=bool)


def yearmoda_to_date(yearmoda: Any) -> date | np.ndarray:
    """Convert YEARMODA values to Python ``date`` objects.

    Translates R ``YEARMODA2Date``. Scalars return ``datetime.date``; array-like
    input returns a NumPy object array with ``date`` values and ``NaT`` for
    missing values.
    """
    values, scalar = _as_series(yearmoda, name="YEARMODA")
    numeric = pd.to_numeric(values, errors="coerce")
    missing = numeric.isna()
    invalid_width = values.loc[~missing].astype(str).str.replace(r"\.0$", "", regex=True).str.len() != 8
    if invalid_width.any():
        raise ValueError("YEARMODA values must use YYYYMMDD format")

    year = np.floor(numeric / 10000)
    month = np.floor((numeric % 10000) / 100)
    day = numeric % 100
    parsed = pd.to_datetime({"year": year, "month": month, "day": day}, errors="coerce")
    invalid = parsed.isna() & ~missing
    if invalid.any():
        raise ValueError("YEARMODA contains invalid dates")

    out = parsed.dt.date.astype(object)
    out.loc[missing] = pd.NaT
    return _finish_array(out, scalar=scalar)


def date_to_yearmoda(value: Any, *, hours: bool = False) -> int | float | np.ndarray:
    """Convert date-like values to YEARMODA or YEARMODAHO integers.

    Translates R ``Date2YEARMODA``. Strings, Python dates, datetimes, pandas
    timestamps, and array-like collections are accepted.
    """
    values, scalar = _as_series(value, name="Date")
    parsed = pd.to_datetime(values, errors="coerce")
    missing = values.isna() if hasattr(values, "isna") else pd.Series([False] * len(values))
    invalid = parsed.isna() & ~missing
    if invalid.any():
        raise ValueError("Date contains invalid date values")

    result = parsed.dt.year * 10000 + parsed.dt.month * 100 + parsed.dt.day
    if hours:
        result = result * 100 + parsed.dt.hour
    result = result.astype("Float64")
    result.loc[parsed.isna()] = pd.NA
    if scalar:
        return np.nan if pd.isna(result.iloc[0]) else int(result.iloc[0])
    return result.to_numpy(dtype=float if result.isna().any() else int)


def _dates_from_columns(frame: pd.DataFrame) -> pd.Series:
    required = ["Year", "Month", "Day"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Table is missing at least one required column ('Day','Month' or 'Year')")

    date_parts = frame[required].apply(pd.to_numeric, errors="coerce")
    has_missing = date_parts.isna().any(axis=1)
    invalid_components = ((date_parts % 1 != 0) & ~date_parts.isna()).any(axis=1)
    if invalid_components.any():
        raise ValueError("Year, Month, and Day must be integers")

    parsed = pd.to_datetime(
        {"year": date_parts["Year"], "month": date_parts["Month"], "day": date_parts["Day"]},
        errors="coerce",
    )
    if (parsed.isna() & ~has_missing).any():
        raise ValueError("Year, Month, and Day contain invalid dates")
    return parsed


def make_jday(dateframe: Any) -> pd.DataFrame:
    """Add Julian day-of-year values to a weather/date table.

    Translates R ``make_JDay`` for tables with ``Year``, ``Month``, and
    ``Day`` columns. As a Python extension, a ``Date`` column is also accepted
    when those components are absent. Row order and duplicate dates are
    preserved, matching the R function's data-frame mutation behavior.
    """
    frame = pd.DataFrame(dateframe).copy()
    if {"Year", "Month", "Day"}.issubset(frame.columns):
        parsed = _dates_from_columns(frame)
    elif "Date" in frame.columns:
        parsed = pd.to_datetime(frame["Date"], errors="coerce")
        invalid = parsed.isna() & frame["Date"].notna()
        if invalid.any():
            raise ValueError("Date contains invalid date values")
        frame["Year"] = parsed.dt.year.astype("Float64")
        frame["Month"] = parsed.dt.month.astype("Float64")
        frame["Day"] = parsed.dt.day.astype("Float64")
    else:
        raise ValueError("Table is missing at least one required column ('Day','Month' or 'Year')")

    frame["JDay"] = parsed.dt.dayofyear.astype("Float64")
    return frame


def add_date(df: Any) -> pd.DataFrame:
    """Add a ``Date`` column from date/time component columns.

    Translates R ``add_date``. ``Year``, ``Month``, and ``Day`` are required;
    ``Hour``, ``Minute``, and ``Second`` are used when present. Invalid finite
    dates raise ``ValueError``; rows with missing components receive ``NaT``.
    """
    frame = pd.DataFrame(df).copy()
    required = ["Year", "Month", "Day"]
    missing_required = [column for column in required if column not in frame.columns]
    if missing_required:
        raise ValueError("Required input column 'Year', 'Month' and/or 'Day' is missing.")

    parts = frame[required].apply(pd.to_numeric, errors="coerce")
    for optional in ["Hour", "Minute", "Second"]:
        if optional in frame.columns:
            parts[optional.lower()] = pd.to_numeric(frame[optional], errors="coerce")

    has_missing = parts.isna().any(axis=1)
    parsed = pd.to_datetime(parts, errors="coerce")
    if (parsed.isna() & ~has_missing).any():
        raise ValueError("date/time component columns contain invalid values")
    frame["Date"] = parsed
    return frame


def _jday_compare(check_date: Any, ref_date: int, season: tuple[int, int], *, later: bool) -> bool | np.ndarray | float:
    check_values, scalar = _as_series(check_date, name="check_date")
    numeric_check = pd.to_numeric(check_values, errors="coerce")
    if not ((numeric_check >= 1) & (numeric_check <= 366) & (numeric_check % 1 == 0)).any():
        raise ValueError("check_date is not a Julian date")
    ref = int(_validate_jday(ref_date, name="ref_date")[0])
    all_days = _season_days(season)
    ref_positions = np.flatnonzero(all_days == ref)
    if ref_positions.size == 0:
        return np.nan
    ref_position = ref_positions[0]

    result: list[Any] = []
    for value in numeric_check:
        if pd.isna(value) or value < 1 or value > 366 or value % 1 != 0:
            result.append(np.nan)
            continue
        positions = np.flatnonzero(all_days == int(value))
        if positions.size == 0:
            result.append(np.nan)
            continue
        result.append(bool(positions[0] > ref_position if later else positions[0] < ref_position))
    return _finish_array(pd.Series(result, dtype=object), scalar=scalar)


def jday_earlier(check_date: Any, ref_date: int, season: tuple[int, int] = (1, 366)) -> bool | np.ndarray | float:
    """Check whether Julian dates are earlier than a reference date in-season."""
    return _jday_compare(check_date, ref_date, season, later=False)


def jday_later(check_date: Any, ref_date: int, season: tuple[int, int] = (1, 366)) -> bool | np.ndarray | float:
    """Check whether Julian dates are later than a reference date in-season."""
    return _jday_compare(check_date, ref_date, season, later=True)


def jday_count(
    start_date: Any,
    end_date: Any,
    season: tuple[int, int] | None = None,
    *,
    leap_year: bool | int = False,
) -> int | float | np.ndarray:
    """Count days between Julian dates within a phenological season.

    Translates R ``JDay_count``. The returned count is the positional
    difference between ``end_date`` and ``start_date`` within the season, so
    adjacent dates differ by 1 rather than counting both endpoints.
    """
    if season is None:
        start_for_season = int(_validate_jday(start_date, name="start_date")[0])
        end_for_season = int(_validate_jday(end_date, name="end_date")[0])
        season = (start_for_season, end_for_season)

    all_days = _season_days(season)
    if isinstance(leap_year, bool):
        is_leap = leap_year
    elif isinstance(leap_year, (int, np.integer)) and int(leap_year) == float(leap_year):
        is_leap = bool(globals()["leap_year"](int(leap_year)))
    else:
        is_leap = bool(leap_year)
    if not is_leap:
        all_days = all_days[all_days != 366]

    starts, scalar_start = _as_series(start_date, name="start_date")
    ends, scalar_end = _as_series(end_date, name="end_date")
    start_numeric = pd.to_numeric(starts, errors="coerce")
    end_numeric = pd.to_numeric(ends, errors="coerce")
    if start_numeric.size != end_numeric.size:
        if start_numeric.size == 1:
            start_numeric = pd.Series(np.repeat(start_numeric.iloc[0], end_numeric.size))
        elif end_numeric.size == 1:
            end_numeric = pd.Series(np.repeat(end_numeric.iloc[0], start_numeric.size))
        else:
            raise ValueError("start_date and end_date must have compatible lengths")
    if start_numeric.isna().any() or ((start_numeric < 1) | (start_numeric > 366) | (start_numeric % 1 != 0)).any():
        raise ValueError("start_date is not a Julian date")
    if end_numeric.isna().any() or ((end_numeric < 1) | (end_numeric > 366) | (end_numeric % 1 != 0)).any():
        raise ValueError("end_date is not a Julian date")

    result: list[float] = []
    for start, end in zip(start_numeric.astype(int), end_numeric.astype(int), strict=False):
        start_positions = np.flatnonzero(all_days == start)
        end_positions = np.flatnonzero(all_days == end)
        if start_positions.size == 0 or end_positions.size == 0:
            result.append(np.nan)
        else:
            result.append(float(end_positions[0] - start_positions[0]))
    out = pd.Series(result)
    scalar = scalar_start and scalar_end
    if scalar:
        return np.nan if pd.isna(out.iloc[0]) else int(out.iloc[0])
    return out.to_numpy(dtype=float)


def daylength(
    latitude: float,
    jday: int | float | list[int | float] | np.ndarray | None = None,
    *,
    JDay: int | float | list[int | float] | np.ndarray | None = None,
    notimes_as_na: bool = False,
) -> dict[str, np.ndarray]:
    """Compute sunrise, sunset, and daylength from latitude and Julian day.

    Translates R ``daylength`` using the Spencer (1971) solar declination
    approximation and Almorox et al. (2005) sunrise/sunset equations. Polar
    nights use sunrise/sunset ``-99`` and daylength ``0``; polar days use
    sunrise/sunset ``99`` and daylength ``24`` unless ``notimes_as_na=True``.
    """
    if jday is None:
        jday = JDay
    if jday is None:
        raise ValueError("'jday' not specified")

    latitude_array = np.asarray(latitude, dtype=float)
    if latitude_array.ndim > 0 and latitude_array.size != 1:
        raise ValueError("'latitude' has more than one element")
    latitude_value = float(latitude_array.reshape(1)[0])
    if latitude_value > 90 or latitude_value < -90:
        warnings.warn("'latitude' is usually between -90 and 90", RuntimeWarning, stacklevel=2)

    jday_values = np.asarray(jday, dtype=float)
    if jday_values.ndim == 0:
        jday_values = jday_values.reshape(1)

    gamma = 2 * np.pi / 365 * (jday_values - 1)
    delta = 180 / np.pi * (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    cos_wo = (
        np.sin(-0.8333 / 360 * 2 * np.pi)
        - np.sin(latitude_value / 360 * 2 * np.pi) * np.sin(delta / 360 * 2 * np.pi)
    ) / (np.cos(latitude_value / 360 * 2 * np.pi) * np.cos(delta / 360 * 2 * np.pi))

    sunrise = np.full(cos_wo.shape, -99.0, dtype=float)
    sunset = np.full(cos_wo.shape, -99.0, dtype=float)
    normal_days = (cos_wo >= -1) & (cos_wo <= 1)
    sunrise[normal_days] = 12 - np.arccos(cos_wo[normal_days]) / (15 / 360 * 2 * np.pi)
    sunset[normal_days] = 12 + np.arccos(cos_wo[normal_days]) / (15 / 360 * 2 * np.pi)

    day_length = sunset - sunrise
    day_length[cos_wo > 1] = 0
    day_length[cos_wo < -1] = 24
    sunrise[day_length == 24] = 99
    sunset[day_length == 24] = 99

    if notimes_as_na:
        sunrise[np.isin(sunrise, [-99, 99])] = np.nan
        sunset[np.isin(sunset, [-99, 99])] = np.nan

    missing_jday = np.isnan(jday_values)
    sunrise[missing_jday] = np.nan
    sunset[missing_jday] = np.nan
    day_length[missing_jday] = np.nan

    return {"Sunrise": sunrise, "Sunset": sunset, "Daylength": day_length}


def get_last_date(dates: Any, *, first: bool = False) -> int | None:
    """Infer the last or first Julian date in a phenology record.

    Translates R ``get_last_date`` by treating the longest annual gap in
    observed dates as the break between phenological seasons.
    """
    values = pd.to_numeric(pd.Series(dates), errors="coerce").dropna()
    if values.empty:
        return None
    if ((values < 1) | (values > 366) | (values % 1 != 0)).any():
        raise ValueError("dates must be Julian dates from 1 through 366")

    pdates = np.sort(values.astype(int).to_numpy())
    if pdates.size == 1:
        return int(pdates[0])
    leg = np.concatenate([np.arange(180, 367, dtype=int), np.arange(1, 180, dtype=int)])
    gaps = np.diff(pdates).astype(float)
    first_pos = np.flatnonzero(leg == pdates[0])[0]
    last_pos = np.flatnonzero(leg == pdates[-1])[0]
    wrap_gap = float(first_pos - last_pos)
    if wrap_gap < 0:
        wrap_gap += 365
    gaps = np.concatenate([gaps, [wrap_gap]])
    max_gap = np.nanmax(gaps)
    gap_indices = np.flatnonzero(gaps == max_gap)
    last = int(np.max(pdates[gap_indices]))
    first_index = int(gap_indices[0])
    first_value = int(pdates[0] if first_index == pdates.size - 1 else pdates[first_index + 1])
    return first_value if first else last


YEARMODA2Date = yearmoda_to_date
Date2YEARMODA = date_to_yearmoda
make_JDay = make_jday
JDay_earlier = jday_earlier
JDay_later = jday_later
JDay_count = jday_count
