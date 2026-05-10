"""Weather import, checking, and station-handler placeholders from chillR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from unicodedata import normalize
import warnings

import numpy as np
import pandas as pd

from .date_utils import leap_year
from ._base import not_implemented, placeholder_record
from .temperature import interpolate_gaps, make_all_day_table


def get_weather(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``get_weather``; network retrieval is not implemented."""
    not_implemented("get_weather")


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(" ".join(part.strip() for part in self._current_cell).strip())
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell != "" for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


def weather_to_chillr(downloaded_weather: Any, database: str = "GSOD", *, drop_most: bool = True) -> pd.DataFrame | dict[str, Any]:
    """Convert downloaded weather records to chillR-style daily weather.

    Translates the deterministic dispatch behavior of R ``weather2chillR``.
    Network retrieval remains outside this function; local GSOD, CIMIS, and
    UCIPM-style downloaded tables are normalized.
    """
    preserve_wrapper = isinstance(downloaded_weather, Mapping) and "database" in downloaded_weather
    if preserve_wrapper:
        database = str(downloaded_weather["database"])
        source = downloaded_weather.get("weather", downloaded_weather)
    else:
        source = downloaded_weather

    database_key = str(database).upper()
    if database_key == "GSOD":
        converted = _convert_gsod(source)
    elif database_key == "CIMIS":
        converted = _convert_cimis(source, drop_most=drop_most)
    elif database_key == "UCIPM":
        converted = _convert_ucipm(source, drop_most=drop_most)
    else:
        raise ValueError("database must be one of 'GSOD', 'CIMIS', or 'UCIPM'")

    if preserve_wrapper:
        return {"database": database_key, "weather": converted}
    return converted


def chile_agromet_to_chillr(downloaded_weather_file: Any, *, drop_most: bool = True) -> pd.DataFrame:
    """Convert Chilean Agromet downloaded tables to chillR-style weather.

    Translates R ``chile_agromet2chillR``. The R function reads malformed Excel
    HTML files; Python accepts a path to an HTML table and, as a convenience for
    tests and pipelines, a DataFrame with the same columns.
    """
    raw = _read_chile_agromet_source(downloaded_weather_file)
    if "FECHA" not in raw.columns and "Date" not in raw.columns:
        raise ValueError("Chile Agromet data must contain a FECHA/Date column")

    out = raw.copy()
    rename: dict[str, str] = {}
    for column in out.columns:
        normalized = _normalize_name(column)
        if "fecha" in normalized:
            rename[column] = "Date"
        elif "xim" in normalized:
            rename[column] = "Tmax"
        elif "nim" in normalized:
            rename[column] = "Tmin"
        elif "aire" in normalized:
            rename[column] = "Tmean"
        elif "prec" in normalized:
            rename[column] = "Prec"
    out = out.rename(columns=rename)

    for column in out.columns:
        if column != "Date":
            out[column] = _coerce_decimal_series(out[column])

    parsed = pd.to_datetime(out["Date"], format="%d-%m-%Y", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Chile Agromet data contains malformed dates")
    out["Year"] = parsed.dt.year.astype(float)
    out["Month"] = parsed.dt.month.astype(float)
    out["Day"] = parsed.dt.day.astype(float)

    return _finalize_daily_conversion(
        out,
        essential_columns=["Year", "Month", "Day", "Tmin", "Tmean", "Tmax", "Prec"],
        drop_most=drop_most,
        required_columns=["Tmin", "Tmax"],
    )


def _normalize_name(value: Any) -> str:
    text = normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def _coerce_decimal_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "NA": pd.NA})
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _read_chile_agromet_source(source: Any) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    content = path.read_text(encoding="utf-8", errors="ignore")
    parser = _HTMLTableParser()
    parser.feed(content)
    rows = [row for row in parser.rows if any(cell for cell in row)]
    if not rows:
        raise ValueError("Chile Agromet file does not contain an HTML table")

    header_index = next((idx for idx, row in enumerate(rows) if any(_normalize_name(cell) == "fecha" for cell in row)), None)
    if header_index is None:
        raise ValueError("Chile Agromet file does not contain a FECHA header")
    header = rows[header_index]
    body = [row for row in rows[header_index + 1 :] if len(row) == len(header)]
    if not body:
        raise ValueError("Chile Agromet file does not contain data rows")
    return pd.DataFrame(body, columns=header)


def _ensure_date_parts(frame: pd.DataFrame, *, date_format: str | None = None) -> pd.DataFrame:
    out = frame.copy()
    if {"Year", "Month", "Day"}.issubset(out.columns):
        return out
    date_column = next((column for column in ["Date", "DATE", "date"] if column in out.columns), None)
    if date_column is None:
        raise ValueError("weather data must contain Year/Month/Day or a Date column")
    parsed = pd.to_datetime(out[date_column], format=date_format, errors="coerce")
    if parsed.isna().any():
        raise ValueError("weather data contains malformed dates")
    out["Year"] = parsed.dt.year.astype(float)
    out["Month"] = parsed.dt.month.astype(float)
    out["Day"] = parsed.dt.day.astype(float)
    return out


def _finalize_daily_conversion(
    frame: pd.DataFrame,
    *,
    essential_columns: Sequence[str],
    drop_most: bool,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    out = _ensure_date_parts(frame)
    for column in ["Year", "Month", "Day", *required_columns, "Tmean", "Prec"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    missing = [column for column in ["Year", "Month", "Day", *required_columns] if column not in out.columns]
    if missing:
        raise ValueError(f"converted weather is missing required column(s): {', '.join(missing)}")
    dates = _component_dates(out, hourly=False)
    out["_date"] = dates

    grouped = out.groupby("_date", sort=True, dropna=False)
    aggregated = grouped.agg(lambda values: values.mean() if pd.api.types.is_numeric_dtype(values) else values.dropna().iloc[0] if values.notna().any() else np.nan)
    aggregated = aggregated.reset_index(drop=True)
    aggregated["Year"] = grouped["_date"].first().dt.year.to_numpy(dtype=float)
    aggregated["Month"] = grouped["_date"].first().dt.month.to_numpy(dtype=float)
    aggregated["Day"] = grouped["_date"].first().dt.day.to_numpy(dtype=float)
    aggregated = aggregated.drop(columns=[column for column in ["_date"] if column in aggregated.columns])

    ordered = [column for column in essential_columns if column in aggregated.columns]
    if drop_most:
        return aggregated[ordered].reset_index(drop=True)
    others = [column for column in aggregated.columns if column not in ordered]
    return aggregated[ordered + others].reset_index(drop=True)


def _convert_gsod(source: Any) -> pd.DataFrame:
    frame = _as_weather_frame(source)
    if frame.empty:
        raise ValueError("GSOD weather data is empty")
    rename = {"DATE": "Date"}
    frame = frame.rename(columns={old: new for old, new in rename.items() if old in frame.columns})
    required = ["Date", "MIN", "MAX", "TEMP", "PRCP"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"GSOD weather data is missing required column(s): {', '.join(missing)}")

    parsed = pd.to_datetime(frame["Date"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("GSOD weather data contains malformed dates")
    out = pd.DataFrame(
        {
            "Year": parsed.dt.year.astype(float),
            "Month": parsed.dt.month.astype(float),
            "Day": parsed.dt.day.astype(float),
            "Tmin": _fahrenheit_to_celsius(frame["MIN"], missing_value=9999.9),
            "Tmax": _fahrenheit_to_celsius(frame["MAX"], missing_value=9999.9),
            "Tmean": _fahrenheit_to_celsius(frame["TEMP"], missing_value=9999.9),
            "Prec": _inches_to_mm(frame["PRCP"], missing_value=99.99),
        }
    )
    first_year = int(out["Year"].min())
    last_year = int(out["Year"].max())
    primers = pd.DataFrame(
        {
            "Year": [first_year, last_year],
            "Month": [1.0, 12.0],
            "Day": [1.0, 31.0],
            "Tmin": [np.nan, np.nan],
            "Tmax": [np.nan, np.nan],
            "Tmean": [np.nan, np.nan],
            "Prec": [np.nan, np.nan],
        }
    )
    complete = make_all_day_table(pd.concat([primers, out], ignore_index=True), add_date=True, no_variable_check=True)
    if "DATE" in complete.columns:
        complete = complete.rename(columns={"DATE": "Date"})
    return complete[["Date", "Year", "Month", "Day", "Tmin", "Tmax", "Tmean", "Prec"]]


def _fahrenheit_to_celsius(values: Any, *, missing_value: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.mask(np.isclose(numeric, missing_value))
    return ((numeric - 32.0) * 5.0 / 9.0).round(3)


def _inches_to_mm(values: Any, *, missing_value: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.mask(np.isclose(numeric, missing_value))
    return (numeric * 25.4).round(3)


def _convert_cimis(source: Any, *, drop_most: bool) -> pd.DataFrame:
    frame = _as_weather_frame(source)
    rename = {
        "Average Air Temperature": "Tmean",
        "Minimum Air Temperature": "Tmin",
        "Maximum Air Temperature": "Tmax",
        "Precipitation": "Prec",
    }
    frame = frame.rename(columns={old: new for old, new in rename.items() if old in frame.columns})
    return _finalize_daily_conversion(
        frame,
        essential_columns=["Year", "Month", "Day", "Tmin", "Tmax", "Tmean", "Prec"],
        drop_most=drop_most,
        required_columns=["Tmin", "Tmax"],
    )


def _convert_ucipm(source: Any, *, drop_most: bool) -> pd.DataFrame:
    frame = _as_weather_frame(source)
    rename = {"min": "Tmin", "Air.max": "Tmax", "Precip": "Prec"}
    frame = frame.rename(columns={old: new for old, new in rename.items() if old in frame.columns})
    if "Date" in frame.columns and not {"Year", "Month", "Day"}.issubset(frame.columns):
        date_text = frame["Date"].astype(str)
        if date_text.str.fullmatch(r"\d{8}").all():
            frame["Year"] = pd.to_numeric(date_text.str.slice(0, 4), errors="coerce")
            frame["Month"] = pd.to_numeric(date_text.str.slice(4, 6), errors="coerce")
            frame["Day"] = pd.to_numeric(date_text.str.slice(6, 8), errors="coerce")
    return _finalize_daily_conversion(
        frame,
        essential_columns=["Year", "Month", "Day", "Tmin", "Tmax", "Prec"],
        drop_most=drop_most,
        required_columns=["Tmin", "Tmax"],
    )




def _as_weather_frame(value: Any, *, name: str = "weather") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value).copy()
    except Exception as exc:  # pragma: no cover - defensive for unusual table-likes
        raise TypeError(f"{name} must be convertible to a pandas DataFrame") from exc


def _empty_weather(value: Any) -> bool:
    if value is None:
        return True
    try:
        return len(value) == 0
    except TypeError:
        return False


def _component_dates(frame: pd.DataFrame, *, hourly: bool) -> pd.Series:
    required = ["Year", "Month", "Day"] + (["Hour"] if hourly else [])
    parts = frame[required].apply(pd.to_numeric, errors="coerce")
    if parts.isna().any(axis=None):
        missing = ", ".join(parts.columns[parts.isna().any()].tolist())
        raise ValueError(f"date/time columns contain missing or non-numeric values: {missing}")
    if ((parts % 1) != 0).any(axis=None):
        raise ValueError("date/time columns must contain integer values")
    if hourly:
        invalid_hour = (parts["Hour"] < 0) | (parts["Hour"] > 23)
        if invalid_hour.any():
            raise ValueError("Hour values must be integers from 0 through 23")
        parsed = pd.to_datetime(
            {"year": parts["Year"], "month": parts["Month"], "day": parts["Day"], "hour": parts["Hour"]},
            errors="coerce",
        )
    else:
        parsed = pd.to_datetime({"year": parts["Year"], "month": parts["Month"], "day": parts["Day"]}, errors="coerce")
    if parsed.isna().any():
        raise ValueError("weather contains invalid date values")
    return parsed


def _check_result(
    *,
    hourly: bool,
    weather_object: bool = False,
    error: str = "none",
    dates_as_yearmoda: bool = False,
    warning_messages: list[str] | None = None,
) -> dict[str, Any]:
    compliant = error == "none"
    messages = warning_messages or []
    return {
        "data_frequency": "hourly" if hourly else "daily",
        "weather_object": weather_object,
        "chillR_compliant": compliant,
        "valid": compliant,
        "error": error,
        "dates_as_YEARMODA": dates_as_yearmoda,
        "warnings": messages,
    }


def _with_error(out: dict[str, Any], message: str) -> dict[str, Any]:
    out["error"] = message
    out["chillR_compliant"] = False
    out["valid"] = False
    out.setdefault("warnings", []).append(f"Error - {message}")
    warnings.warn(f"Error - {message}", RuntimeWarning, stacklevel=2)
    return out


def _coerce_yearmoda_columns(frame: pd.DataFrame, *, hourly: bool) -> tuple[pd.DataFrame, bool]:
    out = frame.copy()
    if not hourly and "YEARMODA" in out.columns and not {"Year", "Month", "Day"}.issubset(out.columns):
        value = pd.to_numeric(out["YEARMODA"], errors="coerce")
        out["Year"] = np.floor(value / 10000)
        out["Month"] = np.floor((value % 10000) / 100)
        out["Day"] = value % 100
        return out, True
    if hourly and "YEARMODAHO" in out.columns and not {"Year", "Month", "Day", "Hour"}.issubset(out.columns):
        value = pd.to_numeric(out["YEARMODAHO"], errors="coerce")
        out["Year"] = np.floor(value / 1_000_000)
        out["Month"] = np.floor((value % 1_000_000) / 10_000)
        out["Day"] = np.floor((value % 10_000) / 100)
        out["Hour"] = value % 100
        return out, True
    return out, False


def _season_day_count(season: int, start_date: int, end_date: int, *, end_at_present: bool) -> int:
    year_length = 366 if leap_year(season) else 365
    start = min(max(int(start_date), 1), year_length)
    end = min(max(int(end_date), 1), year_length)
    if start < end:
        season_days = end - start + 1
        start_ts = pd.Timestamp(year=season, month=1, day=1) + pd.Timedelta(days=start - 1)
        end_ts = pd.Timestamp(year=season, month=1, day=1) + pd.Timedelta(days=end - 1)
    else:
        previous_year_length = 366 if leap_year(season - 1) else 365
        season_days = previous_year_length - start + end + 1
        start_ts = pd.Timestamp(year=season - 1, month=1, day=1) + pd.Timedelta(days=start - 1)
        end_ts = pd.Timestamp(year=season, month=1, day=1) + pd.Timedelta(days=end - 1)

    now = pd.Timestamp.now()
    if end_at_present and season == now.year:
        end_ts = min(end_ts, now)
        season_days = max(0, int(round((end_ts - start_ts).total_seconds() / 86400)) + 1)
    return int(season_days)


def _fix_weather_qc(
    fixedweather: pd.DataFrame,
    *,
    start_date: int,
    end_date: int,
    columns: Sequence[str],
    end_at_present: bool,
) -> pd.DataFrame:
    qc_frame = fixedweather.copy()
    dates = _component_dates(qc_frame, hourly=False)
    qc_frame["JDay"] = dates.dt.dayofyear.astype(int)
    qc_frame["sea"] = np.nan
    if start_date < end_date:
        in_season = (qc_frame["JDay"] >= start_date) & (qc_frame["JDay"] <= end_date)
        qc_frame.loc[in_season, "sea"] = qc_frame.loc[in_season, "Year"]
    else:
        qc_frame.loc[qc_frame["JDay"] >= start_date, "sea"] = qc_frame.loc[qc_frame["JDay"] >= start_date, "Year"] + 1
        qc_frame.loc[qc_frame["JDay"] <= end_date, "sea"] = qc_frame.loc[qc_frame["JDay"] <= end_date, "Year"]

    seasons = pd.to_numeric(qc_frame["sea"], errors="coerce").dropna().astype(int).drop_duplicates().tolist()
    rows: list[dict[str, Any]] = []
    for season in seasons:
        season_mask = qc_frame["sea"] == season
        season_days = _season_day_count(season, start_date, end_date, end_at_present=end_at_present)
        data_days = int(season_mask.sum())
        row: dict[str, Any] = {
            "Season": f"{season - 1}/{season}",
            "End_year": season,
            "Season_days": season_days,
            "Data_days": data_days,
        }
        missing_counts: list[int] = []
        for column in columns:
            no_column = f"no_{column}"
            missing_count = int(qc_frame.loc[season_mask, no_column].fillna(False).astype(bool).sum())
            total_missing = missing_count + season_days - data_days
            row[f"Missing_{column}"] = total_missing
            missing_counts.append(missing_count)
        row["Incomplete_days"] = (max(missing_counts) if missing_counts else 0) + season_days - data_days
        row["Perc_complete"] = round((season_days - row["Incomplete_days"]) / season_days * 100, 1) if season_days else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fix_weather(
    weather: Any,
    start_year: int = 0,
    end_year: int = 3000,
    start_date: int = 1,
    end_date: int = 366,
    columns: Sequence[str] = ("Tmin", "Tmax"),
    end_at_present: bool = True,
) -> dict[str, pd.DataFrame]:
    """Complete, interpolate, and quality-check a daily weather record.

    Translates R ``fix_weather``. The returned dictionary has ``weather`` with
    completed daily rows and interpolated columns, plus ``QC`` with seasonal
    missing-data summaries.
    """
    source = weather["weather"] if isinstance(weather, Mapping) and "weather" in weather else weather
    frame = _as_weather_frame(source)
    if frame.empty:
        raise ValueError("no data provided (weather is null)")
    if "Year" not in frame.columns and "YEAR" in frame.columns:
        frame = frame.rename(columns={"YEAR": "Year", "MONTH": "Month", "DAY": "Day"})
    frame, _ = _coerce_yearmoda_columns(frame, hourly=False)

    required = ["Year", "Month", "Day", *columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"weather is missing required column(s): {', '.join(missing)}")
    dates = _component_dates(frame, hourly=False)
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    subset = frame.loc[(frame["Year"] >= int(start_year)) & (frame["Year"] <= int(end_year))].copy()
    if subset.empty:
        raise ValueError("no weather records remain after applying start_year/end_year")

    fixedweather = make_all_day_table(subset, add_date=True, no_variable_check=True)
    if end_at_present and "DATE" in fixedweather.columns:
        fixedweather = fixedweather.loc[pd.to_datetime(fixedweather["DATE"]) < pd.Timestamp.now()].reset_index(drop=True)

    for column in columns:
        fixedweather[column] = pd.to_numeric(fixedweather[column], errors="coerce")
        interp = interpolate_gaps(fixedweather[column])
        fixedweather[column] = interp["interp"]
        fixedweather[f"no_{column}"] = interp["missing"]

    qc = _fix_weather_qc(
        fixedweather,
        start_date=int(start_date),
        end_date=int(end_date),
        columns=list(columns),
        end_at_present=bool(end_at_present),
    )
    return {"weather": fixedweather, "QC": qc}


def check_temperature_record(
    weather: Any,
    *,
    hourly: bool = False,
    completeness_check: bool = True,
    no_variable_check: bool = False,
) -> dict[str, Any]:
    """Check whether a daily or hourly temperature table is chillR-compatible.

    Translates R ``check_temperature_record``. The output follows the R result
    names and also includes ``valid``/``warnings`` convenience fields for
    Python callers.
    """
    out = _check_result(hourly=hourly)
    if _empty_weather(weather):
        return _with_error(out, "no data provided (weather is null)")

    weather_object = False
    source = weather
    if isinstance(weather, Mapping) and (("hourtemps" if hourly else "weather") in weather):
        source = weather["hourtemps" if hourly else "weather"]
        weather_object = True
        out["weather_object"] = True

    frame = _as_weather_frame(source)
    if frame.empty:
        return _with_error(out, "no data provided (weather is null)")

    if "Year" not in frame.columns and "YEAR" in frame.columns:
        frame = frame.rename(columns={"YEAR": "Year", "MONTH": "Month", "DAY": "Day"})
    frame, dates_as_yearmoda = _coerce_yearmoda_columns(frame, hourly=hourly)
    out["weather_object"] = weather_object
    out["dates_as_YEARMODA"] = dates_as_yearmoda

    if hourly:
        required_columns = ["Year", "Month", "Day", "Hour"] + ([] if no_variable_check else ["Temp"])
    else:
        required_columns = ["Year", "Month", "Day"] + ([] if no_variable_check else ["Tmin", "Tmax"])

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        prefix = "Column missing" if len(missing) == 1 else "Columns missing"
        return _with_error(out, f"{prefix}: {', '.join(missing)}")

    numeric_errors: list[str] = []
    for column in required_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if (converted.isna() & frame[column].notna()).any():
            numeric_errors.append(column)
        else:
            frame[column] = converted
    if numeric_errors:
        prefix = "One column is not numeric" if len(numeric_errors) == 1 else "The following columns are not numeric"
        return _with_error(out, f"{prefix}: {', '.join(numeric_errors)}")

    try:
        alltimes = _component_dates(frame, hourly=hourly)
    except ValueError as exc:
        return _with_error(out, str(exc))

    if completeness_check:
        if len(alltimes) < 2 or not (alltimes.iloc[0] < alltimes.iloc[-1]):
            return _with_error(out, "The first date of the record must be before the last date")

        frequency = "h" if hourly else "D"
        datevec = pd.date_range(alltimes.iloc[0], alltimes.iloc[-1], freq=frequency)
        observed = pd.Index(alltimes)
        missing_records = int((~datevec.isin(observed)).sum())
        repeated_records = int((observed.value_counts() > 1).sum())

        if hourly:
            total_missing_temps = missing_records if no_variable_check else missing_records + int(frame["Temp"].isna().sum())
            out["completeness_check"] = {
                "missing_records": missing_records,
                "repeated_records": repeated_records,
                "total_missing_Temps": total_missing_temps,
            }
        else:
            if no_variable_check:
                total_missing_tmin = missing_records
                total_missing_tmax = missing_records
            else:
                total_missing_tmin = missing_records + int(frame["Tmin"].isna().sum())
                total_missing_tmax = missing_records + int(frame["Tmax"].isna().sum())
            out["completeness_check"] = {
                "missing_records": missing_records,
                "repeated_records": repeated_records,
                "total_missing_Tmin": total_missing_tmin,
                "total_missing_Tmax": total_missing_tmax,
            }

    out["chillR_compliant"] = True
    out["valid"] = True
    return out


def check_temperature_scenario(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``check_temperature_scenario`` returning a mock report."""
    return placeholder_record("check_temperature_scenario", args=args, kwargs=kwargs, valid=True, warnings=[])


def handle_cimis(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``handle_cimis``; external data access is not implemented."""
    not_implemented("handle_cimis")


def handle_dwd(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``handle_dwd``; external data access is not implemented."""
    not_implemented("handle_dwd")


def handle_dwd_old(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``handle_dwd_old``; external data access is not implemented."""
    not_implemented("handle_dwd_old")


def handle_gsod(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``handle_gsod``; external data access is not implemented."""
    not_implemented("handle_gsod")


def handle_gsod_old(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``handle_gsod_old``; external data access is not implemented."""
    not_implemented("handle_gsod_old")


def handle_ucipm(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``handle_ucipm``; external data access is not implemented."""
    not_implemented("handle_ucipm")


def make_california_ucipm_station_list() -> list[dict[str, Any]]:
    """Placeholder for R ``make_california_UCIPM_station_list``."""
    return []


weather2chillR = weather_to_chillr
chile_agromet2chillR = chile_agromet_to_chillr
make_california_UCIPM_station_list = make_california_ucipm_station_list
