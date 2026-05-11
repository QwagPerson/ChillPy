"""Climate and temperature scenario helpers mapped from chillR."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from ._base import as_list, not_implemented, placeholder_record
from .utils import extract_differences_between_characters, select_by_file_extension


def make_climate_scenario(
    metric_summary: Any,
    caption: Sequence[str] | None = None,
    labels: Any = None,
    time_series: bool = False,
    historic_data: Any = None,
    add_to: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Make climate scenarios for plotting from a list of climate metric data.

    Translates R ``make_climate_scenario``.

    Parameters
    ----------
    metric_summary : Any
        List of climate metric data, e.g. produced by ``temp_response_daily_list``.
    caption : Sequence[str], optional
        Vector of up to three character strings for the plot caption.
    labels : Any, optional
        Labels for the scenarios. Defaults to keys of ``metric_summary`` if it's
        a dict.
    time_series : bool, default False
        Whether the scenario contains a time series.
    historic_data : Any, optional
        DataFrame of historic observations.
    add_to : list, optional
        Existing list of climate scenarios to append to.

    Returns
    -------
    list of dict
        Climate scenario objects.
    """
    if labels is None:
        if isinstance(metric_summary, dict):
            labels = list(metric_summary.keys())
        elif isinstance(metric_summary, (list, tuple)):
            labels = [str(i) for i in range(len(metric_summary))]

    if labels is not None:
        if len(as_list(labels)) != len(as_list(metric_summary)):
            raise ValueError("number of labels doesn't match number of elements")
        if time_series:
            labels = [float(lb) for lb in as_list(labels)]

    h_data = historic_data
    if h_data is not None:
        if not isinstance(h_data, pd.DataFrame):
            # R check: if is.data.frame(historic_data[[1]]) historic_data<-historic_data[[1]]
            if isinstance(h_data, (list, dict)) and len(h_data) > 0:
                first_val = (
                    next(iter(h_data.values()))
                    if isinstance(h_data, dict)
                    else h_data[0]
                )
                if isinstance(first_val, pd.DataFrame):
                    h_data = first_val

    new_scenario = {
        "data": metric_summary,
        "caption": caption,
        "time_series": time_series,
        "labels": labels,
        "historic_data": h_data,
    }

    if add_to is None:
        return [new_scenario]

    if not isinstance(add_to, list):
        raise ValueError("add_to needs to be a list of climate scenarios")

    add_to.append(new_scenario)
    return add_to


def make_climate_scenario_from_files(
    metric_folder: str | Path,
    criteria_list: list[str | list[str]],
    caption: Sequence[str] | None = None,
    time_series: bool = False,
    labels: Any = None,
    historic_data: Any = None,
) -> dict[str, Any]:
    """Make climate scenario from multiple saved CSV files.

    Translates R ``make_climate_scenario_from_files``.

    Parameters
    ----------
    metric_folder : str or Path
        Folder holding the CSV files.
    criteria_list : list
        Selection criteria for filenames.
    caption : Sequence[str], optional
        Plot caption.
    time_series : bool, default False
        Whether the scenario contains a time series.
    labels : Any, optional
        Labels for the time scenarios.
    historic_data : Any, optional
        Historic data DataFrame.

    Returns
    -------
    dict
        Climate scenario object.
    """
    folder = Path(metric_folder)
    if not folder.is_dir():
        raise ValueError(f"metric_folder '{metric_folder}' is not a directory")

    import re

    file_list = [str(f) for f in folder.iterdir() if f.is_file()]

    for criteria in criteria_list:
        to_keep = []
        criteria_vals = as_list(criteria)
        for val in criteria_vals:
            pattern = re.compile(str(val))
            to_keep.extend([f for f in file_list if pattern.search(f)])
        file_list = list(set(to_keep))

    dataset = []
    # R sorts them if possible via the loop? R uses alphabetical list.files order usually.
    file_list.sort()
    for f in file_list:
        dataset.append(pd.read_csv(f))

    return {
        "data": dataset,
        "caption": caption,
        "time_series": time_series,
        "labels": labels,
        "historic_data": historic_data,
    }


def temperature_scenario_from_records(
    weather: pd.DataFrame,
    year: int | Sequence[int],
    weather_start: int | None = None,
    weather_end: int | None = None,
    scen_type: str = "running_mean",
    runn_mean: int = 15,
) -> dict[str, dict[str, Any]]:
    """Make monthly temperature scenario from historic records.

    Translates R ``temperature_scenario_from_records``.

    Parameters
    ----------
    weather : pd.DataFrame
        Daily weather data with columns 'Month', 'Day', 'Year', 'Tmin', 'Tmax'.
    year : int or Sequence of int
        Years for which the scenario is to be produced.
    weather_start : int, optional
        Start year of the period to consider.
    weather_end : int, optional
        End year of the period to consider.
    scen_type : {"regression", "running_mean"}, default "running_mean"
        Method to produce the scenario.
    runn_mean : int, default 15
        Number of years for running mean.

    Returns
    -------
    dict
        Dictionary of temperature scenario objects keyed by year.
    """
    from .weather import check_temperature_scenario
    from .utils import runn_mean_pred

    required = ["Tmin", "Tmax", "Day", "Month", "Year"]
    if not all(col in weather.columns for col in required):
        warnings.warn(
            f"Error - required columns missing ({', '.join(required)})", stacklevel=2
        )
        return {}

    w_start = weather_start if weather_start is not None else weather["Year"].min()
    w_end = weather_end if weather_end is not None else weather["Year"].max()

    if w_start == w_end:
        warnings.warn("Error - only one data year selected", stacklevel=2)
        return {}
    if w_start > w_end:
        warnings.warn("Error - End year before start year", stacklevel=2)
        return {}

    weather_filtered = weather[
        (weather["Year"] >= w_start) & (weather["Year"] <= w_end)
    ].copy()

    if weather_filtered.empty:
        warnings.warn("Error - no weather records in specified interval", stacklevel=2)
        return {}

    past_means_tmin = (
        weather_filtered.groupby(["Year", "Month"])["Tmin"].mean().reset_index()
    )
    past_means_tmax = (
        weather_filtered.groupby(["Year", "Month"])["Tmax"].mean().reset_index()
    )

    out_scenarios = {}
    years_to_process = as_list(year)

    for y_val in years_to_process:
        baseclim = pd.DataFrame(
            index=range(1, 13), columns=["Tmin", "Tmax"], dtype=float
        )

        error_occurred = False
        for v, past_means in [("Tmin", past_means_tmin), ("Tmax", past_means_tmax)]:
            for i in range(1, 13):
                monthly = past_means[past_means["Month"] == i]
                if monthly.empty:
                    warnings.warn(
                        f"Error - no data for {v} in month {i}", stacklevel=2
                    )
                    error_occurred = True
                    break

                if scen_type == "regression":
                    try:
                        # Simple linear regression using numpy.polyfit if sklearn is not available
                        X = monthly["Year"].values
                        Y = monthly[v].values
                        coeffs = np.polyfit(X, Y, 1)
                        # coeffs[0] is slope, coeffs[1] is intercept
                        baseclim.loc[i, v] = coeffs[0] * float(y_val) + coeffs[1]
                    except Exception:
                        warnings.warn(
                            f"Error - unable to calculate regression for {v} in month {i}",
                            stacklevel=2,
                        )
                        error_occurred = True
                        break
                elif scen_type == "running_mean":
                    res = runn_mean_pred(
                        indep=monthly["Year"].values,
                        dep=monthly[v].values,
                        pred=[float(y_val)],
                        runn_mean=runn_mean,
                    )
                    val = res["predicted"][0]
                    if pd.isna(val):
                        warnings.warn(
                            f"Error - cannot calculate value for {v} in month {i}",
                            stacklevel=2,
                        )
                        error_occurred = True
                        break
                    baseclim.loc[i, v] = val
            if error_occurred:
                break

        if not error_occurred:
            scenario_obj = {
                "data": baseclim,
                "scenario_year": y_val,
                "reference_year": np.nan,
                "scenario_type": "absolute",
                "labels": "regression-based scenario"
                if scen_type == "regression"
                else "running mean scenario",
            }
            # Final check as in R (though we built it to be compliant)
            out_scenarios[str(y_val)] = check_temperature_scenario(
                scenario_obj, warn_me=False
            )

    return out_scenarios


def temperature_scenario_baseline_adjustment(
    baseline_temperature_scenario: Any,
    temperature_scenario: Any,
    temperature_check_args: dict[str, Any] | None = None,
    warn_me: bool = True,
    required_variables: Sequence[str] = ("Tmin", "Tmax"),
) -> list[dict[str, Any]] | dict[str, Any]:
    """Make temperature scenario relative to a particular baseline.

    Translates R ``temperature_scenario_baseline_adjustment``.
    """
    from .weather import check_temperature_scenario

    # R: if(!is.data.frame(baseline_temperature_scenario[[1]]))
    # baseline_temperature_scenario<-baseline_temperature_scenario[[1]]
    # This seems to handle if the input is a list of one scenario.
    if (
        isinstance(baseline_temperature_scenario, list)
        and len(baseline_temperature_scenario) > 0
    ):
        base_scen = baseline_temperature_scenario[0]
    else:
        base_scen = baseline_temperature_scenario

    if isinstance(temperature_scenario, (pd.DataFrame, dict)) and not isinstance(
        temperature_scenario, list
    ):
        temp_scens = [temperature_scenario]
    else:
        temp_scens = as_list(temperature_scenario)

    check_args = {
        "n_intervals": 12,
        "check_scenario_type": True,
        "scenario_check_thresholds": (-5, 10),
        "update_scenario_type": True,
        "warn_me": True,
    }
    if temperature_check_args:
        check_args.update(temperature_check_args)

    base_scen = check_temperature_scenario(
        base_scen,
        n_intervals=check_args["n_intervals"],
        check_scenario_type=check_args["check_scenario_type"],
        scenario_check_thresholds=check_args["scenario_check_thresholds"],
        update_scenario_type=check_args["update_scenario_type"],
        warn_me=check_args["warn_me"],
        required_variables=required_variables,
    )

    processed_scens = []
    for ts in temp_scens:
        ts_checked = check_temperature_scenario(
            ts,
            n_intervals=check_args["n_intervals"],
            check_scenario_type=check_args["check_scenario_type"],
            scenario_check_thresholds=check_args["scenario_check_thresholds"],
            update_scenario_type=check_args["update_scenario_type"],
            warn_me=check_args["warn_me"],
            required_variables=required_variables,
        )

        out_scen = ts_checked.copy()
        if (
            base_scen["scenario_type"] == "absolute"
            and ts_checked["scenario_type"] == "absolute"
        ):
            if "GCM" in out_scen["data"].columns:
                cols = [c for c in out_scen["data"].columns if c != "GCM"]
                out_scen["data"][cols] = (
                    ts_checked["data"][cols] - base_scen["data"][cols]
                )
            else:
                out_scen["data"] = ts_checked["data"] - base_scen["data"]
            out_scen["reference_year"] = base_scen["scenario_year"]
            out_scen["scenario_type"] = "relative"
        elif (
            base_scen["scenario_type"] == "relative"
            and ts_checked["scenario_type"] == "relative"
        ):
            if pd.isna(base_scen["scenario_year"]) or pd.isna(
                ts_checked["reference_year"]
            ):
                if warn_me:
                    warnings.warn(
                        "scenario year of baseline scenario and/or reference year of the temperature scenario "
                        "not specified - can't verify whether this is a valid transaction!",
                        stacklevel=2,
                    )
            elif base_scen["scenario_year"] != ts_checked["reference_year"]:
                raise ValueError(
                    "scenario year of the baseline scenario isn't equal to the reference year of the temperature scenario - "
                    "these scenarios aren't compatible."
                )

            if "GCM" in out_scen["data"].columns:
                cols = [c for c in out_scen["data"].columns if c != "GCM"]
                out_scen["data"][cols] = (
                    ts_checked["data"][cols] + base_scen["data"][cols]
                )
            else:
                out_scen["data"] = ts_checked["data"] + base_scen["data"]
            out_scen["reference_year"] = base_scen["reference_year"]
            out_scen["scenario_type"] = "relative"

        processed_scens.append(out_scen)

    if len(processed_scens) == 1 and not isinstance(temperature_scenario, list):
        return processed_scens[0]
    return processed_scens


def _is_temperature_scenario(value: Any) -> bool:
    if isinstance(value, pd.DataFrame):
        return True
    if isinstance(value, Mapping):
        return "data" in value or {"Tmin", "Tmax"}.issubset(value.keys())
    return False


def _iter_temperature_scenarios(value: Any) -> list[tuple[str, Any]]:
    if _is_temperature_scenario(value):
        return [("0", value)]
    if isinstance(value, Mapping):
        return [(str(name), scenario) for name, scenario in value.items()]
    return [(str(index), scenario) for index, scenario in enumerate(as_list(value))]


def _scenario_has_na(value: Any) -> bool:
    data = value.get("data", value) if isinstance(value, Mapping) else value
    if isinstance(data, pd.DataFrame):
        return data.isna().any().any()
    try:
        return pd.DataFrame(data).isna().any().any()
    except Exception:
        return False


def _yearmoda(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["Year"].astype(int) * 10000
        + frame["Month"].astype(int) * 100
        + frame["Day"].astype(int)
    )


def temperature_generation(
    weather: Any,
    years: Sequence[int],
    sim_years: Sequence[int],
    temperature_scenario: Any = None,
    seed: int = 99,
    check_temperature_scenario_type: bool = True,
    temperature_check_args: dict[str, Any] | None = None,
    max_reference_year_difference: int = 5,
    warn_me: bool = True,
    remove_na_scenarios: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generation of synthetic temperature records.

    Translates the chillR ``temperature_generation`` interface. The original
    implementation delegates the stochastic weather generator to the R-only
    ``RMAWGEN`` package. This Python version keeps the same input validation and
    monthly scenario semantics, then generates seedable daily records by
    bootstrapping calibrated daily temperature anomalies.
    """
    from .weather import check_temperature_scenario
    from .temperature import make_all_day_table

    if temperature_scenario is None:
        temperature_scenario = pd.DataFrame(
            {"Tmin": np.zeros(12), "Tmax": np.zeros(12)}
        )

    if len(years) != 2:
        raise ValueError("years must contain start and end years")
    if len(sim_years) != 2:
        raise ValueError("sim_years must contain start and end years")

    year_min, year_max = int(years[0]), int(years[1])
    year_min_sim, year_max_sim = int(sim_years[0]), int(sim_years[1])
    if year_min > year_max:
        raise ValueError("years must be in ascending order")
    if year_min_sim > year_max_sim:
        raise ValueError("sim_years must be in ascending order")

    if isinstance(weather, Mapping) and {"weather", "QC"}.issubset(weather.keys()):
        weather = weather["weather"]
    weather_frame = pd.DataFrame(weather).copy()
    required_weather = ["Month", "Day", "Year", "Tmin", "Tmax"]
    missing_weather = [column for column in required_weather if column not in weather_frame.columns]
    if missing_weather:
        raise ValueError(
            "weather must contain the columns Month, Day, Year, Tmin and Tmax"
        )
    for column in required_weather:
        weather_frame[column] = pd.to_numeric(weather_frame[column], errors="coerce")
    if weather_frame[required_weather].isna().any().any():
        raise ValueError("weather contains missing or non-numeric date or temperature values")

    temp_scens = _iter_temperature_scenarios(temperature_scenario)

    check_args = {
        "n_intervals": 12,
        "check_scenario_type": check_temperature_scenario_type,
        "scenario_check_thresholds": (-5, 10),
        "update_scenario_type": True,
        "warn_me": warn_me,
    }
    if temperature_check_args:
        check_args.update(temperature_check_args)

    if remove_na_scenarios:
        temp_scens = [
            (name, scenario)
            for name, scenario in temp_scens
            if not _scenario_has_na(scenario)
        ]

    processed_scens: list[tuple[str, dict[str, Any]]] = []
    for name, ts in temp_scens:
        checked = check_temperature_scenario(
            ts,
            n_intervals=check_args["n_intervals"],
            check_scenario_type=check_args["check_scenario_type"],
            scenario_check_thresholds=check_args["scenario_check_thresholds"],
            update_scenario_type=check_args["update_scenario_type"],
            warn_me=check_args["warn_me"],
        )
        if len(checked["data"]) != 12:
            raise ValueError("This function only works with monthly temperature scenarios")
        processed_scens.append((name, {**checked, "data": checked["data"].copy()}))

    if not processed_scens:
        return {}

    calendar = make_all_day_table(
        pd.DataFrame(
            {
                "Year": [year_min, year_max],
                "Month": [1, 12],
                "Day": [1, 31],
                "YEARMODA": [year_min * 10000 + 101, year_max * 10000 + 1231],
            }
        ),
        no_variable_check=True,
    )
    weather_frame["YEARMODA"] = _yearmoda(weather_frame)
    calibration_weather = weather_frame.loc[
        (weather_frame["Year"] >= year_min) & (weather_frame["Year"] <= year_max),
        ["YEARMODA", "Month", "Day", "Year", "Tmin", "Tmax"],
    ].copy()
    calibration_weather = (
        calibration_weather.groupby("YEARMODA", as_index=False)
        .agg({"Month": "first", "Day": "first", "Year": "first", "Tmin": "mean", "Tmax": "mean"})
    )
    calibration = calendar[["YEARMODA", "Year", "Month", "Day"]].merge(
        calibration_weather[["YEARMODA", "Tmin", "Tmax"]],
        on="YEARMODA",
        how="left",
    )
    if calibration[["Tmin", "Tmax"]].isna().any().any():
        raise ValueError("Weather record not complete for the calibration period - NULL returned")

    monthly = calibration.groupby("Month", sort=True)[["Tmin", "Tmax"]].mean()
    monthly_mid = (monthly["Tmin"] + monthly["Tmax"]) / 2
    monthly_range = monthly["Tmax"] - monthly["Tmin"]
    calibration["Temp_mid"] = (calibration["Tmin"] + calibration["Tmax"]) / 2
    calibration["Temp_range"] = calibration["Tmax"] - calibration["Tmin"]
    calibration["mid_anomaly"] = calibration["Temp_mid"] - calibration["Month"].map(monthly_mid)
    range_denominator = calibration["Month"].map(monthly_range).replace(0, np.nan)
    calibration["range_ratio"] = (calibration["Temp_range"] / range_denominator).fillna(1.0)

    sim_dates = make_all_day_table(
        pd.DataFrame(
            {
                "Year": [year_min_sim, year_max_sim],
                "Month": [1, 12],
                "Day": [1, 31],
                "YEARMODA": [
                    year_min_sim * 10000 + 101,
                    year_max_sim * 10000 + 1231,
                ],
            }
        ),
        no_variable_check=True,
    )[["YEARMODA", "DATE", "Year", "Month", "Day"]].copy()

    sim_out: dict[str, pd.DataFrame] = {}
    baseline_reference = float(np.median([year_min, year_max]))
    for name, ts in processed_scens:
        reference_year = ts["reference_year"]
        if ts["scenario_type"] == "relative":
            if pd.isna(reference_year):
                if warn_me:
                    warnings.warn(
                        "Reference year missing - can't check if relative temperature scenario is valid",
                        stacklevel=2,
                    )
            elif not isinstance(reference_year, (int, float, np.integer, np.floating)):
                if warn_me:
                    warnings.warn("Reference year not numeric", stacklevel=2)
            else:
                year_diff = abs(float(reference_year) - baseline_reference)
                if 0 < year_diff <= max_reference_year_difference and warn_me:
                    warnings.warn(
                        f"{year_diff:g} year(s) difference between reference years of the temperature scenario "
                        "and the dataset used for calibrating the weather generator",
                        stacklevel=2,
                    )
                if year_diff > max_reference_year_difference:
                    raise ValueError(
                        "Difference between reference years of the temperature scenario and the dataset used "
                        f"for calibrating the weather generator greater than {max_reference_year_difference} "
                        f"years ({year_diff:g} years) - this is too much!"
                    )

        temperatures = ts["data"].reset_index(drop=True)
        temperatures.index = pd.Index(range(1, 13), name="Month")
        if ts["scenario_type"] == "relative":
            target_tmin = monthly["Tmin"] + temperatures["Tmin"]
            target_tmax = monthly["Tmax"] + temperatures["Tmax"]
        elif ts["scenario_type"] == "absolute":
            target_tmin = temperatures["Tmin"].astype(float)
            target_tmax = temperatures["Tmax"].astype(float)
            if warn_me:
                warnings.warn(
                    "Absolute temperature scenario specified - calibration weather record only used for "
                    "simulating temperature variation, but not for the means",
                    stacklevel=2,
                )
        else:
            raise ValueError("scenario_type must be either 'relative' or 'absolute'")

        if (target_tmin > target_tmax).any():
            raise ValueError("temperature scenario produces monthly Tmin values greater than Tmax values")

        rng = np.random.default_rng(seed)
        rows = sim_dates.copy()
        generated_tmin = np.empty(len(rows), dtype=float)
        generated_tmax = np.empty(len(rows), dtype=float)
        for month in range(1, 13):
            month_mask = rows["Month"].astype(int).eq(month).to_numpy()
            count = int(month_mask.sum())
            pool = calibration.loc[calibration["Month"].astype(int).eq(month)]
            choices = rng.integers(0, len(pool), size=count)
            sampled = pool.iloc[choices]
            target_mid = (float(target_tmin.loc[month]) + float(target_tmax.loc[month])) / 2
            target_range = float(target_tmax.loc[month]) - float(target_tmin.loc[month])
            day_mid = target_mid + sampled["mid_anomaly"].to_numpy(dtype=float)
            day_range = target_range * sampled["range_ratio"].to_numpy(dtype=float)
            generated_tmin[month_mask] = day_mid - day_range / 2
            generated_tmax[month_mask] = day_mid + day_range / 2

        rows["Tmin"] = generated_tmin
        rows["Tmax"] = generated_tmax
        rows["Year"] = rows["Year"].astype(int)
        rows["Month"] = rows["Month"].astype(int)
        rows["Day"] = rows["Day"].astype(int)
        rows["YEARMODA"] = rows["YEARMODA"].astype(int)
        sim_out[name] = rows[["YEARMODA", "DATE", "Year", "Month", "Day", "Tmin", "Tmax"]]

    return sim_out


def gen_rel_change_scenario(
    downloaded_list: dict[str, pd.DataFrame],
    scenarios: Sequence[int] = (2050, 2085),
    reference_period: Sequence[int] = tuple(range(1986, 2015)),
    future_window_width: int = 30,
) -> pd.DataFrame:
    """Generate relative climate change scenarios based on extracted CMIP6 data.

    Translates R ``gen_rel_change_scenario``.
    """
    hist_keys = [k for k in downloaded_list.keys() if "historical" in k]
    models_with_baseline = [k.replace("historical_", "") for k in hist_keys]

    all_results = []
    ref_year_median = np.median(reference_period)

    for m in models_with_baseline:
        m_hist = downloaded_list[f"historical_{m}"]
        # Aggregate historical means by location, Month, and variable
        m_hist_long = m_hist.melt(
            id_vars=["location", "Month"], value_vars=["Tmin", "Tmax"]
        )
        m_hist_mean = (
            m_hist_long.groupby(["location", "Month", "variable"])["value"]
            .mean()
            .reset_index()
        )

        # Find future scenarios for this model
        future_keys = [
            k for k in downloaded_list.keys() if m in k and "historical" not in k
        ]

        for fk in future_keys:
            m_future_full = downloaded_list[fk]
            ssp = m_future_full["ssp"].iloc[0] if "ssp" in m_future_full.columns else fk

            for t in scenarios:
                t_upper = t + (future_window_width // 2)
                t_lower = t - (future_window_width // 2)

                m_future_window = m_future_full[
                    (m_future_full["Year"] >= t_lower) & (m_future_full["Year"] <= t_upper)
                ]
                if m_future_window.empty:
                    continue

                m_future_long = m_future_window.melt(
                    id_vars=["location", "Month"], value_vars=["Tmin", "Tmax"]
                )
                m_future_mean = (
                    m_future_long.groupby(["location", "Month", "variable"])["value"]
                    .mean()
                    .reset_index()
                )

                m_change = pd.merge(
                    m_future_mean,
                    m_hist_mean,
                    on=["location", "Month", "variable"],
                    suffixes=(".future", ".hist"),
                )
                m_change["change"] = m_change["value.future"] - m_change["value.hist"]

                # Pivot back to wide format (Tmin, Tmax columns)
                m_change_wide = m_change.pivot(
                    index=["location", "Month"], columns="variable", values="change"
                ).reset_index()

                m_change_wide["scenario"] = ssp
                m_change_wide["start_year"] = t_lower
                m_change_wide["end_year"] = t_upper
                m_change_wide["scenario_year"] = t
                m_change_wide["reference_year"] = ref_year_median
                m_change_wide["scenario_type"] = "relative"
                m_change_wide["labels"] = m

                all_results.append(m_change_wide)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def convert_scen_information(
    scenario_object: Any, give_structure: bool = True
) -> pd.DataFrame | dict[str, Any]:
    """Convert list of change scenarios to DataFrame or vice versa.

    Translates R ``convert_scen_information``.
    """
    if isinstance(scenario_object, pd.DataFrame):
        # df_to_list
        # split by location, scenario (ssp), labels (gcm), scenario_year
        # In R: split(scenario_object, f = ~ location + scenario + labels + scenario_year)
        group_cols = ["location", "scenario", "labels", "scenario_year"]
        # Ensure all columns exist
        for col in group_cols:
            if col not in scenario_object.columns:
                scenario_object[col] = np.nan

        scenario_list = {}
        for keys, group in scenario_object.groupby(group_cols, dropna=False):
            if len(group) != 12:
                continue

            loc, ssp, gcm, year = keys
            # Name scheme: Location.SSP.GCM.Timepoint
            name = f"{loc}.{ssp}.{gcm}.{year}"

            possible_vars = ["Tmin", "Tmax", "Prec"]
            present_vars = [v for v in possible_vars if v in group.columns]

            scen_dict = {
                "data": group[present_vars].reset_index(drop=True),
                "scenario": group["scenario"].iloc[0],
                "start_year": group["start_year"].iloc[0]
                if "start_year" in group.columns
                else np.nan,
                "end_year": group["end_year"].iloc[0]
                if "end_year" in group.columns
                else np.nan,
                "scenario_year": group["scenario_year"].iloc[0],
                "reference_year": group["reference_year"].iloc[0]
                if "reference_year" in group.columns
                else np.nan,
                "scenario_type": group["scenario_type"].iloc[0]
                if "scenario_type" in group.columns
                else np.nan,
                "labels": group["labels"].iloc[0],
            }
            scenario_list[name] = scen_dict

        if not give_structure:
            return scenario_list

        # Structured list (nested dict)
        # 1) Location 2) SSP 3) GCM 4) Timepoint
        structured = {}
        for name, scen in scenario_list.items():
            parts = name.split(".")
            loc, ssp, gcm, year = parts
            if loc not in structured:
                structured[loc] = {}
            if ssp not in structured[loc]:
                structured[loc][ssp] = {}
            if gcm not in structured[loc][ssp]:
                structured[loc][ssp][gcm] = {}
            structured[loc][ssp][gcm][year] = scen
        return structured

    else:
        # list_to_df
        # We need to flatten the nested dict or handle flat dict
        all_dfs = []

        def flatten(d: Any, path: list[str]) -> None:
            if (
                isinstance(d, dict)
                and "data" in d
                and isinstance(d["data"], pd.DataFrame)
            ):
                df = d["data"].copy()
                df["scenario"] = d.get("scenario", np.nan)
                df["start_year"] = d.get("start_year", np.nan)
                df["end_year"] = d.get("end_year", np.nan)
                df["scenario_year"] = d.get("scenario_year", np.nan)
                df["reference_year"] = d.get("reference_year", np.nan)
                df["scenario_type"] = d.get("scenario_type", np.nan)
                df["labels"] = d.get("labels", np.nan)
                if path:
                    df["location"] = path[0]
                all_dfs.append(df)
            elif isinstance(d, dict):
                for k, v in d.items():
                    flatten(v, path + [str(k)])

        flatten(scenario_object, [])
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)


def ordered_climate_list(strings: Any, file_extension: str | None = None) -> list[str] | float:
    """Sort files so that numbers are in ascending sequence.

    Recognizes shared leading and trailing symbols around the numeric part of
    strings and sorts according to the embedded numbers.
    Translates R ``ordered_climate_list``.

    Parameters
    ----------
    strings : Any
        Vector of character strings to be sorted.
    file_extension : str, optional
        Extension or trailing string to filter by.

    Returns
    -------
    list or float
        Sorted list of strings, or ``np.nan`` if no matches or empty input.
    """
    if file_extension:
        stringlist = select_by_file_extension(strings, file_extension)
    else:
        stringlist = as_list(strings)

    if stringlist is None or (isinstance(stringlist, float) and np.isnan(stringlist)):
        return np.nan

    if not stringlist:
        return []

    # R: numbers<-extract_differences_between_characters(stringlist)
    # R: stringlist<-stringlist[order(as_numeric(numbers))]
    diffs = extract_differences_between_characters(stringlist)
    if isinstance(diffs, float) and np.isnan(diffs):
        # No differences found, just return sorted strings
        return sorted([str(s) for s in stringlist])

    try:
        # Convert differences to float for numeric sorting
        numeric_vals = [float(d) for d in diffs]
        # Sort stringlist based on these numeric values
        zipped = sorted(zip(numeric_vals, stringlist))
        return [s for _, s in zipped]
    except (ValueError, TypeError):
        # Fallback to string sorting if numeric conversion fails
        return sorted([str(s) for s in stringlist])


def save_temperature_scenarios(
    generated_temperatures: Any, path: str | Path, prefix: str
) -> None:
    """Save temperature scenarios to CSV files.

    Translates R ``save_temperature_scenarios``.

    Parameters
    ----------
    generated_temperatures : Any
        List of temperature scenarios (DataFrames) to save.
    path : str or Path
        Directory where files will be written.
    prefix : str
        Prefix for the filenames.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(generated_temperatures, dict):
        items = generated_temperatures.items()
    elif isinstance(generated_temperatures, (list, tuple)):
        items = enumerate(generated_temperatures, 1)
    else:
        # If it's a single DataFrame, wrap it
        items = [(1, generated_temperatures)]

    for i, (name, df) in enumerate(items, 1):
        if not isinstance(df, pd.DataFrame):
            continue
        filename = f"{prefix}_{i}_{name}.csv"
        df.to_csv(path / filename, index=False)


def load_temperature_scenarios(path: str | Path, prefix: str) -> dict[str, pd.DataFrame]:
    """Load temperature scenarios from CSV files.

    Translates R ``load_temperature_scenarios``.

    Parameters
    ----------
    path : str or Path
        Directory where files are located.
    prefix : str
        Prefix for the filenames.

    Returns
    -------
    dict
        Dictionary of DataFrames.
    """
    path = Path(path)
    if not path.is_dir():
        return {}

    files = [f for f in path.iterdir() if f.is_file() and f.name.startswith(prefix)]

    scenario_info = []
    for f in files:
        # name is prefix_num_name.csv
        # remove prefix and leading underscore
        rest = f.name[len(prefix) + 1 :]
        if "_" not in rest:
            continue
        num_part, name_part = rest.split("_", 1)
        try:
            num = int(num_part)
            # name_part still has .csv
            name = name_part.rsplit(".", 1)[0]
            scenario_info.append({"num": num, "name": name, "file": f})
        except ValueError:
            continue

    # Sort by number
    scenario_info.sort(key=lambda x: x["num"])

    output = {}
    for info in scenario_info:
        output[info["name"]] = pd.read_csv(info["file"])

    return output


def load_climate_wizard_scenarios(
    path: str | Path, prefix: str
) -> dict[str, dict[str, Any]]:
    """Load ClimateWizard scenarios from CSV files.

    Translates R ``load_ClimateWizard_scenarios``.
    """
    scens = load_temperature_scenarios(path, prefix)
    output = {}
    for name, df in scens.items():
        # R expects certain columns to be present if it's a CW scenario
        # In R: output[[o]]<-list(data=data.frame(Tmin=output[[o]]$data.Tmin, Tmax=output[[o]]$data.Tmax), ...)
        # This assumes the CSV has columns like data.Tmin, scenario, etc.
        # Or it might just be the standard scenario structure.

        # Try to reconstruct the scenario object
        res = {
            "data": pd.DataFrame(),
            "scenario": "NA",
            "start_year": np.nan,
            "end_year": np.nan,
            "scenario_year": np.nan,
            "reference_year": np.nan,
            "scenario_type": "NA",
            "labels": name,
        }

        if "data.Tmin" in df.columns and "data.Tmax" in df.columns:
            res["data"] = pd.DataFrame(
                {"Tmin": df["data.Tmin"], "Tmax": df["data.Tmax"]}
            )
        elif "Tmin" in df.columns and "Tmax" in df.columns:
            res["data"] = df[["Tmin", "Tmax"]]

        for field in [
            "scenario",
            "start_year",
            "end_year",
            "scenario_year",
            "reference_year",
            "scenario_type",
            "labels",
        ]:
            if field in df.columns:
                val = df[field].iloc[0]
                res[field] = val

        output[name] = res

    return output


def get_climate_wizard_data(
    coordinates: Sequence[float],
    scenario: str,
    start_year: int,
    end_year: int,
    baseline: Sequence[int] | None = (1950, 2005),
    metric: str = "monthly_min_max_temps",
    gcms: str | Sequence[str] = "all",
    temperature_generation_scenarios: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Extract climate data from the ClimateWizard database.

    Translates R ``getClimateWizardData``.
    Note: This function requires an active internet connection and depends on
    the availability of the ClimateWizard API.
    """
    # Validation logic similar to R
    if len(coordinates) != 2:
        raise ValueError("coordinates must be of length 2")

    valid_scenarios = ["historical", "rcp45", "rcp85"]
    if scenario not in valid_scenarios:
        raise ValueError(f"scenario must be one of {valid_scenarios}")

    if baseline is not None:
        if len(baseline) != 2:
            raise ValueError("baseline must be a sequence of length 2")
        if baseline[1] > 2005:
            warnings.warn("baseline period cannot end after 2005", stacklevel=2)
        if baseline[0] < 1950:
            warnings.warn("baseline period cannot begin before 1950", stacklevel=2)
        if (baseline[1] - baseline[0]) < 20:
            warnings.warn("baseline period must span at least 20 years", stacklevel=2)

    all_available_gcms = [
        "bcc-csm1-1",
        "BNU-ESM",
        "CanESM2",
        "CESM1-BGC",
        "MIROC-ESM",
        "CNRM-CM5",
        "ACCESS1-0",
        "CSIRO-Mk3-6-0",
        "GFDL-CM3",
        "GFDL-ESM2G",
        "GFDL-ESM2M",
        "inmcm4",
        "IPSL-CM5A-LR",
        "IPSL-CM5A-MR",
        "CCSM4",
    ]

    if isinstance(gcms, str):
        if gcms == "all":
            selected_gcms = all_available_gcms
        else:
            selected_gcms = [gcms]
    else:
        selected_gcms = list(gcms)

    for g in selected_gcms:
        if g not in all_available_gcms:
            warnings.warn(f"GCM {g} might not be available in ClimateWizard", stacklevel=2)

    # In a real implementation, we would use 'requests' to call the API.
    # Since we want to avoid new dependencies if possible, and ClimateWizard
    # is often unstable, we provide a structured warning and a partial implementation
    # that would work if 'requests' was present.

    try:
        import requests
    except ImportError:
        warnings.warn(
            "The 'requests' library is required for get_climate_wizard_data. "
            "Returning a placeholder.",
            stacklevel=2,
        )
        return placeholder_record(
            "get_climate_wizard_data",
            coordinates=coordinates,
            scenario=scenario,
            start_year=start_year,
            end_year=end_year,
            baseline=baseline,
            metric=metric,
            gcms=selected_gcms,
        )

    # Partial implementation of API call logic
    base_url = "http://climatewizard.ccafs-climate.org/service"
    results_data = []

    for gcm in selected_gcms:
        gcm_data = {"GCM": gcm}
        available = True

        metrics_to_fetch = []
        if metric == "monthly_min_max_temps":
            for ext in ["tasmin", "tasmax"]:
                for i in range(1, 13):
                    metrics_to_fetch.append(f"{ext}{i}")
        elif metric == "precipitation":
            for i in range(1, 13):
                metrics_to_fetch.append(f"pr{i}")
        elif metric == "monthly_tmean":
            for i in range(1, 13):
                metrics_to_fetch.append(f"tas{i}")
        else:
            metrics_to_fetch = [metric]

        for m in metrics_to_fetch:
            params = {
                "lat": coordinates[1],
                "lon": coordinates[0],
                "index": m,
                "scenario": scenario,
                "gcm": gcm,
                "range": f"{start_year}-{end_year}",
                "avg": "true",
            }
            if baseline is not None:
                params["baseline"] = f"{baseline[0]}-{baseline[1]}"

            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    available = False
                    break
                # data["values"] is typically [[year_range], [value]]
                val = float(data["values"][1][0])
                gcm_data[m] = val
            except Exception as e:
                warnings.warn(f"Failed to fetch {m} for {gcm}: {e}", stacklevel=2)
                available = False
                break

        if available:
            results_data.append(gcm_data)

    if not results_data:
        return {}

    metric_gcms = pd.DataFrame(results_data)

    output = {
        "data": metric_gcms,
        "scenario": scenario,
        "start_year": start_year,
        "end_year": end_year,
        "scenario_year": (start_year + end_year) / 2,
        "reference_year": np.nan if baseline is None else (baseline[0] + baseline[1]) / 2,
        "scenario_type": "absolute" if baseline is None else "relative",
        "labels": [],
    }

    if metric == "monthly_min_max_temps" and temperature_generation_scenarios:
        outputs = {}
        for _, row in metric_gcms.iterrows():
            gcm_name = row["GCM"]
            single_output = output.copy()
            single_output["data"] = pd.DataFrame(
                {
                    "Tmin": [row[f"tasmin{i}"] for i in range(1, 13)],
                    "Tmax": [row[f"tasmax{i}"] for i in range(1, 13)],
                }
            )
            single_output["labels"] = gcm_name
            outputs[gcm_name] = single_output
        return outputs

    return output


def get_climate_wizard_scenarios(
    coordinates: Sequence[float],
    scenarios: Sequence[str],
    start_years: Sequence[int],
    end_years: Sequence[int],
    baseline: Sequence[int] | None = (1950, 2005),
    metric: str = "monthly_min_max_temps",
    gcms: str | Sequence[str] = "all",
) -> list[dict[str, Any]]:
    """Extract multiple scenarios from the ClimateWizard database.

    Translates R ``getClimateWizard_scenarios``.
    """
    if len(start_years) != len(end_years) or len(scenarios) != len(start_years):
        raise ValueError("scenarios, start_years, and end_years must have same length")

    results = []
    for i in range(len(scenarios)):
        results.append(
            get_climate_wizard_data(
                coordinates=coordinates,
                scenario=scenarios[i],
                start_year=start_years[i],
                end_year=end_years[i],
                baseline=baseline,
                metric=metric,
                gcms=gcms,
            )
        )
    return results


def extract_temperatures_from_grids(
    coordinates: Sequence[float],
    grid_format: str,
    grid_specifications: dict[str, Any],
    scenario_year: int | None = None,
    reference_year: int | None = None,
    scenario_type: str | None = None,
    labels: Any = None,
    temperature_check_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract temperature information from gridded dataset.

    Translates R ``extract_temperatures_from_grids``.
    Note: This function requires 'rasterio' or 'xarray' for grid extraction,
    which are not standard dependencies of chillPy.
    """
    from .weather import check_temperature_scenario

    if len(coordinates) < 2:
        raise ValueError("no useable coordinates provided")

    lon = float(coordinates[0])
    lat = float(coordinates[1])

    if grid_format not in ["AFRICLIM", "CCAFS", "WorldClim"]:
        raise ValueError(f"undefined grid format: {grid_format}")

    # This function usually involves unzipping files and using raster tools.
    # Since chillPy aims to avoid heavy dependencies, we provide the logic
    # but guard the heavy imports.

    try:
        import rasterio
    except ImportError:
        warnings.warn(
            "The 'rasterio' library is required for extract_temperatures_from_grids. "
            "Returning a placeholder.",
            stacklevel=2,
        )
        return placeholder_record(
            "extract_temperatures_from_grids",
            coordinates=coordinates,
            grid_format=grid_format,
            grid_specifications=grid_specifications,
        )

    base_folder = Path(grid_specifications.get("base_folder", "."))
    min_file = base_folder / grid_specifications.get("minfile", "")
    max_file = base_folder / grid_specifications.get("maxfile", "")

    if not min_file.exists() or not max_file.exists():
        raise FileNotFoundError("grid files not found")

    # In R, it unzips and looks for .tif or .asc files
    # Here we would implement the unzipping and raster extraction.

    temperatures = pd.DataFrame(
        {"Tmin": np.full(12, np.nan), "Tmax": np.full(12, np.nan)},
        index=range(1, 13)
    )

    import zipfile
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Unzip min and max files
        with zipfile.ZipFile(min_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_path / "min")
        with zipfile.ZipFile(max_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_path / "max")

        def get_sorted_files(path, ext):
            files = list(path.glob(f"**/*.{ext}"))
            # Sort by number in filename if possible
            import re
            def extract_number(f):
                s = re.findall(r'\d+', f.name)
                return int(s[0]) if s else 0
            return sorted(files, key=extract_number)

        ext = "tif" if grid_format in ["AFRICLIM", "WorldClim"] else "asc"
        min_rasters = get_sorted_files(tmp_path / "min", ext)
        max_rasters = get_sorted_files(tmp_path / "max", ext)

        if len(min_rasters) >= 12 and len(max_rasters) >= 12:
            for i in range(12):
                with rasterio.open(min_rasters[i]) as src:
                    # rasterio uses (lon, lat) for sample
                    val = list(src.sample([(lon, lat)]))[0][0]
                    # WorldClim and AFRICLIM often use scale factor 10
                    if grid_format in ["AFRICLIM", "WorldClim"]:
                        val /= 10.0
                    temperatures.loc[i+1, "Tmin"] = val
                
                with rasterio.open(max_rasters[i]) as src:
                    val = list(src.sample([(lon, lat)]))[0][0]
                    if grid_format in ["AFRICLIM", "WorldClim"]:
                        val /= 10.0
                    temperatures.loc[i+1, "Tmax"] = val
        else:
            warnings.warn(
                f"Expected 12 raster files, found {len(min_rasters)} min and {len(max_rasters)} max files.",
                stacklevel=2
            )

    out_scen = {
        "data": temperatures.reset_index(drop=True),
        "scenario_year": scenario_year,
        "reference_year": reference_year,
        "scenario_type": scenario_type,
        "labels": labels,
    }

    check_args = {
        "n_intervals": 12,
        "check_scenario_type": True,
        "scenario_check_thresholds": (-5, 10),
        "update_scenario_type": True,
        "warn_me": True,
    }
    if temperature_check_args:
        check_args.update(temperature_check_args)

    return check_temperature_scenario(
        out_scen,
        n_intervals=check_args["n_intervals"],
        check_scenario_type=check_args["check_scenario_type"],
        scenario_check_thresholds=check_args["scenario_check_thresholds"],
        update_scenario_type=check_args["update_scenario_type"],
        warn_me=check_args["warn_me"],
    )


def extract_cmip6_data(
    stations: pd.DataFrame,
    variable: Sequence[str] = ("Tmin", "Tmax"),
    download_path: str | Path = "cmip6_downloaded",
    keep_downloaded: bool = True,
) -> dict[str, pd.DataFrame]:
    """Unpack and format downloaded CMIP6 data.

    Translates R ``extract_cmip6_data``.
    Note: This function requires 'xarray' and 'netcdf4' for NetCDF extraction,
    which are not standard dependencies of chillPy.
    """
    try:
        import xarray as xr
    except ImportError:
        warnings.warn(
            "The 'xarray' and 'netcdf4' libraries are required for extract_cmip6_data. "
            "Returning an empty record.",
            stacklevel=2,
        )
        return placeholder_record("extract_cmip6_data", stations=stations, data=[])

    download_path = Path(download_path)
    if not download_path.exists():
        raise FileNotFoundError(f"Download path {download_path} does not exist.")

    # NetCDF variable mapping
    var_map = {"Tmin": "tasmin", "Tmax": "tasmax", "Prec": "pr"}
    nc_vars = [var_map[v] for v in variable if v in var_map]

    # Find all subdirectories (areas)
    sub_dirs = [d for d in download_path.iterdir() if d.is_dir()]
    if not sub_dirs:
        sub_dirs = [download_path]

    all_weather_combined = {}

    for area_dir in sub_dirs:
        # Unzip files if present
        zip_files = list(area_dir.glob("*.zip"))
        for zf in zip_files:
            import zipfile

            with zipfile.ZipFile(zf, "r") as zip_ref:
                zip_ref.extractall(area_dir)

        # Identify all .nc files
        nc_files = list(area_dir.glob("*.nc"))
        if not nc_files:
            continue

        # Group files by scenario and GCM
        # Filename format: var_scenario_model_frequency_area.nc
        # e.g., tasmax_ssp126_AWI-CM-1-1-MR_monthly_55_5.5_47_15.1.nc
        file_groups = {}
        for ncf in nc_files:
            parts = ncf.stem.split("_")
            if len(parts) < 4:
                continue
            # parts[0] is variable (tasmin, tasmax, pr)
            # parts[1] is scenario (ssp126, historical, etc.)
            # parts[2] is model
            group_key = f"{parts[1]}_{parts[2]}"
            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(ncf)

        for group_key, group_files in file_groups.items():
            # Check if we have all requested variables for this group
            group_vars = {f.stem.split("_")[0] for f in group_files}
            if not all(v in group_vars for v in nc_vars):
                continue

            # Extract for each station
            group_dfs = []
            for i, station in stations.iterrows():
                st_name = station["station_name"]
                st_lon = station["longitude"]
                st_lat = station["latitude"]

                st_ds_list = []
                for ncf in group_files:
                    var_name = ncf.stem.split("_")[0]
                    if var_name not in nc_vars:
                        continue
                    
                    with xr.open_dataset(ncf) as ds:
                        # Find nearest point
                        # Longitude might be 0-360 or -180-180
                        if "lon" in ds.coords:
                            lon_name = "lon"
                        elif "longitude" in ds.coords:
                            lon_name = "longitude"
                        else:
                            continue
                        
                        if "lat" in ds.coords:
                            lat_name = "lat"
                        elif "latitude" in ds.coords:
                            lat_name = "latitude"
                        else:
                            continue

                        # Adjust longitude if needed
                        ds_lon = ds[lon_name].values
                        target_lon = st_lon
                        if ds_lon.max() > 180 and st_lon < 0:
                            target_lon += 360
                        elif ds_lon.min() < 0 and st_lon > 180:
                            target_lon -= 360

                        st_var = ds.sel({lat_name: st_lat, lon_name: target_lon}, method="nearest")
                        st_df = st_var.to_dataframe()
                        # Clean up index and add info
                        st_df = st_df.reset_index()
                        st_df["location"] = st_name
                        st_df["lat"] = st_lat
                        st_df["lon"] = st_lon
                        st_df["model"] = group_key.split("_")[1]
                        st_df["ssp"] = group_key.split("_")[0]
                        st_ds_list.append(st_df)

                if st_ds_list:
                    from functools import reduce

                    merged_st = reduce(
                        lambda left, right: pd.merge(
                            left,
                            right,
                            on=["time", "lat", "lon", "location", "model", "ssp"],
                            how="outer",
                        ),
                        st_ds_list,
                    )
                    group_dfs.append(merged_st)

            if group_dfs:
                combined_group = pd.concat(group_dfs, ignore_index=True)
                # Formatting: Date, Year, Month, Day, Tmin, Tmax, etc.
                if "time" in combined_group.columns:
                    combined_group["Date"] = pd.to_datetime(combined_group["time"]).dt.date
                    combined_group["Year"] = pd.to_datetime(combined_group["time"]).dt.year
                    combined_group["Month"] = pd.to_datetime(combined_group["time"]).dt.month
                    combined_group["Day"] = pd.to_datetime(combined_group["time"]).dt.day

                # Unit conversion (Kelvin to Celsius)
                if "tasmax" in combined_group.columns:
                    combined_group["Tmax"] = (combined_group["tasmax"] - 273.15).round(4)
                if "tasmin" in combined_group.columns:
                    combined_group["Tmin"] = (combined_group["tasmin"] - 273.15).round(4)
                if "pr" in combined_group.columns:
                    combined_group["Prec"] = (combined_group["pr"] * 86400).round(4)

                # Select final columns
                final_cols = [
                    "Date",
                    "Year",
                    "Month",
                    "Day",
                    "lat",
                    "lon",
                    "location",
                    "model",
                    "ssp",
                ] + list(variable)
                available_cols = [c for c in final_cols if c in combined_group.columns]
                
                if group_key not in all_weather_combined:
                    all_weather_combined[group_key] = combined_group[available_cols]
                else:
                    all_weather_combined[group_key] = pd.concat(
                        [all_weather_combined[group_key], combined_group[available_cols]],
                        ignore_index=True,
                    )

        # Cleanup unzipped .nc files if requested
        if not keep_downloaded:
            for ncf in nc_files:
                ncf.unlink()

    return all_weather_combined


def download_cmip6_ecmwfr(
    scenarios: str | Sequence[str],
    area: Sequence[float],
    model: str | Sequence[str] = "default",
    service: str = "cds",
    frequency: str = "monthly",
    variable: Sequence[str] = ("Tmin", "Tmax"),
    year_start: int = 2015,
    year_end: int = 2100,
    month: Sequence[int] = tuple(range(1, 13)),
    path_download: str | Path = "cmip6_downloaded",
    user: str | None = None,
    key: str | None = None,
) -> None:
    """Download CMIP6 data via the cdsapi.

    Translates R ``download_cmip6_ecmwfr``.
    Note: This function requires 'cdsapi', which is not a standard dependency of chillPy.
    """
    try:
        import cdsapi
    except ImportError:
        warnings.warn(
            "The 'cdsapi' library is required for download_cmip6_ecmwfr.",
            stacklevel=2,
        )
        not_implemented("download_cmip6_ecmwfr")
        return

    scenarios = as_list(scenarios)
    variable = as_list(variable)
    month = as_list(month)

    # Translate variable names to CDS API format
    var_map = {
        "Tmin": "daily_minimum_near_surface_air_temperature",
        "Tmax": "daily_maximum_near_surface_air_temperature",
        "Prec": "precipitation",
    }
    cds_vars = [var_map[v] for v in variable if v in var_map]

    # Translate scenario names if needed (e.g. ssp126 -> ssp1_2_6)
    scen_map = {
        "ssp126": "ssp1_2_6",
        "ssp245": "ssp2_4_5",
        "ssp370": "ssp3_7_0",
        "ssp585": "ssp5_8_5",
    }
    cds_scens = [scen_map.get(s, s) for s in scenarios]

    # Model handling (simplified)
    if model == "default":
        # In reality, R chillR has a long list of default models.
        # Here we might want to just use a few or document that "default" isn't fully ported.
        models = ["access_cm2", "awi_cm_1_1_mr", "canesm5"]
    elif model == "all":
        # Placeholder for 'all'
        models = ["access_cm2", "awi_cm_1_1_mr", "canesm5"] 
    else:
        models = as_list(model)

    path_download = Path(path_download)
    area_folder = path_download / "_".join(map(str, area))
    area_folder.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client(url=None, key=key, user=user) if (key or user) else cdsapi.Client()

    for scen in cds_scens:
        for mod in models:
            for v in cds_vars:
                # Map back to short name for filename
                short_v = [k for k, val in var_map.items() if val == v][0]
                fname = f"{short_v.lower()}_{scen}_{mod}_{frequency}_{'_'.join(map(str, area))}.zip"
                target = area_folder / fname

                if target.exists():
                    print(f"File {fname} already exists, skipping.")
                    continue

                print(f"Requesting {fname}...")
                try:
                    c.retrieve(
                        "projections-cmip6",
                        {
                            "format": "zip",
                            "temporal_resolution": frequency,
                            "experiment": scen,
                            "variable": v,
                            "model": mod,
                            "year": [str(y) for y in range(year_start, year_end + 1)],
                            "month": [f"{m:02d}" for m in month],
                            "area": area,
                        },
                        str(target),
                    )
                except Exception as e:
                    warnings.warn(f"Failed to download {fname}: {e}")

def download_baseline_cmip6_ecmwfr(
    area: Sequence[float],
    model: str | Sequence[str] = "match_downloaded",
    service: str = "cds",
    frequency: str = "monthly",
    variable: Sequence[str] = ("Tmin", "Tmax"),
    year_start: int = 1986,
    year_end: int = 2014,
    month: Sequence[int] = tuple(range(1, 13)),
    path_download: str | Path = "cmip6_downloaded",
    user: str | None = None,
    key: str | None = None,
) -> None:
    """Download historical CMIP6 data via the cdsapi.

    Translates R ``download_baseline_cmip6_ecmwfr``.
    """
    if model == "match_downloaded":
        path_download = Path(path_download)
        area_str = "_".join(map(str, area))
        area_folder = path_download / area_str
        if not area_folder.exists():
            raise FileNotFoundError(f"No downloaded data found for area {area_str}")
        
        # Identify models from existing files
        existing_files = list(area_folder.glob("*.zip")) + list(area_folder.glob("*.nc"))
        models = set()
        for f in existing_files:
            parts = f.stem.split("_")
            if len(parts) >= 3 and parts[1] != "historical":
                models.add(parts[2])
        model = list(models) if models else "default"

    download_cmip6_ecmwfr(
        scenarios="historical",
        area=area,
        model=model,
        service=service,
        frequency=frequency,
        variable=variable,
        year_start=year_start,
        year_end=year_end,
        month=month,
        path_download=path_download,
        user=user,
        key=key,
    )


load_ClimateWizard_scenarios = load_climate_wizard_scenarios
getClimateWizardData = get_climate_wizard_data
getClimateWizard_scenarios = get_climate_wizard_scenarios
