"""Generate lightweight example datasets and getting-started notebooks."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DATA = EXAMPLES / "data"


def make_synthetic_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2015-01-01", "2022-12-31", freq="D")
    rng = np.random.default_rng(42)
    seasonal = np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 90) / 365.25)
    winter = np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 15) / 365.25)
    trend = (dates.year.to_numpy() - dates.year.min()) * 0.08
    noise = rng.normal(0, 0.35, len(dates))
    tmean = 10.5 + 9.0 * seasonal + 1.2 * winter + trend + noise
    amplitude = 6.5 + 1.5 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    weather = pd.DataFrame(
        {
            "Year": dates.year,
            "Month": dates.month,
            "Day": dates.day,
            "Tmin": np.round(tmean - amplitude / 2, 2),
            "Tmax": np.round(tmean + amplitude / 2, 2),
        }
    )
    weather.to_csv(DATA / "synthetic_daily_weather.csv", index=False)

    bloom_rows = []
    for year, group in weather.groupby("Year"):
        jan_mar = group.loc[group["Month"].isin([1, 2, 3])]
        warm_signal = ((jan_mar["Tmin"] + jan_mar["Tmax"]) / 2).mean()
        bloom = int(round(112 - 1.5 * (warm_signal - 7.5) + (year - 2015) * 0.4))
        bloom_rows.append({"Year": int(year), "pheno": bloom})
    pd.DataFrame(bloom_rows).to_csv(DATA / "synthetic_phenology.csv", index=False)

    readme = DATA / "README.md"
    readme.write_text(
        "# Example data\n\n"
        "The CSV files in this folder are deterministic synthetic data generated "
        "for the chillPy notebooks. They are not observations and should only be "
        "used for examples and smoke tests.\n",
        encoding="utf-8",
    )


_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"{prefix}-{_CELL_COUNTER:03d}"


def md(source: str) -> dict:
    normalized = textwrap.dedent(source).strip()
    title = re.sub(r"[^a-z0-9]+", "-", normalized.splitlines()[0].lower()).strip("-")[:20]
    return {"cell_type": "markdown", "id": _cell_id(title or "markdown"), "metadata": {}, "source": normalized.splitlines(True)}


def code(source: str) -> dict:
    normalized = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "id": _cell_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalized.splitlines(True),
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def make_notebooks() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)

    write_notebook(
        EXAMPLES / "01_getting_started_weather_and_chill.ipynb",
        [
            md(
                """
                # Getting started: weather records and chill metrics

                This notebook starts with a small synthetic daily weather record, checks and fixes the
                date/temperature structure, converts daily extremes to hourly temperatures, and calculates
                standard chill and heat metrics. The data in `examples/data/` are synthetic and deterministic.
                """
            ),
            code(
                """
                from pathlib import Path

                import matplotlib.pyplot as plt
                import pandas as pd

                from chillPy import (
                    check_temperature_record,
                    chilling,
                    daily_chill,
                    fix_weather,
                    make_chill_plot,
                    stack_hourly_temps,
                )

                DATA = Path("examples/data")
                weather = pd.read_csv(DATA / "synthetic_daily_weather.csv")
                weather.head()
                """
            ),
            md("Create a small imperfection so the completion and interpolation workflow has something to repair."),
            code(
                """
                demo_weather = weather.query("2018 <= Year <= 2021").copy()
                demo_weather = demo_weather.drop(demo_weather.query("Year == 2019 and Month == 2 and Day == 10").index)
                demo_weather.loc[
                    (demo_weather["Year"] == 2020) & (demo_weather["Month"] == 1) & (demo_weather["Day"] == 15),
                    "Tmin",
                ] = pd.NA

                before = check_temperature_record(demo_weather)
                fixed = fix_weather(demo_weather, start_year=2018, end_year=2021, end_at_present=False)
                after = check_temperature_record(fixed["weather"])

                {
                    "before_valid": before["valid"],
                    "before_error": before["error"],
                    "after_valid": after["valid"],
                    "fixed_rows": len(fixed["weather"]),
                }
                """
            ),
            md("Convert fixed daily records to hourly temperatures and summarize seasonal chill/heat accumulation."),
            code(
                """
                hourly = stack_hourly_temps(fixed["weather"], latitude=42.0)
                chill_summary = chilling(hourly, start_jday=305, end_jday=60)
                chill_summary[["End_year", "Chilling_Hours", "Utah_Model", "Chill_portions", "GDH", "Perc_complete"]]
                """
            ),
            md("Daily chill is useful for plotting how metrics accumulate through the season."),
            code(
                """
                daily = daily_chill(hourly, running_mean=3)
                plot = make_chill_plot(
                    daily,
                    metrics=["Chilling_Hours", "Chill_Portions", "GDH"],
                    startdate=305,
                    enddate=80,
                    cumulative=True,
                    focusyears=[2020, 2021],
                )
                plot["figure"]
                """
            ),
        ],
    )

    write_notebook(
        EXAMPLES / "02_temperature_interpolation_and_gap_filling.ipynb",
        [
            md(
                """
                # Temperature interpolation and gap filling

                This notebook demonstrates two gap-filling layers: simple one-dimensional interpolation
                and the hourly temperature gap workflow that uses daily temperature-curve structure.
                """
            ),
            code(
                """
                from pathlib import Path

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd

                from chillPy import interpolate_gaps, interpolate_gaps_hourly, stack_hourly_temps

                DATA = Path("examples/data")
                daily = pd.read_csv(DATA / "synthetic_daily_weather.csv").query("Year == 2020 and Month == 1 and Day <= 12")
                hourly = stack_hourly_temps(daily, latitude=42.0)["hourtemps"]
                hourly.head()
                """
            ),
            code(
                """
                series = pd.Series([2.0, np.nan, np.nan, 8.0, np.nan, 10.0], name="example")
                filled = interpolate_gaps(series)
                pd.DataFrame({"original": series, "filled": filled["interp"], "was_missing": filled["missing"]})
                """
            ),
            md("Create realistic hourly gaps, including a multi-hour gap that crosses midnight."),
            code(
                """
                hourly_with_gaps = hourly.copy()
                gap_mask = (
                    ((hourly_with_gaps["Day"] == 4) & (hourly_with_gaps["Hour"].isin([10, 11, 12, 13])))
                    | ((hourly_with_gaps["Day"] == 6) & (hourly_with_gaps["Hour"].isin([22, 23])))
                    | ((hourly_with_gaps["Day"] == 7) & (hourly_with_gaps["Hour"].isin([0, 1, 2])))
                )
                hourly_with_gaps.loc[gap_mask, "Temp"] = np.nan

                filled_hourly = interpolate_gaps_hourly(
                    hourly_with_gaps,
                    latitude=42.0,
                    minimum_values_for_solving=12,
                    runn_mean_test_diff=999,
                )["weather"]

                filled_hourly.loc[gap_mask.to_numpy(), ["Year", "Month", "Day", "Hour", "Temp_measured", "Temp"]].head(12)
                """
            ),
            code(
                """
                comparison = filled_hourly.copy()
                comparison["timestamp"] = pd.to_datetime(
                    comparison[["Year", "Month", "Day"]].astype(int)
                ) + pd.to_timedelta(comparison["Hour"].astype(int), unit="h")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(comparison["timestamp"], comparison["Temp"], label="filled", color="black")
                ax.scatter(
                    comparison["timestamp"],
                    comparison["Temp_measured"],
                    label="measured",
                    color="tab:blue",
                    s=12,
                )
                ax.set_ylabel("Temperature (deg C)")
                ax.set_title("Hourly gap filling")
                ax.legend(frameon=False)
                fig.autofmt_xdate()
                fig
                """
            ),
        ],
    )

    write_notebook(
        EXAMPLES / "03_phenology_and_bloom_prediction.ipynb",
        [
            md(
                """
                # Phenology and bloom prediction

                This notebook combines hourly temperature-derived chill/heat metrics with simple bloom
                requirement scenarios, then compares predicted bloom dates with synthetic observations.
                """
            ),
            code(
                """
                from pathlib import Path

                import matplotlib.pyplot as plt
                import pandas as pd

                from chillPy import bloom_prediction2, chilling_hourtable, stack_hourly_temps

                DATA = Path("examples/data")
                weather = pd.read_csv(DATA / "synthetic_daily_weather.csv")
                pheno = pd.read_csv(DATA / "synthetic_phenology.csv")
                pheno
                """
            ),
            code(
                """
                hourly = stack_hourly_temps(weather.query("2015 <= Year <= 2022"), latitude=42.0)["hourtemps"]
                chill_table = chilling_hourtable(hourly, start_jday=305)

                predictions = bloom_prediction2(
                    chill_table,
                    chill_req=[38],
                    heat_req=[5200],
                    chill_model="Chill_Portions",
                    heat_model="GDH",
                    start_jday=305,
                )
                predictions.head()
                """
            ),
            code(
                """
                comparison = pheno.merge(
                    predictions[["Season", "Pheno_date"]].rename(columns={"Season": "Year", "Pheno_date": "predicted"}),
                    on="Year",
                    how="inner",
                )
                comparison["error_days"] = comparison["predicted"] - comparison["pheno"]
                comparison
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(comparison["Year"], comparison["pheno"], marker="o", label="observed")
                ax.plot(comparison["Year"], comparison["predicted"], marker="s", label="predicted")
                ax.set_ylabel("Bloom day of year")
                ax.set_title("Synthetic bloom-date prediction")
                ax.legend(frameon=False)
                fig
                """
            ),
        ],
    )

    write_notebook(
        EXAMPLES / "04_pls_analysis.ipynb",
        [
            md(
                """
                # PLS analysis

                This notebook runs a compact PLS phenology analysis on the synthetic multi-year weather
                record, extracts VIP/coefficient interpretation tables, and renders the PLS summary plot.
                """
            ),
            code(
                """
                from pathlib import Path

                import pandas as pd

                from chillPy import PLS_pheno, plot_pls, prepare_pls_plot_data

                DATA = Path("examples/data")
                weather = pd.read_csv(DATA / "synthetic_daily_weather.csv")
                pheno = pd.read_csv(DATA / "synthetic_phenology.csv")

                pls = PLS_pheno(
                    weather,
                    pheno,
                    split_month=6,
                    runn_mean=7,
                    ncomp_fix=1,
                    end_at_pheno_end=150,
                )
                pls["PLS_summary"].head()
                """
            ),
            code(
                """
                interpreted = prepare_pls_plot_data(pls, vip_threshold=0.8)
                interpreted["important_windows"].head(10)
                """
            ),
            code(
                """
                plot = plot_pls(pls, vip_threshold=0.8)
                plot["figure"]
                """
            ),
            code(
                """
                top = pls["PLS_summary"].assign(abs_coef=lambda df: df["Coef"].abs()).nlargest(8, "VIP")
                top[["Date", "JDay", "Coef", "VIP", "Tmean"]]
                """
            ),
        ],
    )

    write_notebook(
        EXAMPLES / "05_train_phenoflex_cherry_bloom_prediction.ipynb",
        [
            md(
                """
                # Train PhenoFlex-style cherry bloom models

                This notebook builds a compact synthetic cherry bloom dataset from deterministic
                hourly weather, fits PhenoFlex-style models with `phenologyFitter`, and compares
                observed and predicted bloom dates. The optimization is intentionally small so the
                notebook stays useful as a documentation example and as a smoke test.

                The data are synthetic: bloom dates are generated from a plausible PhenoFlex-GDH
                parameter set with small deterministic year-to-year observation offsets.
                """
            ),
            code(
                """
                from pathlib import Path

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd

                from chillPy import (
                    PhenoFlex_GDHwrapper,
                    PhenoFlex_fixedDynModelwrapper,
                    genSeasonList,
                    phenologyFitter,
                    stack_hourly_temps,
                )

                DATA = Path("examples/data")
                weather = pd.read_csv(DATA / "synthetic_daily_weather.csv")
                hourly = stack_hourly_temps(weather, latitude=42.0)["hourtemps"]
                years = list(range(2016, 2023))
                season_list = genSeasonList(hourly, mrange=(8, 6), years=years)

                pd.DataFrame(
                    {
                        "Year": years,
                        "records": [len(season) for season in season_list],
                        "first_jday": [season["JDay"].iloc[0] for season in season_list],
                        "last_jday": [season["JDay"].iloc[-1] for season in season_list],
                    }
                )
                """
            ),
            md(
                """
                Generate synthetic cherry bloom dates from a PhenoFlex-GDH parameter set. The
                deterministic offsets mimic observation noise and cultivar/site effects without
                hiding the known data-generating process.
                """
            ),
            code(
                """
                true_phenoflex_gdh = [
                    28.0,       # yc: chill requirement
                    75.0,       # zc: forcing requirement
                    0.55,       # s1: chill/heat interaction
                    25.0,       # Tu: optimum forcing temperature
                    3372.8,     # E0
                    9900.3,     # E1
                    6319.5,     # A0
                    5.939917e13,# A1
                    4.0,        # Tf
                    36.0,       # Tc
                    4.0,        # Tb
                    1.6,        # slope
                ]

                latent_bloom = np.array(
                    [PhenoFlex_GDHwrapper(season, true_phenoflex_gdh) for season in season_list]
                )
                observation_offsets = np.array([1.5, -2.0, 0.5, 1.0, -1.0, 2.0, -0.5])
                bloom = pd.DataFrame(
                    {
                        "Year": years,
                        "cultivar": "Synthetic Bing cherry",
                        "latent_model_jday": latent_bloom.round(1),
                        "observed_bloom_jday": np.round(latent_bloom + observation_offsets, 1),
                    }
                )
                bloom
                """
            ),
            md(
                """
                Fit a PhenoFlex-GDH model. To keep this small and identifiable, only the chill and
                heat requirements are estimated; the temperature-response constants are fixed at
                literature-like values.
                """
            ),
            code(
                """
                phenoflex_gdh_fit = phenologyFitter(
                    par_guess=[25.0, 70.0, 0.55, 25.0, 3372.8, 9900.3, 6319.5, 5.939917e13, 4.0, 36.0, 4.0, 1.6],
                    lower=[15.0, 50.0, 0.55, 25.0, 3372.8, 9900.3, 6319.5, 5.939917e13, 4.0, 36.0, 4.0, 1.6],
                    upper=[40.0, 95.0, 0.55, 25.0, 3372.8, 9900.3, 6319.5, 5.939917e13, 4.0, 36.0, 4.0, 1.6],
                    bloom_jdays=bloom["observed_bloom_jday"],
                    season_list=season_list,
                    modelfn=PhenoFlex_GDHwrapper,
                    control={"maxit": 25, "tol": 1e-3},
                )

                phenoflex_gdh_fit["residuals"].assign(Year=years)
                """
            ),
            md("Fit a simpler PhenoFlex-style fixed dynamic model against the same observations."),
            code(
                """
                fixed_dynamic_fit = phenologyFitter(
                    par_guess=[25.0, 70.0, 0.55, 25.0, 36.0, 4.0],
                    lower=[15.0, 50.0, 0.55, 25.0, 36.0, 4.0],
                    upper=[40.0, 95.0, 0.55, 25.0, 36.0, 4.0],
                    bloom_jdays=bloom["observed_bloom_jday"],
                    season_list=season_list,
                    modelfn=PhenoFlex_fixedDynModelwrapper,
                    control={"maxit": 25, "tol": 1e-3},
                )

                fixed_dynamic_fit["residuals"].assign(Year=years)
                """
            ),
            md("Compare model skill and collect the fitted parameter values."),
            code(
                """
                def score_row(name, fit):
                    return {
                        "model": name,
                        "rmse_days": fit["rmse"],
                        "objective": fit["objective"],
                        "parameters": np.array2string(fit["par"], precision=3, suppress_small=False),
                    }

                scores = pd.DataFrame(
                    [
                        score_row("PhenoFlex GDH", phenoflex_gdh_fit),
                        score_row("Fixed dynamic PhenoFlex", fixed_dynamic_fit),
                    ]
                )
                scores
                """
            ),
            code(
                """
                comparison = bloom[["Year", "observed_bloom_jday"]].copy()
                comparison["PhenoFlex GDH"] = phenoflex_gdh_fit["pbloomJDays"]
                comparison["Fixed dynamic PhenoFlex"] = fixed_dynamic_fit["pbloomJDays"]
                comparison
                """
            ),
            code(
                """
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

                axes[0].plot(comparison["Year"], comparison["observed_bloom_jday"], marker="o", label="observed")
                axes[0].plot(comparison["Year"], comparison["PhenoFlex GDH"], marker="s", label="PhenoFlex GDH")
                axes[0].plot(
                    comparison["Year"],
                    comparison["Fixed dynamic PhenoFlex"],
                    marker="^",
                    label="fixed dynamic",
                )
                axes[0].set_ylabel("Bloom day of year")
                axes[0].set_title("Cherry bloom predictions")
                axes[0].legend(frameon=False)

                for model_name in ["PhenoFlex GDH", "Fixed dynamic PhenoFlex"]:
                    residual = comparison["observed_bloom_jday"] - comparison[model_name]
                    axes[1].plot(comparison["Year"], residual, marker="o", label=model_name)
                axes[1].axhline(0, color="black", linewidth=0.8)
                axes[1].set_ylabel("Residual (observed - predicted days)")
                axes[1].set_title("Prediction residuals")
                axes[1].legend(frameon=False)

                fig
                """
            ),
        ],
    )


def main() -> None:
    make_synthetic_data()
    make_notebooks()


if __name__ == "__main__":
    main()
