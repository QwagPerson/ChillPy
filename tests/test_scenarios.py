import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from chillPy import (
    download_cmip6_ecmwfr,
    load_temperature_scenarios,
    make_climate_scenario,
    ordered_climate_list,
    save_temperature_scenarios,
    temperature_generation,
    temperature_scenario_from_records,
    temperature_scenario_baseline_adjustment,
    gen_rel_change_scenario,
    convert_scen_information,
    make_climate_scenario_from_files,
)
from chillPy.weather import check_temperature_scenario


def _complete_daily_weather(start_year=2000, end_year=2001):
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    day_angle = 2 * np.pi * (dates.dayofyear.to_numpy() - 1) / 365.25
    tmin = 4 + 8 * np.sin(day_angle) + (dates.day.to_numpy() % 5) * 0.2
    tmax = tmin + 9 + (dates.dayofyear.to_numpy() % 7) * 0.3
    return pd.DataFrame(
        {
            "Year": dates.year,
            "Month": dates.month,
            "Day": dates.day,
            "Tmin": tmin,
            "Tmax": tmax,
        }
    )


def test_ordered_climate_list():
    # Standard alphabetical sorting would put 10 before 2
    files = ["T1.csv", "T10.csv", "T2.csv"]
    assert ordered_climate_list(files, ".csv") == ["T1.csv", "T2.csv", "T10.csv"]

    # Without extension
    assert ordered_climate_list(["10", "1", "2"]) == ["1", "2", "10"]

    # Complex case with shared leading/trailing
    complex_files = ["SiteA_2020_v1.txt", "SiteA_2010_v1.txt", "SiteA_2015_v1.txt"]
    assert ordered_climate_list(complex_files) == [
        "SiteA_2010_v1.txt",
        "SiteA_2015_v1.txt",
        "SiteA_2020_v1.txt"
    ]


def test_download_placeholder_raises():
    # download_cmip6_ecmwfr now has arguments
    with pytest.raises(TypeError):
        download_cmip6_ecmwfr()


def test_make_climate_scenario():
    metrics = {"2000": pd.DataFrame({"Chill": [10, 20]}), "2010": pd.DataFrame({"Chill": [15, 25]})}
    scens = make_climate_scenario(metrics, caption=["Test", "Caption"], time_series=True)

    assert len(scens) == 1
    assert scens[0]["data"] == metrics
    assert scens[0]["caption"] == ["Test", "Caption"]
    assert scens[0]["time_series"] is True
    assert scens[0]["labels"] == [2000.0, 2010.0]

    # Add to
    scens2 = make_climate_scenario({"2020": pd.DataFrame()}, add_to=scens)
    assert len(scens2) == 2
    assert scens2 is scens


def test_temperature_scenario_from_records():
    # Create synthetic weather: Tmin and Tmax increasing over years
    data = []
    for year in range(2000, 2011):
        for month in range(1, 13):
            # Tmin = 0 + (year-2000) * 0.1, Tmax = 10 + (year-2000) * 0.1
            data.append({
                "Year": year,
                "Month": month,
                "Day": 1,
                "Tmin": (year - 2000) * 0.1 + month * 0.01,
                "Tmax": 10 + (year - 2000) * 0.1 + month * 0.01
            })
    weather = pd.DataFrame(data)

    # Test regression
    scen = temperature_scenario_from_records(weather, year=2020, scen_type="regression")
    assert "2020" in scen
    # 2020 should have Tmin around (2020-2000)*0.1 = 2.0
    # For Month 1: 2.0 + 0.01 = 2.01
    assert scen["2020"]["data"].loc[1, "Tmin"] == pytest.approx(2.01)
    assert scen["2020"]["data"].loc[1, "Tmax"] == pytest.approx(12.01)

    # Test running_mean
    scen_rm = temperature_scenario_from_records(weather, year=2005, scen_type="running_mean", runn_mean=5)
    assert "2005" in scen_rm
    assert scen_rm["2005"]["data"].loc[1, "Tmin"] == pytest.approx(0.51)


def test_temperature_scenario_baseline_adjustment():
    base = {
        "data": pd.DataFrame({"Tmin": [10.0] * 12, "Tmax": [20.0] * 12}),
        "scenario_year": 2000,
        "scenario_type": "absolute"
    }
    future = {
        "data": pd.DataFrame({"Tmin": [12.0] * 12, "Tmax": [22.0] * 12}),
        "scenario_year": 2050,
        "scenario_type": "absolute"
    }

    rel = temperature_scenario_baseline_adjustment(base, future)
    assert rel["scenario_type"] == "relative"
    assert (rel["data"]["Tmin"] == 2.0).all()
    assert rel["reference_year"] == 2000

    # Test relative + relative
    future_rel = {
        "data": pd.DataFrame({"Tmin": [1.0] * 12, "Tmax": [1.0] * 12}),
        "scenario_year": 2080,
        "reference_year": 2050,
        "scenario_type": "relative"
    }
    # We need to make sure future (rel to 2000) is used as baseline for future_rel
    combined = temperature_scenario_baseline_adjustment(rel, future_rel)
    assert combined["scenario_type"] == "relative"
    assert (combined["data"]["Tmin"] == 3.0).all()  # 2.0 + 1.0
    assert combined["reference_year"] == 2000


def test_temperature_generation_relative_scenario_is_seeded_and_named():
    weather = _complete_daily_weather()
    scenario = {
        "warm": {
            "data": pd.DataFrame({"Tmin": [1.0] * 12, "Tmax": [2.0] * 12}),
            "reference_year": 2000.5,
            "scenario_type": "relative",
        }
    }

    generated = temperature_generation(
        weather,
        years=[2000, 2001],
        sim_years=[2002, 2002],
        temperature_scenario=scenario,
        seed=123,
        warn_me=False,
    )
    repeated = temperature_generation(
        weather,
        years=[2000, 2001],
        sim_years=[2002, 2002],
        temperature_scenario=scenario,
        seed=123,
        warn_me=False,
    )

    assert list(generated) == ["warm"]
    frame = generated["warm"]
    assert list(frame.columns) == ["YEARMODA", "DATE", "Year", "Month", "Day", "Tmin", "Tmax"]
    assert len(frame) == 365
    assert frame["YEARMODA"].iloc[0] == 20020101
    assert frame["YEARMODA"].iloc[-1] == 20021231
    assert (frame["Tmin"] <= frame["Tmax"]).all()
    pd.testing.assert_frame_equal(frame, repeated["warm"])


def test_temperature_generation_absolute_scenario_sets_monthly_climate():
    weather = _complete_daily_weather()
    scenario = {
        "data": pd.DataFrame({"Tmin": [5.0] * 12, "Tmax": [15.0] * 12}),
        "scenario_type": "absolute",
        "reference_year": 2000.5,
    }

    with pytest.warns(UserWarning, match="Absolute temperature scenario"):
        generated = temperature_generation(
            weather,
            years=[2000, 2001],
            sim_years=[2002, 2002],
            temperature_scenario=scenario,
            seed=7,
            warn_me=True,
            temperature_check_args={"warn_me": False},
        )["0"]

    assert generated["Tmin"].mean() == pytest.approx(5.0, abs=0.5)
    assert generated["Tmax"].mean() == pytest.approx(15.0, abs=0.5)


def test_temperature_generation_rejects_incomplete_calibration_weather():
    weather = _complete_daily_weather()
    weather = weather.loc[
        ~(
            (weather["Year"] == 2000)
            & (weather["Month"] == 2)
            & (weather["Day"] == 29)
        )
    ]

    with pytest.raises(ValueError, match="Weather record not complete"):
        temperature_generation(
            weather,
            years=[2000, 2001],
            sim_years=[2002, 2002],
            warn_me=False,
        )


def test_temperature_generation_reference_year_difference_limit():
    weather = _complete_daily_weather()
    scenario = {
        "data": pd.DataFrame({"Tmin": [1.0] * 12, "Tmax": [2.0] * 12}),
        "scenario_type": "relative",
        "reference_year": 1990,
    }

    with pytest.raises(ValueError, match="greater than 5 years"):
        temperature_generation(
            weather,
            years=[2000, 2001],
            sim_years=[2002, 2002],
            temperature_scenario=scenario,
            warn_me=False,
        )


def test_gen_rel_change_scenario():
    hist = pd.DataFrame({
        "location": ["Loc1"] * 12,
        "Month": list(range(1, 13)),
        "Tmin": [0.0] * 12,
        "Tmax": [10.0] * 12
    })
    fut = pd.DataFrame({
        "location": ["Loc1"] * 12,
        "Month": list(range(1, 13)),
        "Year": [2050] * 12,
        "Tmin": [2.0] * 12,
        "Tmax": [12.0] * 12,
        "ssp": ["ssp126"] * 12
    })
    downloaded = {
        "historical_ModelA": hist,
        "ssp126_ModelA": fut
    }

    res = gen_rel_change_scenario(downloaded, scenarios=[2050], reference_period=[2000])
    assert not res.empty
    assert (res["Tmin"] == 2.0).all()
    assert res["scenario"].iloc[0] == "ssp126"


def test_convert_scen_information():
    df = pd.DataFrame({
        "location": ["Loc1"] * 12,
        "Month": list(range(1, 13)),
        "Tmin": [1.0] * 12,
        "Tmax": [1.0] * 12,
        "scenario": ["ssp126"] * 12,
        "labels": ["ModelA"] * 12,
        "scenario_year": [2050] * 12
    })

    # df to list
    lst = convert_scen_information(df, give_structure=False)
    assert "Loc1.ssp126.ModelA.2050" in lst
    assert lst["Loc1.ssp126.ModelA.2050"]["scenario"] == "ssp126"

    # list to df
    df2 = convert_scen_information(lst)
    assert isinstance(df2, pd.DataFrame)
    assert len(df2) == 12
    assert "location" in df2.columns


def test_make_climate_scenario_from_files(tmp_path):
    d = tmp_path / "metrics"
    d.mkdir()
    f1 = d / "ModelA_2050.csv"
    f2 = d / "ModelB_2050.csv"
    pd.DataFrame({"Tmin": [1.0] * 12}).to_csv(f1, index=False)
    pd.DataFrame({"Tmin": [2.0] * 12}).to_csv(f2, index=False)

    scen = make_climate_scenario_from_files(d, criteria_list=[["ModelA", "ModelB"], "2050"])
    assert len(scen["data"]) == 2
    assert scen["data"][0]["Tmin"].iloc[0] == 1.0  # ModelA (alphabetical)
    assert scen["data"][1]["Tmin"].iloc[0] == 2.0  # ModelB
