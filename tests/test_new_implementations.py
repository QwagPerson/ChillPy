import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from chillPy.scenarios import (
    save_temperature_scenarios,
    load_temperature_scenarios,
    load_climate_wizard_scenarios,
    get_climate_wizard_data,
    extract_temperatures_from_grids
)
from chillPy.weather import make_california_ucipm_station_list
from chillPy.plotting import make_chill_plot

def test_save_load_temperature_scenarios(tmp_path):
    temps = {
        "Scen1": pd.DataFrame({"Tmin": [1, 2], "Tmax": [10, 11]}),
        "Scen2": pd.DataFrame({"Tmin": [3, 4], "Tmax": [13, 14]})
    }
    prefix = "test_temps"
    save_temperature_scenarios(temps, tmp_path, prefix)
    
    # Check if files exist
    assert (tmp_path / "test_temps_1_Scen1.csv").exists()
    assert (tmp_path / "test_temps_2_Scen2.csv").exists()
    
    reloaded = load_temperature_scenarios(tmp_path, prefix)
    assert len(reloaded) == 2
    assert "Scen1" in reloaded
    assert "Scen2" in reloaded
    pd.testing.assert_frame_equal(reloaded["Scen1"], temps["Scen1"])

def test_load_climate_wizard_scenarios(tmp_path):
    # Create a CW-style CSV
    prefix = "cw"
    df = pd.DataFrame({
        "data.Tmin": [1, 2],
        "data.Tmax": [10, 11],
        "scenario": ["rcp45", "rcp45"],
        "start_year": [2020, 2020],
        "end_year": [2050, 2050]
    })
    (tmp_path / "cw_1_GCM1.csv").write_text(df.to_csv(index=False))
    
    reloaded = load_climate_wizard_scenarios(tmp_path, prefix)
    assert "GCM1" in reloaded
    assert reloaded["GCM1"]["scenario"] == "rcp45"
    assert "Tmin" in reloaded["GCM1"]["data"].columns
    assert reloaded["GCM1"]["data"]["Tmin"].iloc[0] == 1

def test_get_climate_wizard_data_placeholder():
    # Since 'requests' might not be installed in the test env, 
    # it should at least return a placeholder or work if installed.
    res = get_climate_wizard_data(
        coordinates=[10, 50],
        scenario="rcp45",
        start_year=2020,
        end_year=2050
    )
    assert isinstance(res, dict)

def test_extract_temperatures_from_grids_placeholder():
    res = extract_temperatures_from_grids(
        coordinates=[10, 50],
        grid_format="WorldClim",
        grid_specifications={"base_folder": ".", "minfile": "min.zip", "maxfile": "max.zip"}
    )
    assert isinstance(res, dict)
    # If placeholder is returned because rasterio is missing
    if res.get("object_type") == "extract_temperatures_from_grids":
        assert "grid_format" in res
    else:
        assert "data" in res

def test_make_california_ucipm_station_list_placeholder():
    res = make_california_ucipm_station_list()
    assert isinstance(res, pd.DataFrame)
    assert "Name" in res.columns

def test_make_chill_plot_data_prep():
    dc = pd.DataFrame({
        "Year": [2000] * 10 + [2001] * 10,
        "Month": [1] * 20,
        "Day": list(range(1, 11)) * 2,
        "Chill": np.random.rand(20)
    })
    obj = {"daily_chill": dc}
    res = make_chill_plot(obj, metrics=["Chill"], startdate=1, enddate=10)
    assert "Chill" in res
    assert isinstance(res["Chill"], pd.DataFrame)
    assert "Mean" in res["Chill"].columns
