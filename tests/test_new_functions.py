import pytest
import pandas as pd
import numpy as np
import warnings
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from chillPy.scenarios import extract_cmip6_data, download_cmip6_ecmwfr, extract_temperatures_from_grids
from chillPy.plotting import make_daily_chill_figures, plot_pls, plot_phenology_trends

def test_extract_cmip6_data_missing_deps():
    stations = pd.DataFrame({
        "station_name": ["Test"],
        "longitude": [0],
        "latitude": [0]
    })
    with pytest.warns(UserWarning, match="xarray.*required"):
        res = extract_cmip6_data(stations)
        assert res["object_type"] == "extract_cmip6_data"

def test_download_cmip6_ecmwfr_missing_deps():
    with pytest.warns(UserWarning, match="cdsapi.*required"):
        with pytest.raises(NotImplementedError):
            download_cmip6_ecmwfr("ssp126", [0, 0, 0, 0])

def test_extract_temperatures_from_grids_missing_deps():
    coords = [0, 0]
    specs = {"base_folder": ".", "minfile": "min.zip", "maxfile": "max.zip"}
    # Create dummy files to avoid FileNotFoundError before dependency check
    with open("min.zip", "w") as f: f.write("")
    with open("max.zip", "w") as f: f.write("")
    
    with pytest.warns(UserWarning, match="rasterio.*required"):
        res = extract_temperatures_from_grids(coords, "WorldClim", specs)
        assert res["object_type"] == "extract_temperatures_from_grids"
    
    import os
    os.remove("min.zip")
    os.remove("max.zip")

def test_make_daily_chill_figures():
    dc = {
        "daily_chill": pd.DataFrame({
            "Year": [2000, 2000],
            "Month": [1, 1],
            "Day": [1, 2],
            "Chilling_Hours": [1, 2],
            "Utah_Model": [1, 2],
            "Chill_Portions": [1, 2],
            "GDH": [1, 2]
        })
    }
    res = make_daily_chill_figures(dc, "output/")
    assert "daily_chill_figure_summary" in res
    assert isinstance(res["daily_chill_figure_summary"], pd.DataFrame)
    assert isinstance(res["figure"], Figure)
    assert len(res["axes"]) == 4

def test_plot_pls():
    pls_res = {
        "PLS_summary": pd.DataFrame({"Date": [101], "VIP": [1.0], "Coef": [0.5]})
    }
    res = plot_pls(pls_res)
    assert isinstance(res["figure"], Figure)
    assert "VIP" in res["data"].columns

def test_plot_phenology_trends():
    pheno_data = pd.DataFrame({
        "Year": [2000, 2001, 2002],
        "pheno": [100, 102, 104]
    })
    res = plot_phenology_trends(pheno_data)
    assert isinstance(res["figure"], Figure)
    assert "trend" in res["data"].columns
    assert not res["data"]["trend"].isna().all()
