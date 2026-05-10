from pathlib import Path

import pytest

from chillPy import (
    download_cmip6_ecmwfr,
    load_temperature_scenarios,
    make_climate_scenario,
    ordered_climate_list,
    save_temperature_scenarios,
)


def test_scenario_stubs_return_structures(tmp_path: Path):
    assert make_climate_scenario()["object_type"] == "make_climate_scenario"
    assert ordered_climate_list(["b.csv", "a.txt", "a.csv"], ".csv") == ["a.csv", "b.csv"]
    plan = save_temperature_scenarios([], tmp_path, "demo")
    assert plan["prefix"] == "demo"
    assert load_temperature_scenarios(tmp_path, "demo")["scenarios"] == []


def test_download_placeholder_raises():
    with pytest.raises(NotImplementedError):
        download_cmip6_ecmwfr()

