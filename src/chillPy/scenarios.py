"""Climate and temperature scenario placeholders mapped from chillR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._base import as_list, not_implemented, placeholder_record


def make_climate_scenario(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``make_climate_scenario``."""
    return placeholder_record("make_climate_scenario", args=args, kwargs=kwargs, scenarios=[])


def make_climate_scenario_from_files(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``make_climate_scenario_from_files``."""
    return placeholder_record("make_climate_scenario_from_files", args=args, kwargs=kwargs, scenarios=[])


def temperature_scenario_from_records(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``temperature_scenario_from_records``."""
    return placeholder_record("temperature_scenario_from_records", args=args, kwargs=kwargs, scenario={})


def temperature_scenario_baseline_adjustment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``temperature_scenario_baseline_adjustment``."""
    return placeholder_record("temperature_scenario_baseline_adjustment", args=args, kwargs=kwargs, scenario={})


def temperature_generation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``temperature_generation``."""
    return placeholder_record("temperature_generation", args=args, kwargs=kwargs, generated=[])


def gen_rel_change_scenario(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``gen_rel_change_scenario``."""
    return placeholder_record("gen_rel_change_scenario", args=args, kwargs=kwargs, scenario={})


def convert_scen_information(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``convert_scen_information``."""
    return placeholder_record("convert_scen_information", args=args, kwargs=kwargs, converted={})


def ordered_climate_list(strings: Any, file_extension: str | None = None) -> list[Any]:
    """Partial placeholder for R ``ordered_climate_list`` returning sorted values."""
    values = as_list(strings)
    if file_extension:
        values = [value for value in values if str(value).endswith(file_extension)]
    return sorted(values, key=str)


def save_temperature_scenarios(generated_temperatures: Any, path: str | Path, prefix: str) -> dict[str, Any]:
    """Placeholder for R ``save_temperature_scenarios``.

    This does not write files yet; it returns the intended save plan.
    """
    return placeholder_record(
        "save_temperature_scenarios",
        generated_temperatures=generated_temperatures,
        path=Path(path),
        prefix=prefix,
    )


def load_temperature_scenarios(path: str | Path, prefix: str) -> dict[str, Any]:
    """Placeholder for R ``load_temperature_scenarios`` returning an empty set."""
    return placeholder_record("load_temperature_scenarios", path=Path(path), prefix=prefix, scenarios=[])


def load_climate_wizard_scenarios(path: str | Path, prefix: str) -> dict[str, Any]:
    """Placeholder for R ``load_ClimateWizard_scenarios`` returning an empty set."""
    return placeholder_record("load_ClimateWizard_scenarios", path=Path(path), prefix=prefix, scenarios=[])


def get_climate_wizard_data(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``getClimateWizardData``; external access is not implemented."""
    not_implemented("get_climate_wizard_data")


def get_climate_wizard_scenarios(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``getClimateWizard_scenarios``; external access is not implemented."""
    not_implemented("get_climate_wizard_scenarios")


def extract_temperatures_from_grids(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``extract_temperatures_from_grids``."""
    return placeholder_record("extract_temperatures_from_grids", args=args, kwargs=kwargs, temperatures=[])


def extract_cmip6_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for R ``extract_cmip6_data``."""
    return placeholder_record("extract_cmip6_data", args=args, kwargs=kwargs, data=[])


def download_cmip6_ecmwfr(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``download_cmip6_ecmwfr``; download is not implemented."""
    not_implemented("download_cmip6_ecmwfr")


def download_baseline_cmip6_ecmwfr(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``download_baseline_cmip6_ecmwfr``; download is not implemented."""
    not_implemented("download_baseline_cmip6_ecmwfr")


load_ClimateWizard_scenarios = load_climate_wizard_scenarios
getClimateWizardData = get_climate_wizard_data
getClimateWizard_scenarios = get_climate_wizard_scenarios

