import inspect

import chillPy


def test_public_pythonic_names_are_importable():
    for name in [
        "chilling_hours",
        "dynamic_model",
        "daily_chill",
        "temperature_generation",
        "phenology_fit",
        "weather_to_chillr",
    ]:
        assert callable(getattr(chillPy, name))


def test_r_style_aliases_are_importable_when_valid_identifiers():
    for name in ["Chilling_Hours", "Dynamic_Model", "Date2YEARMODA", "tempResponse", "weather2chillR"]:
        assert callable(getattr(chillPy, name))


def test_stub_functions_have_docstrings():
    assert inspect.getdoc(chillPy.chilling_hours)
    assert inspect.getdoc(chillPy.daily_chill)


def test_dot_export_is_traceable_with_getattr():
    assert getattr(chillPy, "bootstrap.phenologyFit") is chillPy.bootstrap_phenology_fit

