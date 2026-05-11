import numpy as np
import pytest

from chillPy import (
    Chilling_Hours,
    DynModel_driver,
    Dynamic_Model,
    GDD,
    GDH,
    GDH_model,
    PhenoFlex,
    Utah_Model,
    step_model,
)


def test_step_model_and_utah_boundaries_match_r_intervals():
    temps = np.array([-1, 0, 1.4, 2.4, 5, 9.1, 12.4, 15.9, 18, 20], dtype=float)
    expected = np.array([0, 0, 0, 0.5, 1, 1, 0.5, 0, -0.5, -1], dtype=float)

    np.testing.assert_allclose(step_model(temps, summ=False), expected)
    np.testing.assert_allclose(Utah_Model(temps, summ=False), expected)
    np.testing.assert_allclose(Utah_Model(temps), np.cumsum(expected))


def test_chilling_hours_match_r_thresholds():
    temps = np.array([-1, 0, 1.4, 7.2, 7.3, np.nan])
    expected = np.array([0, 1, 1, 1, 0, 0], dtype=float)

    np.testing.assert_allclose(Chilling_Hours(temps, summ=False), expected)
    np.testing.assert_allclose(Chilling_Hours(temps), np.cumsum(expected))


def test_gdh_and_gdd_match_hand_checked_r_values():
    temps = np.array([-1, 0, 5, 25, 30, 36, 40], dtype=float)

    np.testing.assert_allclose(
        GDH(temps, summ=False),
        np.array([0, 0, 0.1172763246, 21, 7.2479245871, 0, 0]),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(GDH_model(temps, summ=False), GDH(temps, summ=False))
    np.testing.assert_allclose(
        GDD(temps, summ=False),
        np.array([0.2083333333, 0.2083333333, 0.2083333333, 0.8333333333, 1.0416666667, 1.0416666667, 1.0416666667]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_dynamic_model_matches_r_hourly_driver_for_constant_temperatures():
    temps = np.repeat(5.0, 40)
    expected_delta = np.zeros(40)
    expected_delta[29] = 0.8413139305

    np.testing.assert_allclose(Dynamic_Model(temps, summ=False), expected_delta, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(Dynamic_Model(temps), np.cumsum(expected_delta), rtol=1e-10, atol=1e-10)

    driver = DynModel_driver(temps)
    assert set(driver) == {"x", "y", "delta", "xs"}
    np.testing.assert_allclose(driver["delta"], expected_delta, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(driver["y"], np.cumsum(expected_delta), rtol=1e-10, atol=1e-10)


def test_temperature_models_validate_inputs():
    with pytest.raises(ValueError):
        step_model([2001])
    with pytest.raises(ValueError):
        Dynamic_Model([1.0, np.nan])
    with pytest.raises(ValueError):
        DynModel_driver([1.0, 2.0], times=[1.0])


def test_phenoflex_placeholder_shape():
    result = PhenoFlex([1.0, 2.0], [0, 1])
    # PhenoFlex is implemented, it returns a dict with several keys
    assert "bloomindex" in result
    assert result["values"].shape == (2,)
