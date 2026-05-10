import numpy as np
import pytest

from chillPy import RPD, RPIQ, RMSEP


def test_validation_metrics_match_r_example_values():
    predicted = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    observed = np.array([1.5, 1.8, 3.3, 3.9, 4.4, 6, 7.5, 9, 11, 10], dtype=float)

    assert RMSEP(predicted, observed) == pytest.approx(0.7745967, rel=1e-7)
    assert RPD(predicted, observed) == pytest.approx(4.391574, rel=1e-7)
    assert RPIQ(predicted, observed) == pytest.approx(6.680896, rel=1e-7)


def test_metrics_raise_on_missing_values_unless_na_rm_is_true():
    predicted = np.array([1.0, np.nan, 3.0])
    observed = np.array([1.0, 2.0, 5.0])

    with pytest.raises(ValueError):
        RMSEP(predicted, observed)

    assert RMSEP(predicted, observed, na_rm=True) == pytest.approx(np.sqrt(2))


def test_metrics_validate_shapes():
    with pytest.raises(ValueError):
        RMSEP([1, 2], [1])
