import numpy as np
from chillPy import identify_common_string, runn_mean, runn_mean_pred, select_by_file_extension, test_if_equal


def test_string_and_file_utilities():
    assert select_by_file_extension(["a.csv", "b.txt"], ".csv") == ["a.csv"]
    assert identify_common_string(["flower_2020", "flower_2021"]) == "flower_202"
    assert test_if_equal([1, 1, 1]) is True


def test_runn_mean_matches_r_edge_window_behavior():
    np.testing.assert_allclose(runn_mean([1, 2, 3, 4, 5], 3), np.array([1.5, 2, 3, 4, 4.5]))
    np.testing.assert_allclose(runn_mean([1, 2, 3, 4, 5, 6], 4), np.array([1.5, 2, 3, 4, 5, 5.5]))
    np.testing.assert_allclose(
        runn_mean([1, 2, 3, 4, 5], 3, exclude_central_value=True),
        np.array([2, 2, 3, 4, 4]),
    )
    np.testing.assert_allclose(
        runn_mean([1, 2, 3], 1, exclude_central_value=True),
        np.array([np.nan, np.nan, np.nan]),
    )


def test_runn_mean_handles_missing_values_like_r_mean():
    result = runn_mean([1, np.nan, 3, 4, 5], 3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    np.testing.assert_allclose(
        runn_mean([1, np.nan, 3, 4, 5], 3, na_rm=True),
        np.array([1, 2, 3.5, 4, 4.5]),
    )


def test_runn_mean_pred_interpolates_running_values():
    result = runn_mean_pred(
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [1, 1.5, 3.2, 5, 6, np.nan],
        runn_mean=3,
    )

    np.testing.assert_allclose(result["x"], np.array([1, 1.5, 3.2, 5, 6, np.nan]))
    np.testing.assert_allclose(result["predicted"], np.array([15, 17.5, 32, 45, np.nan, np.nan]))
