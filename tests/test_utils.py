import numpy as np
import pandas as pd
import pytest
from chillPy import (
    identify_common_string,
    runn_mean,
    runn_mean_pred,
    select_by_file_extension,
    test_if_equal,
    extract_differences_between_characters,
    read_tab
)


def test_string_and_file_utilities():
    assert select_by_file_extension(["a.csv", "b.txt"], ".csv") == ["a.csv"]
    assert np.isnan(select_by_file_extension(["a.csv", "b.txt"], ".pdf"))
    assert identify_common_string(["flower_2020", "flower_2021"]) == "flower_202"
    # 'flower_2020' and 'flower_2021' have no common trailing string
    assert pd.isna(identify_common_string(["flower_2020", "flower_2021"], leading=False))
    assert identify_common_string(["flower_2020_A", "flower_2021_A"], leading=False) == "_A"
    assert pd.isna(identify_common_string(["abc", "def"]))
    assert test_if_equal([1, 1, 1]) is True
    assert test_if_equal([1, 2, 1]) is False

    diffs = extract_differences_between_characters(["Temp_01_Tmin", "Temp_02_Tmin", "Temp_03_Tmin"])
    # R removes all common characters, so "Temp_0" is removed
    np.testing.assert_array_equal(diffs, np.array(["1", "2", "3"], dtype=object))

    diffs_none = extract_differences_between_characters(["abc", "abc"])
    assert pd.isna(diffs_none)


def test_read_tab(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    
    csv_file = d / "test.csv"
    csv_file.write_text("Var1,Var2,Var3,Var4\n1.2,3.4,5.6,7.8\n9.0,1.1,2.2,3.3")
    df1 = read_tab(str(csv_file))
    assert df1.iloc[0, 0] == 1.2
    assert df1.shape == (2, 4)
    
    semi_file = d / "test_semi.csv"
    # 5 semicolons, 4 commas -> should detect semicolon
    semi_file.write_text("Var1;Var2;Var3;Var4\n1,2;3,4;5,6;7,8\n9,0;1,1;2,2;3,3")
    df2 = read_tab(str(semi_file))
    assert df2.iloc[0, 0] == 1.2
    assert df2.shape == (2, 4)


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
