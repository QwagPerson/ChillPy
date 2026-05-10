from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from chillPy import (
    Date2YEARMODA,
    JDay_count,
    JDay_earlier,
    JDay_later,
    YEARMODA2Date,
    add_date,
    daylength,
    get_last_date,
    leap_year,
    make_JDay,
)


def test_date_aliases_round_trip():
    assert YEARMODA2Date(20260131) == date(2026, 1, 31)
    assert Date2YEARMODA(date(2026, 1, 31)) == 20260131
    assert Date2YEARMODA(datetime(2026, 1, 31, 14), hours=True) == 2026013114
    np.testing.assert_array_equal(
        Date2YEARMODA([date(2024, 2, 29), "2024-12-31"]),
        np.array([20240229, 20241231]),
    )
    np.testing.assert_array_equal(
        YEARMODA2Date([20240229, 20241231]),
        np.array([date(2024, 2, 29), date(2024, 12, 31)], dtype=object),
    )


def test_make_jday_adds_day_of_year_and_preserves_order_and_duplicates():
    dates = pd.DataFrame(
        {
            "Year": [1977, 1980, 2004, 2011, 2016, 2016],
            "Month": [11, 8, 3, 12, 8, 8],
            "Day": [1, 21, 2, 24, 2, 2],
            "Value": [5, 4, 3, 2, 1, 1],
        }
    )

    result = make_JDay(dates)

    assert list(result["Value"]) == [5, 4, 3, 2, 1, 1]
    np.testing.assert_array_equal(result["JDay"].to_numpy(dtype=int), np.array([305, 234, 62, 358, 215, 215]))
    assert "JDay" not in dates.columns


def test_make_jday_accepts_python_date_column_as_extension():
    result = make_JDay(pd.DataFrame({"Date": ["2024-02-29", pd.Timestamp("2023-12-31"), date(2023, 1, 1)]}))

    np.testing.assert_array_equal(result["JDay"].to_numpy(dtype=int), np.array([60, 365, 1]))
    np.testing.assert_array_equal(result["Year"].to_numpy(dtype=int), np.array([2024, 2023, 2023]))


def test_make_jday_rejects_missing_columns_and_invalid_dates():
    with pytest.raises(ValueError, match="required column"):
        make_JDay(pd.DataFrame({"Year": [2024], "Month": [1]}))
    with pytest.raises(ValueError, match="invalid dates"):
        make_JDay(pd.DataFrame({"Year": [2023], "Month": [2], "Day": [29]}))
    with pytest.raises(ValueError, match="invalid date"):
        make_JDay(pd.DataFrame({"Date": ["not-a-date"]}))


def test_add_date_uses_optional_time_columns_and_preserves_rows():
    weather = pd.DataFrame(
        {
            "Year": [2024, 2024, 2024],
            "Month": [2, 2, 3],
            "Day": [28, 29, 1],
            "Hour": [23, 0, 1],
            "Minute": [30, 0, 15],
            "Temp": [5.0, 6.0, 7.0],
        }
    )

    result = add_date(weather)

    assert list(result["Temp"]) == [5.0, 6.0, 7.0]
    assert result.loc[0, "Date"] == pd.Timestamp("2024-02-28 23:30")
    assert result.loc[1, "Date"] == pd.Timestamp("2024-02-29 00:00")
    assert "Date" not in weather.columns


def test_add_date_validates_required_columns_and_invalid_dates():
    with pytest.raises(ValueError, match="Required input column"):
        add_date(pd.DataFrame({"Year": [2024], "Month": [1]}))
    with pytest.raises(ValueError, match="invalid values"):
        add_date(pd.DataFrame({"Year": [2023], "Month": [2], "Day": [29]}))


def test_jday_count_and_daylength_values():
    assert leap_year(2024) is True
    assert leap_year(1900) is False
    assert leap_year(2000) is True
    np.testing.assert_array_equal(leap_year([2023, 2024, 2100]), np.array([False, True, False]))
    assert JDay_count(360, 5) == 10
    assert JDay_count(360, 5, leap_year=True) == 11
    assert JDay_count(320, 20, season=(305, 59), leap_year=2004) == 66
    result = daylength(50.0, 120)
    assert set(result) == {"Sunrise", "Sunset", "Daylength"}
    np.testing.assert_allclose(result["Sunrise"], np.array([4.70708068]))
    np.testing.assert_allclose(result["Sunset"], np.array([19.29291932]))
    np.testing.assert_allclose(result["Daylength"], np.array([14.58583865]))


def test_jday_earlier_later_handle_cross_year_seasons_and_vectors():
    assert JDay_earlier(10, 365, season=(305, 59)) is False
    assert JDay_later(10, 365, season=(305, 59)) is True
    vector_result = JDay_earlier([320, 365, 10, 100], 365, season=(305, 59))
    assert list(vector_result[:3]) == [True, False, False]
    assert np.isnan(vector_result[3])
    assert np.isnan(JDay_later(365, 100, season=(305, 59)))
    with pytest.raises(ValueError, match="Julian date"):
        JDay_earlier(400, 365)


def test_get_last_date_uses_largest_annual_gap():
    assert get_last_date([1, 3, 6, 8, 10, 25]) == 25
    assert get_last_date([345, 356, 360, 365, 2, 5, 7, 10]) == 10
    assert get_last_date([345, 356, 360, 365, 2, 5, 7, 10], first=True) == 345
    assert get_last_date([]) is None
    with pytest.raises(ValueError, match="Julian dates"):
        get_last_date([10, 400])
