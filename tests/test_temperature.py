import numpy as np
import pandas as pd
import pytest

from chillPy import (
    filter_temperatures,
    interpolate_gaps,
    interpolate_gaps_hourly,
    make_all_day_table,
    make_hourly_temps,
    patch_daily_temperatures,
    patch_daily_temps,
    stack_hourly_temps,
)
from chillPy.temperature import (
    chilling,
    chilling_hourtable,
    daily_chill,
    temp_response,
    temp_response_daily_list,
    temp_response_hourtable,
)
from chillPy.temperature_models import chilling_hours, dynamic_model, gdh


def test_interpolate_gaps_matches_r_endpoint_and_internal_gap_behavior():
    result = interpolate_gaps([np.nan, "2", None, 8, np.nan])

    np.testing.assert_allclose(result["interp"], np.array([2, 2, 5, 8, 8], dtype=float))
    np.testing.assert_array_equal(result["missing"], np.array([True, False, True, False, True]))


def test_interpolate_gaps_warns_when_all_values_are_missing():
    with pytest.warns(RuntimeWarning, match="no data"):
        result = interpolate_gaps(["missing", None])

    np.testing.assert_array_equal(result["missing"], np.array([True, True]))
    assert np.isnan(result["interp"]).all()


def test_make_hourly_temps_matches_linvill_formula_for_simple_daily_record():
    daily = pd.DataFrame(
        {
            "Year": [2020, 2020, 2020],
            "Month": [3, 3, 3],
            "Day": [20, 21, 22],
            "JDay": [80, 81, 82],
            "Tmin": [10.0, 10.0, 10.0],
            "Tmax": [20.0, 20.0, 20.0],
        }
    )

    hourly = make_hourly_temps(0, daily, keep_sunrise_sunset=True)

    assert list(hourly.columns[-24:]) == [f"Hour_{hour}" for hour in range(24)]
    np.testing.assert_allclose(
        hourly[["Hour_0", "Hour_6", "Hour_12", "Hour_18", "Hour_23"]].to_numpy(),
        np.array(
            [
                [10.0, 10.10832443, 19.24912513, 17.10926312, 12.12905315],
                [11.52738378, 10.10832613, 19.24912529, 17.10926372, 12.12905272],
                [11.52738279, 10.10833294, 19.24912594, 17.10926611, 10.0],
            ]
        ),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(hourly["Sunrise"], np.array([5.94444663, 5.94444575, 5.94444224]))
    np.testing.assert_allclose(hourly["Sunset"], np.array([18.05555337, 18.05555425, 18.05555776]))


def test_make_hourly_temps_validates_required_columns():
    with pytest.raises(ValueError, match="Tmax"):
        make_hourly_temps(50, pd.DataFrame({"Year": [2020], "Month": [1], "Day": [1], "Tmin": [0]}))


def test_stack_hourly_temps_generates_long_hourly_table_from_daily_weather():
    daily = pd.DataFrame(
        {
            "Year": [2020, 2020],
            "Month": [3, 3],
            "Day": [20, 21],
            "JDay": [80, 81],
            "Tmin": [10.0, 11.0],
            "Tmax": [20.0, 21.0],
        }
    )

    result = stack_hourly_temps(daily, latitude=0)
    hourtemps = result["hourtemps"]

    assert result["QC"] is np.nan or np.isnan(result["QC"])
    assert hourtemps.shape[0] == 48
    assert list(hourtemps.columns[-2:]) == ["Hour", "Temp"]
    np.testing.assert_array_equal(hourtemps["Hour"].head(24).to_numpy(), np.arange(24, dtype=float))
    assert hourtemps.loc[0, "Year"] == 2020
    assert hourtemps.loc[0, "JDay"] == 80
    assert hourtemps.loc[24, "JDay"] == 81


def _constant_daily_extremes(days: int = 3, *, tmin: float = 5.0, tmax: float = 15.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2021] * days,
            "Month": [1] * days,
            "Day": list(range(1, days + 1)),
            "JDay": list(range(1, days + 1)),
            "Tmin": [tmin] * days,
            "Tmax": [tmax] * days,
        }
    )


def _ideal_hourly_extremes(days: int = 3, *, tmin: float = 5.0, tmax: float = 15.0) -> pd.DataFrame:
    return stack_hourly_temps(_constant_daily_extremes(days, tmin=tmin, tmax=tmax), latitude=0)["hourtemps"]


def test_interpolate_gaps_hourly_preserves_complete_hourly_records():
    hourly = _ideal_hourly_extremes(days=3)

    result = interpolate_gaps_hourly(
        hourly,
        latitude=0,
        minimum_values_for_solving=2,
        runn_mean_test_diff=999,
    )
    weather = result["weather"]

    assert len(weather) == len(hourly)
    assert {"Temp_measured", "Tmin_source", "Tmax_source"}.issubset(weather.columns)
    assert {"Tmin", "Tmax"}.isdisjoint(weather.columns)
    assert weather[["Year", "Month", "Day", "Hour"]].duplicated().sum() == 0
    assert weather[["Year", "Month", "Day", "Hour"]].reset_index(drop=True).equals(
        weather[["Year", "Month", "Day", "Hour"]].sort_values(["Year", "Month", "Day", "Hour"]).reset_index(drop=True)
    )
    np.testing.assert_allclose(weather["Temp"].to_numpy(), hourly["Temp"].to_numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(weather["Temp_measured"].to_numpy(), hourly["Temp"].to_numpy(), rtol=1e-10, atol=1e-10)
    assert weather["Tmin_source"].isna().all()
    assert weather["Tmax_source"].isna().all()


def test_interpolate_gaps_hourly_fills_isolated_and_cross_midnight_hourly_gaps():
    expected = _ideal_hourly_extremes(days=3)
    hourly = expected.copy()
    missing = ((hourly["Day"] == 1) & (hourly["Hour"] == 23)) | (
        (hourly["Day"] == 2) & (hourly["Hour"].isin([0, 1, 2, 12]))
    )
    hourly.loc[missing, "Temp"] = np.nan

    result = interpolate_gaps_hourly(
        hourly,
        latitude=0,
        minimum_values_for_solving=2,
        runn_mean_test_diff=999,
    )
    weather = result["weather"]
    merged = weather.merge(
        expected.rename(columns={"Temp": "expected"}),
        on=["Year", "Month", "Day", "JDay", "Hour"],
        how="left",
    )

    assert merged.loc[missing.to_numpy(), "Temp_measured"].isna().all()
    assert merged.loc[missing.to_numpy(), "Tmin_source"].notna().all()
    assert merged.loc[missing.to_numpy(), "Tmax_source"].notna().all()
    np.testing.assert_allclose(
        merged.loc[missing.to_numpy(), "Temp"].to_numpy(),
        merged.loc[missing.to_numpy(), "expected"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_interpolate_gaps_hourly_fills_leading_and_trailing_gaps():
    expected = _ideal_hourly_extremes(days=3)
    hourly = expected.copy()
    hourly.loc[[0, len(hourly) - 1], "Temp"] = np.nan

    weather = interpolate_gaps_hourly(
        hourly,
        latitude=0,
        minimum_values_for_solving=2,
        runn_mean_test_diff=999,
    )["weather"]

    assert pd.isna(weather.loc[0, "Temp_measured"])
    assert pd.isna(weather.loc[len(weather) - 1, "Temp_measured"])
    np.testing.assert_allclose(
        weather.loc[[0, len(weather) - 1], "Temp"].to_numpy(),
        expected.loc[[0, len(expected) - 1], "Temp"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_interpolate_gaps_hourly_uses_daily_proxy_for_full_missing_days():
    daily = _constant_daily_extremes(days=4)
    expected = _ideal_hourly_extremes(days=4)
    hourly = expected.copy()
    hourly.loc[hourly["Day"] == 2, "Temp"] = np.nan

    result = interpolate_gaps_hourly(
        hourly,
        latitude=0,
        daily_temps={"proxy": daily},
        minimum_values_for_solving=12,
        runn_mean_test_diff=999,
    )
    weather = result["weather"]
    report = result["daily_patch_report"]
    merged = weather.merge(
        expected.rename(columns={"Temp": "expected"}),
        on=["Year", "Month", "Day", "JDay", "Hour"],
        how="left",
    )
    day_2 = merged["Day"] == 2

    assert merged.loc[day_2, "Temp_measured"].isna().all()
    assert not merged.loc[day_2, "Temp"].isna().any()
    np.testing.assert_allclose(
        merged.loc[day_2, "Temp"].to_numpy(),
        merged.loc[day_2, "expected"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    assert "proxy" in set(report["Proxy"])


def test_interpolate_gaps_hourly_sorts_and_averages_duplicate_timestamps():
    hourly = _ideal_hourly_extremes(days=2)
    shuffled_with_duplicates = pd.concat([hourly.iloc[[5]], hourly.iloc[::-1], hourly.iloc[[5]]], ignore_index=True)

    weather = interpolate_gaps_hourly(
        shuffled_with_duplicates,
        latitude=0,
        minimum_values_for_solving=2,
        runn_mean_test_diff=999,
    )["weather"]

    assert len(weather) == 48
    assert weather[["Year", "Month", "Day", "Hour"]].duplicated().sum() == 0
    assert weather[["Year", "Month", "Day", "Hour"]].reset_index(drop=True).equals(
        weather[["Year", "Month", "Day", "Hour"]].sort_values(["Year", "Month", "Day", "Hour"]).reset_index(drop=True)
    )


def test_interpolate_gaps_hourly_validates_required_columns_and_dates():
    with pytest.raises(ValueError, match="Temp"):
        interpolate_gaps_hourly(pd.DataFrame({"Year": [2021], "Month": [1], "Day": [1], "Hour": [0]}))
    with pytest.raises(ValueError, match="Hour values"):
        interpolate_gaps_hourly(
            pd.DataFrame({"Year": [2021], "Month": [1], "Day": [1], "Hour": [24], "Temp": [5.0]})
        )
    with pytest.raises(ValueError, match="invalid Year"):
        interpolate_gaps_hourly(
            pd.DataFrame({"Year": [2021], "Month": [2], "Day": [30], "Hour": [0], "Temp": [5.0]})
        )
    with pytest.raises(ValueError, match="at least 2"):
        interpolate_gaps_hourly(
            pd.DataFrame({"Year": [2021], "Month": [1], "Day": [1], "Hour": [0], "Temp": [5.0]}),
            minimum_values_for_solving=1,
        )


def _two_day_hourly_record(temp: float = 5.0) -> pd.DataFrame:
    rows = []
    for year, month, day, jday in [(2020, 12, 31, 366), (2021, 1, 1, 1)]:
        for hour in range(24):
            rows.append(
                {
                    "Year": year,
                    "Month": month,
                    "Day": day,
                    "JDay": jday,
                    "Hour": hour,
                    "Temp": temp,
                }
            )
    return pd.DataFrame(rows)


def test_temp_response_summarizes_cross_year_season_with_default_models():
    hourly = _two_day_hourly_record(5.0)

    result = temp_response(hourly, start_jday=366, end_jday=1, mean_out=True)

    assert list(result["Season"]) == ["2020/2021"]
    assert result.loc[0, "End_year"] == 2021
    assert result.loc[0, "Season_days"] == 2
    assert result.loc[0, "Data_days"] == 2
    assert result.loc[0, "Perc_complete"] == pytest.approx(100)
    assert result.loc[0, "Input_mean"] == pytest.approx(5)
    assert result.loc[0, "Chilling_Hours"] == pytest.approx(48)
    assert result.loc[0, "Utah_Chill_Units"] == pytest.approx(48)
    assert result.loc[0, "GDH"] == pytest.approx(gdh(np.repeat(5.0, 48))[-1])
    assert result.loc[0, "Chill_Portions"] == pytest.approx(dynamic_model(np.repeat(5.0, 48))[-1])


def test_chilling_uses_legacy_column_names_for_standard_metrics():
    hourly = _two_day_hourly_record(5.0)

    result = chilling({"hourtemps": hourly, "QC": np.nan}, start_jday=366, end_jday=1)

    assert {"Chilling_Hours", "Utah_Model", "Chill_portions", "GDH"}.issubset(result.columns)
    assert result.loc[0, "Chilling_Hours"] == pytest.approx(48)
    assert result.loc[0, "Utah_Model"] == pytest.approx(48)


def test_temp_response_whole_record_returns_last_cumulative_values():
    hourly = _two_day_hourly_record(5.0)

    result = temp_response(hourly, models={"CH": chilling_hours}, whole_record=True)

    assert result["CH"] == pytest.approx(48)


def test_daily_chill_aggregates_hourly_weights_by_day():
    hourly = _two_day_hourly_record(5.0)
    hourly.loc[hourly["JDay"] == 1, "Temp"] = 10.0

    result = daily_chill(hourly, models={"CH": chilling_hours, "GDH": gdh})
    daily = result["daily_chill"]

    assert result["object_type"] == "daily_chill"
    assert list(daily["YYMMDD"]) == [20201231, 20210101]
    np.testing.assert_allclose(daily["CH"], np.array([24, 0], dtype=float))
    np.testing.assert_allclose(daily["Tmean"], np.array([5, 10], dtype=float))
    np.testing.assert_allclose(
        daily["GDH"],
        np.array([gdh(np.repeat(5.0, 24), summ=False).sum(), gdh(np.repeat(10.0, 24), summ=False).sum()]),
    )


def test_chilling_hourtable_resets_all_hours_on_start_jday_like_r():
    hourly = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021, 2021, 2021],
            "JDay": [1, 1, 2, 2, 2, 3],
            "Hour": [0, 1, 0, 1, 2, 0],
            "Temp": [5.0, 10.0, 5.0, 5.0, 20.0, 5.0],
        }
    )

    result = chilling_hourtable(hourly, start_jday=2)

    np.testing.assert_allclose(result["Chilling_Hours"], np.array([0, 0, 0, 0, 0, 1], dtype=float))
    assert {"Chill_Portions", "Chill_Units", "GDH"}.issubset(result.columns)


def test_temp_response_hourtable_offsets_each_season_at_start_jday():
    hourly = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021, 2021, 2021],
            "JDay": [1, 1, 2, 2, 2, 3],
            "Hour": [0, 1, 0, 1, 2, 0],
            "Temp": [5.0, 10.0, 5.0, 5.0, 20.0, 5.0],
        }
    )

    result = temp_response_hourtable(hourly, start_jday=2, models={"CH": chilling_hours})

    np.testing.assert_allclose(result.loc[result["JDay"] >= 2, "CH"], np.array([0, 1, 1, 2], dtype=float))


def test_temp_response_daily_list_uses_idealized_hourly_generation():
    daily = pd.DataFrame(
        {
            "Year": [2020, 2021],
            "Month": [12, 1],
            "Day": [31, 1],
            "JDay": [366, 1],
            "Tmin": [5.0, 5.0],
            "Tmax": [5.0, 5.0],
        }
    )

    result = temp_response_daily_list(daily, latitude=0, start_jday=366, end_jday=1, models={"CH": chilling_hours})

    assert len(result) == 1
    assert result[0].loc[0, "CH"] == pytest.approx(48)


def test_accumulation_functions_validate_inputs_and_partial_branches():
    with pytest.raises(ValueError, match="Temp"):
        temp_response(pd.DataFrame({"Year": [2021], "JDay": [1], "Hour": [0]}))
    with pytest.raises(ValueError, match="missing hourly temperatures"):
        daily_chill(pd.DataFrame({"Year": [2021], "JDay": [1], "Hour": [0], "Temp": [np.nan]}))
    with pytest.raises(NotImplementedError, match="empirical"):
        temp_response_daily_list([pd.DataFrame()], latitude=0, empirical=pd.DataFrame())


def test_make_all_day_table_fills_missing_daily_records_and_averages_duplicates():
    weather = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 1],
            "Day": [1, 1, 3, 4],
            "Tmin": [1.0, 3.0, 7.0, 9.0],
            "Station": ["A", "A", "A", "A"],
        }
    )

    result = make_all_day_table(weather, timestep="day")

    assert list(result["Day"]) == [1, 2, 3, 4]
    np.testing.assert_allclose(result["Tmin"].to_numpy(), np.array([2.0, np.nan, 7.0, 9.0]))
    assert result.loc[0, "Station"] == "A"
    assert pd.isna(result.loc[1, "Station"])
    assert "DATE" in result.columns


def test_make_all_day_table_fills_missing_hourly_records_from_yearmodaho():
    hourly = pd.DataFrame({"YEARMODAHO": [2021010100, 2021010102], "Temp": [1.0, 5.0]})

    result = make_all_day_table(hourly, timestep="hour", input_timestep="hour", add_date=False)

    assert list(result["Hour"]) == [0, 1, 2]
    np.testing.assert_allclose(result["Temp"].to_numpy(), np.array([1.0, np.nan, 5.0]))
    assert list(result["YEARMODAHO"]) == [2021010100, 2021010101, 2021010102]


def test_make_all_day_table_aggregates_hourly_input_to_daily_extremes():
    hourly = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 1],
            "Day": [1, 1, 1, 1],
            "Hour": [0, 1, 2, 3],
            "Temp": [3.0, 1.0, 4.0, 2.0],
        }
    )

    result = make_all_day_table(hourly, timestep="day", input_timestep="hour")

    assert result.loc[0, "Tmin"] == pytest.approx(1.0)
    assert result.loc[0, "Tmean"] == pytest.approx(2.5)
    assert result.loc[0, "Tmax"] == pytest.approx(4.0)


def test_make_all_day_table_aggregation_hours_requires_extreme_windows():
    hourly = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 1, 1, 1],
            "Day": [1, 1, 1, 2, 2, 2],
            "Hour": [0, 1, 2, 0, 1, 2],
            "Temp": [0.0, 10.0, 20.0, 1.0, np.nan, 21.0],
        }
    )

    result = make_all_day_table(hourly, timestep="day", input_timestep="hour", aggregation_hours=[1, 1, 1])

    np.testing.assert_allclose(result["Tmin"].to_numpy(), np.array([0.0, 1.0]))
    np.testing.assert_allclose(result["Tmax"].to_numpy(), np.array([20.0, 21.0]))


def test_filter_temperatures_marks_remove_values_extremes_outliers_and_sparse_windows():
    weather = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 1, 1, 1],
            "Day": [1, 2, 3, 4, 5, 6],
            "Temp": [10.0, 11.0, 50.0, -99.0, -20.0, 12.0],
        }
    )

    result = filter_temperatures(
        weather,
        remove_value=-99,
        running_mean_filter=20,
        running_mean_length=3,
        min_extreme=-10,
        max_extreme=40,
        max_missing_in_window=0.6,
        missing_window_size=3,
    )

    assert "DATE" not in result.columns
    assert result.loc[result["Day"] == 1, "Temp"].iloc[0] == pytest.approx(10.0)
    assert np.isnan(result.loc[result["Day"] == 3, "Temp"].iloc[0])
    assert np.isnan(result.loc[result["Day"] == 4, "Temp"].iloc[0])
    assert np.isnan(result.loc[result["Day"] == 5, "Temp"].iloc[0])


def test_patch_daily_temperatures_fills_gaps_with_bias_corrected_auxiliary_data():
    weather = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 1],
            "Day": [1, 2, 3, 4],
            "Tmin": [10.0, 11.0, np.nan, np.nan],
            "Tmax": [20.0, np.nan, 22.0, np.nan],
        }
    )
    auxiliary = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 1],
            "Day": [1, 2, 3, 4],
            "Tmin": [8.0, 9.0, 10.0, 11.0],
            "Tmax": [18.0, 19.0, 20.0, 21.0],
        }
    )

    result = patch_daily_temperatures(weather, {"station": auxiliary})
    patched = result["weather"]
    stats = result["statistics"]["station"]

    np.testing.assert_allclose(patched["Tmin"].to_numpy(), np.array([10.0, 11.0, 12.0, 13.0]))
    np.testing.assert_allclose(patched["Tmax"].to_numpy(), np.array([20.0, 21.0, 22.0, 23.0]))
    assert patched.loc[2, "Tmin_source"] == "daily_station"
    assert patched.loc[1, "Tmax_source"] == "daily_station"
    assert stats.loc["Tmin", "mean_bias"] == pytest.approx(2.0)
    assert stats.loc["Tmin", "filled"] == pytest.approx(2)
    assert stats.loc["Tmax", "gaps_remain"] == pytest.approx(0)


def test_patch_daily_temperatures_respects_mean_bias_threshold():
    weather = pd.DataFrame(
        {
            "Year": [2021, 2021],
            "Month": [1, 1],
            "Day": [1, 2],
            "Tmin": [10.0, np.nan],
        }
    )
    auxiliary = pd.DataFrame(
        {
            "Year": [2021, 2021],
            "Month": [1, 1],
            "Day": [1, 2],
            "Tmin": [0.0, 1.0],
        }
    )

    result = patch_daily_temperatures(weather, auxiliary, vars=("Tmin",), max_mean_bias=1)

    assert np.isnan(result["weather"].loc[1, "Tmin"])
    assert result["statistics"][0].loc["Tmin", "filled"] == pytest.approx(0)
def test_patch_daily_temps_applies_interval_specific_bias_and_sources():
    weather = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021, 2021, 2021],
            "Month": [2, 1, 1, 2, 1, 2],
            "Day": [3, 1, 3, 1, 2, 2],
            "Tmin": [np.nan, 10.0, np.nan, 20.0, 12.0, 22.0],
        }
    )
    auxiliary = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 2, 2, 2],
            "Day": [1, 2, 3, 1, 2, 3],
            "Tmin": [12.0, 14.0, 16.0, 17.0, 19.0, 21.0],
        }
    )

    result = patch_daily_temps(weather, {"proxy": auxiliary}, vars=("Tmin",), time_interval="month")
    patched = result["weather"]
    stats = result["statistics"]["Tmin"]["proxy"]

    assert patched["YEARMODA"].is_monotonic_increasing
    jan3 = patched.loc[patched["YEARMODA"] == 20210103].iloc[0]
    feb3 = patched.loc[patched["YEARMODA"] == 20210203].iloc[0]
    assert jan3["Tmin"] == pytest.approx(14.0)
    assert feb3["Tmin"] == pytest.approx(24.0)
    assert jan3["Tmin_source"] == "proxy"
    assert feb3["Tmin_source"] == "proxy"
    assert patched.loc[patched["YEARMODA"] == 20210101, "Tmin_source"].iloc[0] == "original"
    assert list(stats["Interval"].head(2)) == [1, 2]
    assert stats.loc[stats["Interval"] == 1, "Mean_bias"].iloc[0] == pytest.approx(2.0)
    assert stats.loc[stats["Interval"] == 2, "Mean_bias"].iloc[0] == pytest.approx(-3.0)
    assert stats.loc[stats["Interval"] == 1, "Filled"].iloc[0] == pytest.approx(1)


def test_patch_daily_temps_respects_interval_bias_thresholds():
    weather = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 2, 2, 2],
            "Day": [1, 2, 3, 1, 2, 3],
            "Tmin": [10.0, 12.0, np.nan, 20.0, 22.0, np.nan],
        }
    )
    auxiliary = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2021, 2021, 2021],
            "Month": [1, 1, 1, 2, 2, 2],
            "Day": [1, 2, 3, 1, 2, 3],
            "Tmin": [20.0, 22.0, 24.0, 21.0, 23.0, 25.0],
        }
    )

    result = patch_daily_temps(
        weather,
        {"proxy": auxiliary},
        vars=("Tmin",),
        max_mean_bias=2,
        time_interval="month",
    )
    patched = result["weather"]
    stats = result["statistics"]["Tmin"]["proxy"]

    assert np.isnan(patched.loc[patched["YEARMODA"] == 20210103, "Tmin"].iloc[0])
    assert patched.loc[patched["YEARMODA"] == 20210203, "Tmin"].iloc[0] == pytest.approx(24.0)
    assert stats.loc[stats["Interval"] == 1, "Filled"].iloc[0] == pytest.approx(0)
    assert stats.loc[stats["Interval"] == 2, "Filled"].iloc[0] == pytest.approx(1)


def test_make_all_day_table_validates_unsupported_conversion():
    with pytest.raises(ValueError, match="daily to hourly"):
        make_all_day_table(
            pd.DataFrame({"Year": [2021], "Month": [1], "Day": [1]}),
            timestep="hour",
            input_timestep="day",
        )
