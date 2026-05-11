import pytest
import numpy as np
import pandas as pd

from chillPy import (
    check_temperature_record,
    chile_agromet2chillR,
    fix_weather,
    get_weather,
    handle_gsod,
    make_california_UCIPM_station_list,
    weather2chillR,
)


def test_weather2chillr_converts_gsod_raw_weather_and_completes_full_year():
    raw = pd.DataFrame(
        {
            "DATE": ["2020-03-01", "2020-02-28"],
            "MIN": [9999.9, 32.0],
            "MAX": [68.0, 50.0],
            "TEMP": [59.0, 41.0],
            "PRCP": [99.99, 0.10],
        }
    )

    converted = weather2chillR(raw, database="GSOD")

    assert len(converted) == 366
    assert converted["Date"].is_monotonic_increasing
    feb28 = converted.loc[(converted["Month"] == 2) & (converted["Day"] == 28)].iloc[0]
    mar1 = converted.loc[(converted["Month"] == 3) & (converted["Day"] == 1)].iloc[0]
    assert feb28["Tmin"] == pytest.approx(0.0)
    assert feb28["Tmax"] == pytest.approx(10.0)
    assert feb28["Tmean"] == pytest.approx(5.0)
    assert feb28["Prec"] == pytest.approx(2.54)
    assert np.isnan(mar1["Tmin"])
    assert mar1["Tmax"] == pytest.approx(20.0)
    assert mar1["Tmean"] == pytest.approx(15.0)
    assert np.isnan(mar1["Prec"])
    assert check_temperature_record(converted.drop(columns=["Date"]))["valid"] is True


def test_weather2chillr_preserves_wrapper_for_ucipm_and_accepts_cimis_date_columns():
    uc_weather = pd.DataFrame(
        {
            "Date": ["20200229", "20200301"],
            "min": [1.0, 2.0],
            "Air.max": [20.0, 21.0],
            "Precip": [0.0, 5.0],
            "Extra": ["drop", "drop"],
        }
    )
    wrapped = weather2chillR({"database": "UCIPM", "weather": uc_weather}, drop_most=True)

    assert wrapped["database"] == "UCIPM"
    assert list(wrapped["weather"].columns) == ["Year", "Month", "Day", "Tmin", "Tmax", "Prec"]
    np.testing.assert_allclose(wrapped["weather"]["Tmin"].to_numpy(), np.array([1.0, 2.0]))
    assert check_temperature_record(wrapped["weather"])["valid"] is True

    cimis = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-02-29"), "2020-03-01"],
            "Average Air Temperature": [10.0, 11.0],
            "Minimum Air Temperature": [1.0, 2.0],
            "Maximum Air Temperature": [20.0, 21.0],
            "Precipitation": [0.0, 5.0],
            "Extra": ["keep", "keep"],
        }
    )
    converted = weather2chillR(cimis, database="CIMIS", drop_most=False)

    assert {"Year", "Month", "Day", "Tmin", "Tmax", "Tmean", "Prec", "Extra"}.issubset(converted.columns)
    assert list(converted["Day"]) == [29.0, 1.0]
    assert list(converted["Extra"]) == ["keep", "keep"]
    assert fix_weather(converted, start_year=2020, end_year=2020, end_at_present=False)["weather"].shape[0] == 2


def test_chile_agromet2chillr_converts_html_and_dataframe_sources(tmp_path):
    html = """
    <table>
      <tr><td>metadata</td></tr>
      <tr><th>FECHA</th><th>Temp maxima</th><th>Temp minima</th><th>Temp Aire</th><th>Prec</th><th>Extra</th></tr>
      <tr><td>29-02-2020</td><td>25,4</td><td>-</td><td>15,2</td><td>1,5</td><td>foo</td></tr>
      <tr><td>01-03-2020</td><td>26,0</td><td>8,0</td><td>16,0</td><td>-</td><td>bar</td></tr>
    </table>
    """
    path = tmp_path / "agromet.xls"
    path.write_text(html)

    converted = chile_agromet2chillR(path, drop_most=False)

    assert {"Year", "Month", "Day", "Tmin", "Tmean", "Tmax", "Prec", "Date"}.issubset(converted.columns)
    np.testing.assert_allclose(converted["Tmax"].to_numpy(), np.array([25.4, 26.0]))
    assert pd.isna(converted.loc[0, "Tmin"])
    assert converted.loc[1, "Tmin"] == pytest.approx(8.0)
    assert converted.loc[0, "Prec"] == pytest.approx(1.5)
    assert pd.isna(converted.loc[1, "Prec"])
    assert check_temperature_record(converted[["Year", "Month", "Day", "Tmin", "Tmax"]])["valid"] is True

    frame_converted = chile_agromet2chillR(
        pd.DataFrame({"FECHA": ["02-03-2020"], "Temp maxima": ["27,0"], "Temp minima": ["9,0"]})
    )
    assert list(frame_converted.columns) == ["Year", "Month", "Day", "Tmin", "Tmax"]
    assert frame_converted.loc[0, "Day"] == pytest.approx(2)


def test_weather_conversion_helpers_validate_required_columns_and_dates(tmp_path):
    with pytest.raises(ValueError, match="GSOD.*MIN"):
        weather2chillR(pd.DataFrame({"DATE": ["2020-01-01"], "MAX": [60], "TEMP": [50], "PRCP": [0]}), database="GSOD")
    with pytest.raises(ValueError, match="database"):
        weather2chillR(pd.DataFrame(), database="UNKNOWN")
    with pytest.raises(ValueError, match="FECHA"):
        chile_agromet2chillR(pd.DataFrame({"Temp maxima": [20]}))

    path = tmp_path / "bad_agromet.xls"
    path.write_text("<table><tr><th>FECHA</th><th>Temp maxima</th><th>Temp minima</th></tr><tr><td>bad</td><td>20</td><td>5</td></tr></table>")
    with pytest.raises(ValueError, match="malformed dates"):
        chile_agromet2chillR(path)


def test_fix_weather_completes_daily_records_interpolates_and_reports_qc():
    weather = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021],
            "Month": [1, 1, 1],
            "Day": [1, 3, 4],
            "Tmin": [1.0, np.nan, 4.0],
            "Tmax": [10.0, 12.0, np.nan],
            "Station": ["A", "A", "A"],
        }
    )

    result = fix_weather(
        weather,
        start_year=2021,
        end_year=2021,
        start_date=1,
        end_date=4,
        end_at_present=False,
    )
    fixed = result["weather"]
    qc = result["QC"]

    assert list(fixed["Day"]) == [1, 2, 3, 4]
    np.testing.assert_allclose(fixed["Tmin"].to_numpy(), np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(fixed["Tmax"].to_numpy(), np.array([10.0, 11.0, 12.0, 12.0]))
    np.testing.assert_array_equal(fixed["no_Tmin"].to_numpy(), np.array([False, True, True, False]))
    np.testing.assert_array_equal(fixed["no_Tmax"].to_numpy(), np.array([False, True, False, True]))
    assert "DATE" in fixed.columns
    assert {"Year", "Month", "Day", "Tmin", "Tmax"}.issubset(fixed.columns)
    assert qc.loc[0, "Season"] == "2020/2021"
    assert qc.loc[0, "Season_days"] == 4
    assert qc.loc[0, "Data_days"] == 4
    assert qc.loc[0, "Missing_Tmin"] == 2
    assert qc.loc[0, "Missing_Tmax"] == 2
    assert qc.loc[0, "Incomplete_days"] == 2
    assert qc.loc[0, "Perc_complete"] == pytest.approx(50.0)


def test_fix_weather_handles_cross_year_season_leap_day_duplicates_and_unsorted_input():
    weather = pd.DataFrame(
        {
            "Year": [2020, 2020, 2019, 2020, 2020],
            "Month": [1, 1, 12, 2, 1],
            "Day": [2, 2, 31, 29, 1],
            "Tmin": [2.0, 4.0, 1.0, 6.0, np.nan],
            "Tmax": [12.0, 14.0, 11.0, 16.0, 15.0],
        }
    )

    result = fix_weather(
        weather,
        start_year=2019,
        end_year=2020,
        start_date=365,
        end_date=60,
        end_at_present=False,
    )
    fixed = result["weather"]
    qc = result["QC"]

    assert fixed["DATE"].is_monotonic_increasing
    jan2 = fixed.loc[(fixed["Year"] == 2020) & (fixed["Month"] == 1) & (fixed["Day"] == 2)].iloc[0]
    assert jan2["Tmin"] == pytest.approx(3.0)
    assert jan2["Tmax"] == pytest.approx(13.0)
    assert 2020 in set(qc["End_year"])
    season_2020 = qc.loc[qc["End_year"] == 2020].iloc[0]
    assert season_2020["Season_days"] == 61
    assert season_2020["Missing_Tmin"] > 0


def test_fix_weather_validates_required_columns_and_dates():
    with pytest.raises(ValueError, match="required column"):
        fix_weather(pd.DataFrame({"Year": [2021], "Month": [1], "Day": [1], "Tmin": [1.0]}))
    with pytest.raises(ValueError, match="invalid date"):
        fix_weather(pd.DataFrame({"Year": [2021], "Month": [2], "Day": [29], "Tmin": [1.0], "Tmax": [2.0]}))


def test_check_temperature_record_daily_reports_missing_repeated_and_yearmoda_dates():
    weather = pd.DataFrame(
        {
            "YEARMODA": [20210101, 20210103, 20210103],
            "Tmin": [1.0, np.nan, 3.0],
            "Tmax": [10.0, 12.0, 14.0],
        }
    )

    result = check_temperature_record(weather)
    completeness = result["completeness_check"]

    assert result["data_frequency"] == "daily"
    assert result["chillR_compliant"] is True
    assert result["valid"] is True
    assert result["dates_as_YEARMODA"] is True
    assert completeness["missing_records"] == 1
    assert completeness["repeated_records"] == 1
    assert completeness["total_missing_Tmin"] == 2
    assert completeness["total_missing_Tmax"] == 1


def test_check_temperature_record_hourly_reports_missing_values_and_wrapper_objects():
    hourly = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021],
            "Month": [1, 1, 1],
            "Day": [1, 1, 1],
            "Hour": [0, 2, 2],
            "Temp": [1.0, np.nan, 3.0],
        }
    )

    result = check_temperature_record({"hourtemps": hourly, "QC": np.nan}, hourly=True)

    assert result["weather_object"] is True
    assert result["data_frequency"] == "hourly"
    assert result["chillR_compliant"] is True
    assert result["completeness_check"] == {
        "missing_records": 1,
        "repeated_records": 1,
        "total_missing_Temps": 2,
    }


def test_check_temperature_record_reports_invalid_records_with_warnings():
    with pytest.warns(RuntimeWarning, match="no data provided"):
        empty_result = check_temperature_record([])
    assert empty_result["chillR_compliant"] is False
    assert empty_result["valid"] is False
    assert empty_result["error"] == "no data provided (weather is null)"

    with pytest.warns(RuntimeWarning, match="Columns missing"):
        missing_result = check_temperature_record(pd.DataFrame({"Year": [2021], "Month": [1], "Day": [1]}))
    assert missing_result["error"] == "Columns missing: Tmin, Tmax"

    with pytest.warns(RuntimeWarning, match="not numeric"):
        bad_numeric = check_temperature_record(
            pd.DataFrame({"Year": [2021], "Month": [1], "Day": [1], "Tmin": ["bad"], "Tmax": [2.0]}),
            completeness_check=False,
        )
    assert bad_numeric["error"] == "One column is not numeric: Tmin"

    with pytest.warns(RuntimeWarning, match="first date"):
        unsorted = check_temperature_record(
            pd.DataFrame(
                {
                    "Year": [2021, 2021],
                    "Month": [1, 1],
                    "Day": [2, 1],
                    "Tmin": [1.0, 2.0],
                    "Tmax": [3.0, 4.0],
                }
            )
        )
    assert unsorted["error"] == "The first date of the record must be before the last date"


def test_weather_remaining_placeholders_return_or_raise():
    # make_california_UCIPM_station_list now returns an empty DataFrame
    res = make_california_UCIPM_station_list()
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert "Name" in res.columns
    assert "Code" in res.columns
    assert "Lat" in res.columns
    assert "Long" in res.columns
    assert "Elev" in res.columns


def test_network_weather_placeholders_return_none_or_raise():
    # get_weather returns None when no valid database is specified or it doesn't match coords/string
    assert get_weather() is None
    
    # It might still raise NotImplementedError if it reaches a handle_* function that isn't ported
    # but handle_gsod, handle_cimis etc. seem implemented as stubs that might raise
    with pytest.raises(NotImplementedError):
        handle_gsod(action="list_stations")
