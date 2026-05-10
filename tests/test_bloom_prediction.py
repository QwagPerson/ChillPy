import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from chillPy import bloom_prediction, bloom_prediction2, bloom_prediction3, chilling_hourtable, chilling_hours, gdh


def _chill_table():
    return pd.DataFrame(
        {
            "Year": [2020] * 6,
            "Month": [1] * 6,
            "Day": [1, 2, 3, 4, 5, 6],
            "JDay": [1, 2, 3, 4, 5, 6],
            "Season": [2020] * 6,
            "Chill_Portions": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            "GDH": [0.0, 0.0, 3.0, 6.0, 9.0, 12.0],
        }
    )


def test_bloom_prediction_scalar_returns_r_style_fulfillment_rows():
    result = bloom_prediction(_chill_table(), chill_req=12, heat_req=4, start_jday=1)

    assert result.to_dict("records") == [
        {
            "Season": 2020,
            "Creqfull": 4,
            "Creq_year": 2020,
            "Creq_month": 1,
            "Creq_day": 4,
            "Creq_JDay": 4,
            "Hreqfull": 3,
            "Hreq_year": 2020,
            "Hreq_month": 1,
            "Hreq_day": 6,
            "Hreq_JDay": 6,
        }
    ]


def test_bloom_prediction2_pairs_requirements_and_preserves_infocol():
    result = bloom_prediction2(
        _chill_table(),
        chill_req=[12, 18],
        heat_req=[4, 2],
        start_jday=1,
        infocol=["green_tip", "bloom"],
    )

    expected = pd.DataFrame(
        {
            "infocol": ["green_tip", "bloom"],
            "Season": [2020, 2020],
            "Creq": [12.0, 18.0],
            "Hreq": [4.0, 2.0],
            "Chill_comp": [3.0, 4.0],
            "Pheno_date": [4.0, 4.0],
        }
    )
    assert_frame_equal(result, expected)


def test_bloom_prediction2_permutations_and_missing_fulfillment():
    result = bloom_prediction2(
        _chill_table(),
        chill_req=[12, 30],
        heat_req=[2, 4],
        permutations=True,
        start_jday=1,
    )

    assert len(result) == 4
    assert result.loc[(result["Creq"] == 12) & (result["Hreq"] == 2), "Pheno_date"].iloc[0] == pytest.approx(3)
    missing = result.loc[result["Creq"] == 30]
    assert missing["Chill_comp"].isna().all()
    assert missing["Pheno_date"].isna().all()


def cumulative_chill(temps, summ=True):
    values = np.full(len(temps), 5.0)
    return np.cumsum(values) if summ else values


def cumulative_heat(temps, summ=True):
    values = np.full(len(temps), 3.0)
    return np.cumsum(values) if summ else values


def test_bloom_prediction3_computes_metrics_from_hourly_temperatures():
    hourtemps = pd.DataFrame(
        {
            "Year": [2020] * 6,
            "Month": [1] * 6,
            "Day": [1, 2, 3, 4, 5, 6],
            "JDay": [1, 2, 3, 4, 5, 6],
            "Hour": [0] * 6,
            "Temp": [1.0] * 6,
        }
    )

    result = bloom_prediction3(
        {"hourtemps": hourtemps},
        chill_req=[12],
        heat_req=[4],
        models={"CH": cumulative_chill, "Heat": cumulative_heat},
        chill_model="CH",
        heat_model="Heat",
        start_jday=1,
    )

    expected = pd.DataFrame(
        {
            "Season": [2020],
            "Creq": [12.0],
            "Hreq": [4.0],
            "Chill_comp": [3.0],
            "Pheno_date": [4.0],
            "Chill_comp_YEARMODA": [20200103],
            "Pheno_YEARMODA": [20200104],
        }
    )
    assert_frame_equal(result, expected)


def test_bloom_prediction3_integrates_with_chilling_hourtable_defaults():
    hourtemps = pd.DataFrame(
        {
            "Year": [2020] * 6,
            "Month": [1] * 6,
            "Day": [1, 2, 3, 4, 5, 6],
            "JDay": [1, 2, 3, 4, 5, 6],
            "Hour": [0] * 6,
            "Temp": [5.0] * 6,
        }
    )
    table = chilling_hourtable(hourtemps, start_jday=1)

    direct = bloom_prediction2(
        table,
        chill_req=[2],
        heat_req=[1],
        chill_model="Chilling_Hours",
        heat_model="GDH",
        start_jday=1,
    )
    wrapped = bloom_prediction3(
        hourtemps,
        chill_req=[2],
        heat_req=[1],
        models={"Chilling_Hours": chilling_hours, "GDH": gdh},
        chill_model="Chilling_Hours",
        heat_model="GDH",
        start_jday=1,
    )

    assert_frame_equal(wrapped[["Season", "Creq", "Hreq", "Chill_comp", "Pheno_date"]], direct)
    assert {"Chill_comp_YEARMODA", "Pheno_YEARMODA"}.issubset(wrapped.columns)


def test_bloom_prediction_validates_inputs():
    table = _chill_table()

    with pytest.raises(ValueError, match="missing required"):
        bloom_prediction(table.drop(columns="GDH"), chill_req=1, heat_req=1, start_jday=1)
    with pytest.raises(ValueError, match="start_jday"):
        bloom_prediction(table, chill_req=1, heat_req=1, start_jday=367)
    with pytest.raises(ValueError, match="different length"):
        bloom_prediction2(table, chill_req=[1, 2], heat_req=[1], start_jday=1)
    with pytest.raises(ValueError, match="metric not calculated"):
        bloom_prediction3(
            pd.DataFrame(
                {
                    "Year": [2020],
                    "Month": [1],
                    "Day": [1],
                    "JDay": [1],
                    "Hour": [0],
                    "Temp": [5.0],
                }
            ),
            chill_req=[1],
            heat_req=[1],
            models={"CH": cumulative_chill},
            chill_model="CH",
            heat_model="Heat",
            start_jday=1,
        )
