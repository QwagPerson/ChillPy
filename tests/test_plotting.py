import pytest
import numpy as np
import pandas as pd

from chillPy import PLS_pheno, color_bar_maker, make_chill_plot, plot_PLS, prepare_pls_plot_data


def test_remaining_plotting_placeholders_raise():
    with pytest.raises(NotImplementedError):
        make_chill_plot()
    with pytest.raises(NotImplementedError):
        plot_PLS()


def test_color_bar_maker_matches_r_threshold_logic():
    colors = color_bar_maker(
        [0.7, 0.8, 1.2, np.nan],
        [-1.0, -0.1, 0.0, 2.0],
        0.8,
        "RED",
        "GREEN",
        "GREY",
    )

    assert colors.tolist()[:3] == ["GREY", "RED", "GREEN"]
    assert pd.isna(colors[3])


def test_prepare_pls_plot_data_for_temperature_summary_classifies_windows_and_overlays():
    summary = pd.DataFrame(
        {
            "Date": [1201, 1202, 1203, 1204],
            "JDay": [-31, -30, -29, -28],
            "Coef": [-2.0, -1.0, 3.0, 0.5],
            "VIP": [0.7, 0.9, 1.1, 0.6],
            "Tmean": [4.0, 5.0, 6.0, 7.0],
            "Tstdev": [0.4, 0.5, 0.6, 0.7],
        }
    )
    pls_output = {
        "object_type": "PLS_Temp_pheno",
        "pheno": pd.DataFrame({"Year": [2020, 2021, 2022], "pheno": [100, 110, 120]}),
        "PLS_summary": summary,
    }

    prepared = prepare_pls_plot_data(
        pls_output,
        vip_threshold=0.8,
        add_chill=(335, 338),
        add_heat=(336, 337),
    )

    panels = prepared["panels"]
    assert prepared["object_type"] == "PLS_Temp_pheno"
    assert panels["Important"].tolist() == [False, True, True, False]
    assert panels["Effect"].tolist() == ["unimportant", "negative", "positive", "unimportant"]
    assert panels["Coef_color"].tolist() == ["DARK GREY", "RED", "DARK GREEN", "DARK GREY"]
    assert panels["CalendarJDay"].tolist() == [335.0, 336.0, 337.0, 338.0]

    windows = prepared["important_windows"]
    assert windows[["Effect", "start_position", "end_position", "n_days"]].to_dict("records") == [
        {"Effect": "negative", "start_position": 2, "end_position": 2, "n_days": 1},
        {"Effect": "positive", "start_position": 3, "end_position": 3, "n_days": 1},
    ]

    overlays = prepared["overlays"].set_index("Kind")
    assert overlays.loc["chill", "start_position"] == pytest.approx(1.0)
    assert overlays.loc["chill", "end_position"] == pytest.approx(4.0)
    assert overlays.loc["heat", "start_position"] == pytest.approx(2.0)
    assert overlays.loc["heat", "end_position"] == pytest.approx(3.0)
    assert "bloom_range" in overlays.index
    assert "bloom_median" in overlays.index


def test_prepare_pls_plot_data_for_chill_force_splits_metric_panels_and_scales():
    dates = [101, 102, 103]
    chill = pd.DataFrame(
        {
            "Date": dates,
            "Type": ["Chill"] * 3,
            "JDay": [1, 2, 3],
            "Coef": [-1.0, 2.0, -3.0],
            "VIP": [0.9, 0.7, 1.2],
            "MetricMean": [10.0, 11.0, 12.0],
            "MetricStdev": [1.0, 1.1, 1.2],
        }
    )
    heat = pd.DataFrame(
        {
            "Date": dates,
            "Type": ["Heat"] * 3,
            "JDay": [1, 2, 3],
            "Coef": [3.0, -2.0, 1.0],
            "VIP": [0.6, 1.1, 1.3],
            "MetricMean": [20.0, 21.0, 22.0],
            "MetricStdev": [2.0, 2.1, 2.2],
        }
    )
    pls_output = {
        "object_type": "PLS_chillforce_pheno",
        "pheno": pd.DataFrame({"Year": [2020, 2021], "pheno": [2, 3]}),
        "CH": {"GDH": {"PLS_summary": pd.concat([chill, heat], ignore_index=True)}},
    }

    prepared = prepare_pls_plot_data(pls_output, vip_threshold=0.8, colorscheme="bw")

    panels = prepared["panels"]
    assert set(panels["Panel"]) == {"Chill", "Heat"}
    assert panels["Model"].unique().tolist() == ["CH/GDH"]
    assert panels.loc[panels["Panel"] == "Chill", "Coef_color"].tolist() == ["BLACK", "#7B7B7B", "BLACK"]
    assert panels.loc[panels["Panel"] == "Heat", "Coef_color"].tolist() == ["#7B7B7B", "BLACK", "#CECECE"]
    scales = prepared["scales"].iloc[0]
    assert scales["VIP_min"] == pytest.approx(0.6)
    assert scales["VIP_max"] == pytest.approx(1.3)
    assert scales["Coef_min"] == pytest.approx(-3.0)
    assert scales["Coef_max"] == pytest.approx(3.0)


def test_prepare_pls_plot_data_validates_inputs():
    with pytest.raises(ValueError, match="PLS_pheno"):
        prepare_pls_plot_data({"object_type": "not_pls"})
    with pytest.raises(ValueError, match="missing required"):
        prepare_pls_plot_data({"object_type": "PLS_Temp_pheno", "PLS_summary": pd.DataFrame({"VIP": [1]})})
    with pytest.raises(ValueError, match="colorscheme"):
        prepare_pls_plot_data(
            {
                "object_type": "PLS_Temp_pheno",
                "PLS_summary": pd.DataFrame(
                    {"Date": [101], "JDay": [1], "Coef": [1], "VIP": [1], "Tmean": [1], "Tstdev": [0]}
                ),
            },
            colorscheme="sepia",
        )


def test_prepare_pls_plot_data_accepts_pls_pheno_output():
    rows = []
    for year in [2017, 2018, 2019, 2020]:
        for day, date in enumerate(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")[:365], start=1):
            temp = 8.0 + (year - 2017) * 0.2 + np.sin(day / 30.0)
            rows.append(
                {
                    "Year": date.year,
                    "Month": date.month,
                    "Day": date.day,
                    "Tmin": temp - 2.0,
                    "Tmax": temp + 2.0,
                }
            )
    pls = PLS_pheno(
        pd.DataFrame(rows),
        pd.DataFrame({"Year": [2017, 2018, 2019, 2020], "pheno": [90, 92, 94, 96]}),
        split_month=12,
        ncomp_fix=1,
        runn_mean=1,
        end_at_pheno_end=100,
    )

    prepared = prepare_pls_plot_data(pls, vip_threshold=0.8)

    assert len(prepared["panels"]) == 100
    assert {"Position", "Important", "Effect", "VIP_color", "Coef_color"}.issubset(prepared["panels"].columns)
