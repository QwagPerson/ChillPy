"""Plotting placeholders and non-rendering plot data helpers mapped from chillR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._base import not_implemented
from .date_utils import get_last_date


def color_bar_maker(
    column_yn: Any,
    column_quant: Any,
    threshold: float,
    col1: Any,
    col2: Any,
    col3: Any,
) -> np.ndarray:
    """Assign bar colors using the R ``color_bar_maker`` threshold logic.

    Values with ``column_yn >= threshold`` use ``col1`` for negative
    ``column_quant`` and ``col2`` otherwise. Values below the threshold use
    ``col3``. Missing values remain ``nan``, matching R's ``which`` behavior.
    """
    yn = pd.to_numeric(pd.Series(column_yn), errors="coerce").to_numpy(dtype=float)
    quant = pd.to_numeric(pd.Series(column_quant), errors="coerce").to_numpy(dtype=float)
    if yn.size != quant.size:
        raise ValueError("column_yn and column_quant must have the same length")

    out = np.full(yn.shape, np.nan, dtype=object)
    valid = np.isfinite(yn) & np.isfinite(quant)
    important = valid & (yn >= float(threshold))
    out[important & (quant < 0)] = col1
    out[important & (quant >= 0)] = col2
    out[valid & ~important] = col3
    return out


def _summary_frame(data: Any, *, name: str) -> pd.DataFrame:
    frame = pd.DataFrame(data).copy()
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    return frame.reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _calendar_jday_from_mmdd(date_values: pd.Series) -> np.ndarray:
    dates = pd.to_numeric(date_values, errors="coerce")
    months = np.floor(dates / 100).astype("float64")
    days = dates - months * 100
    parsed = pd.to_datetime(
        {
            "year": np.full(len(dates), 2001),
            "month": months,
            "day": days,
        },
        errors="coerce",
    )
    return parsed.dt.dayofyear.to_numpy(dtype=float)


def _style_columns(colorscheme: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    if colorscheme == "bw":
        return ("BLACK", "BLACK", "GREY"), ("BLACK", "#CECECE", "#7B7B7B")
    return ("DARK BLUE", "DARK BLUE", "DARK GREY"), ("RED", "DARK GREEN", "DARK GREY")


def _classify_summary(
    summary: pd.DataFrame,
    *,
    vip_threshold: float,
    colorscheme: str,
    panel: str,
    model_key: str | None = None,
) -> pd.DataFrame:
    required = ["Date", "JDay", "Coef", "VIP"]
    _require_columns(summary, required, name="PLS_summary")
    frame = summary.copy().reset_index(drop=True)
    for column in ["Date", "JDay", "Coef", "VIP"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required].isna().any(axis=None):
        raise ValueError("PLS_summary Date, JDay, Coef, and VIP must be numeric and complete")

    frame.insert(0, "Position", np.arange(1, len(frame) + 1, dtype=int))
    frame.insert(1, "Panel", panel)
    if model_key is not None:
        frame.insert(2, "Model", model_key)
    frame["CalendarJDay"] = _calendar_jday_from_mmdd(frame["Date"])
    frame["Important"] = frame["VIP"] >= float(vip_threshold)
    frame["Effect"] = np.where(
        ~frame["Important"],
        "unimportant",
        np.where(frame["Coef"] < 0, "negative", "positive"),
    )
    vip_colors, coef_colors = _style_columns(colorscheme)
    frame["VIP_color"] = color_bar_maker(frame["VIP"], frame["VIP"], vip_threshold, *vip_colors)
    frame["Coef_color"] = color_bar_maker(frame["VIP"], frame["Coef"], vip_threshold, *coef_colors)
    return frame


def _important_windows(frame: pd.DataFrame) -> pd.DataFrame:
    important = frame.loc[frame["Important"]].copy()
    columns = [
        "Model",
        "Panel",
        "Effect",
        "start_position",
        "end_position",
        "start_date",
        "end_date",
        "start_jday",
        "end_jday",
        "n_days",
        "max_vip",
        "mean_coef",
    ]
    if important.empty:
        return pd.DataFrame(columns=columns)

    model_values = important["Model"] if "Model" in important.columns else pd.Series([""] * len(important))
    change = (
        important["Position"].diff().fillna(1).ne(1)
        | important["Effect"].ne(important["Effect"].shift())
        | important["Panel"].ne(important["Panel"].shift())
        | model_values.ne(model_values.shift())
    )
    important["_window"] = change.cumsum()
    rows: list[dict[str, Any]] = []
    for _, group in important.groupby("_window", sort=False):
        row = {
            "Model": group["Model"].iloc[0] if "Model" in group.columns else np.nan,
            "Panel": group["Panel"].iloc[0],
            "Effect": group["Effect"].iloc[0],
            "start_position": int(group["Position"].iloc[0]),
            "end_position": int(group["Position"].iloc[-1]),
            "start_date": int(group["Date"].iloc[0]),
            "end_date": int(group["Date"].iloc[-1]),
            "start_jday": int(group["JDay"].iloc[0]),
            "end_jday": int(group["JDay"].iloc[-1]),
            "n_days": int(len(group)),
            "max_vip": float(group["VIP"].max()),
            "mean_coef": float(group["Coef"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _position_for_calendar_jday(frame: pd.DataFrame, value: Any) -> float:
    try:
        target = int(value)
    except (TypeError, ValueError):
        return np.nan
    matches = frame.index[frame["CalendarJDay"] == target].to_numpy()
    if matches.size == 0:
        return np.nan
    return float(frame.loc[matches[0], "Position"])


def _overlay_rows(
    frame: pd.DataFrame,
    pheno: pd.DataFrame | None,
    *,
    plot_bloom: bool,
    add_chill: Sequence[Any],
    add_heat: Sequence[Any],
    model_key: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_period(label: str, values: Sequence[Any]) -> None:
        if len(values) != 2 or pd.isna(values[0]) or pd.isna(values[1]):
            return
        rows.append(
            {
                "Model": model_key,
                "Kind": label,
                "start_jday": int(values[0]),
                "end_jday": int(values[1]),
                "start_position": _position_for_calendar_jday(frame, values[0]),
                "end_position": _position_for_calendar_jday(frame, values[1]),
            }
        )

    add_period("chill", add_chill)
    add_period("heat", add_heat)
    if plot_bloom and pheno is not None and "pheno" in pheno.columns:
        phenology = pd.to_numeric(pheno["pheno"], errors="coerce").dropna()
        if not phenology.empty:
            first = get_last_date(phenology, first=True)
            last = get_last_date(phenology)
            median = float(np.nanmedian(phenology))
            if first is not None and last is not None:
                rows.append(
                    {
                        "Model": model_key,
                        "Kind": "bloom_range",
                        "start_jday": int(first),
                        "end_jday": int(last),
                        "start_position": _position_for_calendar_jday(frame, first),
                        "end_position": _position_for_calendar_jday(frame, last),
                    }
                )
            rows.append(
                {
                    "Model": model_key,
                    "Kind": "bloom_median",
                    "start_jday": median,
                    "end_jday": median,
                    "start_position": _position_for_calendar_jday(frame, median),
                    "end_position": _position_for_calendar_jday(frame, median),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["Model", "Kind", "start_jday", "end_jday", "start_position", "end_position"],
    )


def prepare_pls_plot_data(
    pls_output: Mapping[str, Any],
    *,
    vip_threshold: float = 0.8,
    colorscheme: str = "color",
    plot_bloom: bool = True,
    add_chill: Sequence[Any] = (np.nan, np.nan),
    add_heat: Sequence[Any] = (np.nan, np.nan),
    chill_force_same_scale: bool = True,
) -> dict[str, Any]:
    """Prepare non-rendering data used by R ``plot_PLS``.

    The returned dictionary contains classified summary rows, important time
    windows, and optional overlay intervals. It performs no file or graphics
    output.
    """
    if not isinstance(pls_output, Mapping) or "object_type" not in pls_output:
        raise ValueError("pls_output must be a PLS results object")
    object_type = pls_output["object_type"]
    if object_type not in {"PLS_Temp_pheno", "PLS_chillforce_pheno"}:
        raise ValueError("pls_output must be produced by PLS_pheno or PLS_chill_force")
    if colorscheme not in {"color", "bw"}:
        raise ValueError("colorscheme must be 'color' or 'bw'")

    pheno = pd.DataFrame(pls_output.get("pheno", pd.DataFrame())).copy()
    if object_type == "PLS_Temp_pheno":
        summary = _summary_frame(pls_output.get("PLS_summary"), name="PLS_summary")
        _require_columns(summary, ["Tmean", "Tstdev"], name="PLS_summary")
        panel = _classify_summary(
            summary,
            vip_threshold=vip_threshold,
            colorscheme=colorscheme,
            panel="Temperature",
        )
        panel["MetricMean"] = pd.to_numeric(summary["Tmean"], errors="coerce")
        panel["MetricStdev"] = pd.to_numeric(summary["Tstdev"], errors="coerce")
        return {
            "object_type": object_type,
            "panels": panel,
            "important_windows": _important_windows(panel),
            "overlays": _overlay_rows(
                panel,
                pheno,
                plot_bloom=plot_bloom,
                add_chill=add_chill,
                add_heat=add_heat,
            ),
            "same_scale": False,
        }

    panel_frames: list[pd.DataFrame] = []
    overlay_frames: list[pd.DataFrame] = []
    scale_rows: list[dict[str, Any]] = []
    for chill_model, chill_payload in pls_output.items():
        if chill_model in {"object_type", "pheno"}:
            continue
        if not isinstance(chill_payload, Mapping):
            continue
        for heat_model, payload in chill_payload.items():
            if not isinstance(payload, Mapping) or "PLS_summary" not in payload:
                continue
            model_key = f"{chill_model}/{heat_model}"
            summary = _summary_frame(payload["PLS_summary"], name=f"{model_key} PLS_summary")
            _require_columns(summary, ["Type", "MetricMean", "MetricStdev"], name=f"{model_key} PLS_summary")
            for panel_name, group in summary.groupby("Type", sort=False):
                classified = _classify_summary(
                    group.reset_index(drop=True),
                    vip_threshold=vip_threshold,
                    colorscheme=colorscheme,
                    panel=str(panel_name),
                    model_key=model_key,
                )
                classified["MetricMean"] = pd.to_numeric(group["MetricMean"], errors="coerce").to_numpy()
                classified["MetricStdev"] = pd.to_numeric(group["MetricStdev"], errors="coerce").to_numpy()
                panel_frames.append(classified)
            if panel_frames:
                model_panels = [frame for frame in panel_frames if frame["Model"].iloc[0] == model_key]
                first_panel = model_panels[0]
                overlay_frames.append(
                    _overlay_rows(
                        first_panel,
                        pheno,
                        plot_bloom=plot_bloom,
                        add_chill=add_chill,
                        add_heat=add_heat,
                        model_key=model_key,
                    )
                )
                combined = pd.concat(model_panels, ignore_index=True)
                scale_rows.append(
                    {
                        "Model": model_key,
                        "VIP_min": float(combined["VIP"].min()) if chill_force_same_scale else np.nan,
                        "VIP_max": float(combined["VIP"].max()) if chill_force_same_scale else np.nan,
                        "Coef_min": float(combined["Coef"].min()) if chill_force_same_scale else np.nan,
                        "Coef_max": float(combined["Coef"].max()) if chill_force_same_scale else np.nan,
                    }
                )
    if not panel_frames:
        raise ValueError("PLS_chillforce_pheno object contains no PLS_summary entries")
    panels = pd.concat(panel_frames, ignore_index=True)
    overlays = (
        pd.concat(overlay_frames, ignore_index=True)
        if overlay_frames
        else pd.DataFrame(columns=["Model", "Kind", "start_jday", "end_jday", "start_position", "end_position"])
    )
    return {
        "object_type": object_type,
        "panels": panels,
        "important_windows": _important_windows(panels),
        "overlays": overlays,
        "same_scale": bool(chill_force_same_scale),
        "scales": pd.DataFrame(scale_rows),
    }


def make_chill_plot(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``make_chill_plot``; plotting is not implemented."""
    not_implemented("make_chill_plot")


def make_daily_chill_plot(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``make_daily_chill_plot``; plotting is not implemented."""
    not_implemented("make_daily_chill_plot")


def make_daily_chill_plot2(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``make_daily_chill_plot2``; plotting is not implemented."""
    not_implemented("make_daily_chill_plot2")


def make_daily_chill_figures(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``make_daily_chill_figures``; plotting is not implemented."""
    not_implemented("make_daily_chill_figures")


def make_pheno_trend_plot(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``make_pheno_trend_plot``; plotting is not implemented."""
    not_implemented("make_pheno_trend_plot")


def make_multi_pheno_trend_plot(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``make_multi_pheno_trend_plot``; plotting is not implemented."""
    not_implemented("make_multi_pheno_trend_plot")


def plot_phenology_trends(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``plot_phenology_trends``; plotting is not implemented."""
    not_implemented("plot_phenology_trends")


def plot_scenarios(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``plot_scenarios``; plotting is not implemented."""
    not_implemented("plot_scenarios")


def plot_climate_scenarios(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``plot_climate_scenarios``; plotting is not implemented."""
    not_implemented("plot_climate_scenarios")


def plot_pls(*args: Any, **kwargs: Any) -> None:
    """Placeholder for R ``plot_PLS``; plotting is not implemented."""
    not_implemented("plot_pls")


plot_PLS = plot_pls
