"""Plotting helpers mapped from chillR, rendered with matplotlib."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .date_utils import get_last_date


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _plot_result(data: Any, figure: Any, axes: Any, **extra: Any) -> dict[str, Any]:
    return {"data": data, "figure": figure, "axes": axes, **extra}


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


def _as_frame_list(data: Any, *, name: str) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(data, Mapping):
        return [(str(key), pd.DataFrame(value).copy()) for key, value in data.items()]
    if isinstance(data, pd.DataFrame):
        return [("0", data.copy())]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [(str(index), pd.DataFrame(value).copy()) for index, value in enumerate(data)]
    raise ValueError(f"{name} must be a DataFrame, mapping, or sequence of DataFrames")


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


def make_chill_plot(
    daily_chill_obj: dict[str, Any],
    metrics: Sequence[str] | None = None,
    startdate: int = 1,
    enddate: int = 366,
    useyears: Sequence[int] | None = None,
    metriclabels: Sequence[str] | None = None,
    focusyears: Sequence[int] | str = "none",
    cumulative: bool = False,
    title: str | None = None,
    plotylim: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Plot daily climate metric accumulation and return data plus figure.

    Translates R ``make_chill_plot``.
    """
    if not isinstance(daily_chill_obj, Mapping) or "daily_chill" not in daily_chill_obj:
        raise ValueError("daily_chill_obj must contain a 'daily_chill' DataFrame")
    dc = daily_chill_obj["daily_chill"].copy()

    # Calculate JDay if not present
    if "JDay" not in dc.columns:
        dc["JDay"] = (
            pd.to_datetime(
                {"year": dc["Year"], "month": dc["Month"], "day": dc["Day"]}
            ).dt.dayofyear
        )

    # Filter out missing data if quality flags are present
    if "no_Tmin" in dc.columns and "no_Tmax" in dc.columns:
        last_valid = dc[~(dc["no_Tmin"] | dc["no_Tmax"])].index.max()
        if not pd.isna(last_valid):
            dc = dc.loc[:last_valid]

    # Define relevant days
    if enddate > startdate:
        relevant_days = list(range(startdate, enddate + 1))
    else:
        relevant_days = list(range(startdate, 367)) + list(range(1, enddate + 1))

    # Actual days present in data
    relevant_days = [d for d in relevant_days if d in dc["JDay"].unique()]

    # Assign End_year (season year)
    if enddate > startdate:
        dc.loc[dc["JDay"].isin(relevant_days), "End_year"] = dc["Year"]
    else:
        dc.loc[dc["JDay"].isin(range(1, enddate + 1)), "End_year"] = dc["Year"]
        dc.loc[dc["JDay"].isin(range(startdate, 367)), "End_year"] = dc["Year"] + 1

    if useyears is None:
        useyears = dc["End_year"].dropna().unique()

    dc = dc[dc["End_year"].isin(useyears)]

    if isinstance(focusyears, (list, tuple, np.ndarray)):
        focusyears = [f for f in focusyears if f in dc["End_year"].unique()]
        if not focusyears:
            focusyears = "none"
    elif focusyears != "none" and focusyears not in dc["End_year"].unique():
        focusyears = "none"

    if metrics is None:
        exclude = [
            "YYMMDD",
            "Year",
            "Month",
            "Day",
            "Tmean",
            "JDay",
            "End_year",
            "no_Tmin",
            "no_Tmax",
            "YEARMODA",
        ]
        metrics = [c for c in dc.columns if c not in exclude]

    if metriclabels is None:
        metriclabels = metrics

    # X-axis values (Julian days relative to start)
    if startdate < enddate:
        jdays_plot = relevant_days
    else:
        jdays_plot = [(d - 366 if d >= startdate else d) for d in relevant_days]

    output_dfs: dict[str, pd.DataFrame] = {}

    for met in metrics:
        met_dc = dc.copy()
        if cumulative:
            for year in met_dc["End_year"].unique():
                mask = (met_dc["End_year"] == year) & (met_dc["JDay"].isin(relevant_days))
                met_dc.loc[mask, met] = met_dc.loc[mask, met].cumsum()

        met_dc.loc[~met_dc["JDay"].isin(relevant_days), met] = 0

        # Calculate mean and SD per day
        summary_data = []
        for i, rday in enumerate(relevant_days):
            day_values = met_dc.loc[met_dc["JDay"] == rday, met]
            summary_data.append(
                {"JDay": jdays_plot[i], "Mean": day_values.mean(), "Sd": day_values.std()}
            )

        df_res = pd.DataFrame(summary_data).fillna(0)

        # Add focus years
        if focusyears != "none":
            for fy in focusyears:
                fy_data = met_dc[met_dc["End_year"] == fy]
                # Map back to JDay in df_res
                for i, rday in enumerate(relevant_days):
                    val = fy_data.loc[fy_data["JDay"] == rday, met]
                    if not val.empty:
                        df_res.loc[i, str(fy)] = val.iloc[0]

        output_dfs[met] = df_res

    plt = _pyplot()
    fig, axes = plt.subplots(
        len(output_dfs),
        1,
        figsize=(8, max(3, 2.8 * len(output_dfs))),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    label_lookup = dict(zip(metrics, metriclabels))
    for ax, (metric, frame) in zip(axes_flat, output_dfs.items()):
        x = frame["JDay"].to_numpy(dtype=float)
        mean = frame["Mean"].to_numpy(dtype=float)
        sd = frame["Sd"].to_numpy(dtype=float)
        ax.plot(x, mean, color="black", linewidth=1.8, label="Mean")
        ax.fill_between(x, mean - sd, mean + sd, color="0.80", alpha=0.8, label="Std. dev.")
        if focusyears != "none":
            for fy in focusyears:
                column = str(fy)
                if column in frame.columns:
                    ax.plot(x, frame[column].to_numpy(dtype=float), linewidth=1.2, label=column)
        ax.set_ylabel(
            f"Cumulative {label_lookup.get(metric, metric)}"
            if cumulative
            else f"{label_lookup.get(metric, metric)} per day"
        )
        ax.set_xlabel("Julian day")
        ax.set_title(title or f"{label_lookup.get(metric, metric)} accumulation")
        if plotylim is not None and len(plotylim) == 2 and not pd.isna(plotylim[0]):
            ax.set_ylim(plotylim)
        if focusyears != "none":
            ax.legend(frameon=False)
    fig.tight_layout()

    result = _plot_result(output_dfs, fig, axes_flat)
    result.update(output_dfs)
    return result


def make_daily_chill_plot(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for ``make_chill_plot``."""
    return make_chill_plot(*args, **kwargs)


def make_daily_chill_figures(
    daily_chill_obj: dict[str, Any],
    file_path: str | Path,
    models: Sequence[str] = ("Chilling_Hours", "Utah_Model", "Chill_Portions", "GDH"),
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Plot mean daily chill/heat accumulation and return summary data.

    Translates R ``make_daily_chill_figures``.
    """
    if "daily_chill" not in daily_chill_obj:
        raise ValueError("not a daily chill object")
    
    dc = daily_chill_obj["daily_chill"].copy()
    
    # Ensure JDay is present
    if "JDay" not in dc.columns:
        dc["JDay"] = pd.to_datetime(
            {"year": dc["Year"], "month": dc["Month"], "day": dc["Day"]}
        ).dt.dayofyear

    if labels is None:
        labels = models

    df_summary = pd.DataFrame({"JDay": range(1, 366)})
    
    for m, label in zip(models, labels):
        if m not in dc.columns:
            continue
            
        means = []
        sds = []
        for i in range(1, 366):
            day_data = dc[dc["JDay"] == i][m]
            means.append(day_data.mean())
            sds.append(day_data.std())
        
        df_summary[f"{m}_mean"] = means
        df_summary[f"{m}_sd"] = sds
        
        # Mann-Kendall test could be added here if scipy is available
        # For now, we omit it or return NaNs
        df_summary[f"{m}_Kendall_p"] = np.nan
        df_summary[f"{m}_Kendall_tau"] = np.nan

    plotted_models = [m for m in models if f"{m}_mean" in df_summary.columns]
    if not plotted_models:
        raise ValueError("none of the requested models are present in daily_chill")

    plt = _pyplot()
    fig, axes = plt.subplots(
        len(plotted_models),
        1,
        figsize=(8, max(3, 2.8 * len(plotted_models))),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    label_map = dict(zip(models, labels))
    for ax, model in zip(axes_flat, plotted_models):
        mean = df_summary[f"{model}_mean"].to_numpy(dtype=float)
        sd = df_summary[f"{model}_sd"].fillna(0).to_numpy(dtype=float)
        jday = df_summary["JDay"].to_numpy(dtype=float)
        ax.plot(jday, mean, color="black", linewidth=1.8)
        ax.fill_between(jday, mean - sd, mean + sd, color="0.80", alpha=0.8)
        ax.set_ylabel(f"{label_map.get(model, model)} per day")
        ax.set_xlabel("Julian day")
        ax.set_title(f"{label_map.get(model, model)} accumulation")
    fig.tight_layout()

    return {
        "daily_chill_figure_summary": df_summary,
        "figure": fig,
        "axes": axes_flat,
    }


def make_daily_chill_plot2(
    daily_chill_obj: dict[str, Any],
    metrics: Sequence[str] | None = None,
    startdate: int = 1,
    enddate: int = 366,
    useyears: Sequence[int] | None = None,
    metriclabels: Sequence[str] | None = None,
    focusyears: Sequence[int] | str = "none",
    cumulative: bool = False,
    title: str | None = None,
    plotylim: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Preparation of data for the second version of daily chill plots.

    Translates R ``make_daily_chill_plot2``.
    This is very similar to ``make_chill_plot`` but may have different defaults or layout.
    """
    return make_chill_plot(
        daily_chill_obj,
        metrics=metrics,
        startdate=startdate,
        enddate=enddate,
        useyears=useyears,
        metriclabels=metriclabels,
        focusyears=focusyears,
        cumulative=cumulative,
        title=title,
        plotylim=plotylim,
    )


def make_pheno_trend_plot(
    weather_data: pd.DataFrame,
    pheno_data: pd.DataFrame,
    pheno_column: str = "pheno",
    **kwargs: Any,
) -> dict[str, Any]:
    """Plot phenology observations and their linear trend.

    Translates R ``make_pheno_trend_plot``.
    """
    df = pheno_data.copy()
    _require_columns(df, ["Year", pheno_column], name="pheno_data")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[pheno_column] = pd.to_numeric(df[pheno_column], errors="coerce")
    mask = df[["Year", pheno_column]].notna().all(axis=1)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(df.loc[mask, "Year"], df.loc[mask, pheno_column], 1)
        df.loc[mask, "trend"] = slope * df.loc[mask, "Year"] + intercept
    else:
        df["trend"] = np.nan

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4)))
    ax.scatter(df["Year"], df[pheno_column], color=kwargs.get("point_color", "black"), label="Observed")
    if df["trend"].notna().any():
        trend = df.loc[df["trend"].notna()].sort_values("Year")
        ax.plot(trend["Year"], trend["trend"], color=kwargs.get("trend_color", "tab:red"), label="Trend")
    ax.set_xlabel(kwargs.get("x_axis_name", "Year"))
    ax.set_ylabel(kwargs.get("y_axis_name", pheno_column))
    if kwargs.get("title") is not None:
        ax.set_title(kwargs["title"])
    ax.legend(frameon=False)
    fig.tight_layout()
    return _plot_result(df, fig, ax)


def make_multi_pheno_trend_plot(
    weather_data: pd.DataFrame,
    pheno_data: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Plot one or more phenology trend columns against year.

    Translates R ``make_multi_pheno_trend_plot``.
    """
    df = pheno_data.copy()
    _require_columns(df, ["Year"], name="pheno_data")
    pheno_columns = kwargs.get("pheno_columns")
    if pheno_columns is None:
        pheno_columns = [c for c in df.columns if c != "Year" and pd.api.types.is_numeric_dtype(df[c])]
    pheno_columns = list(pheno_columns)
    if not pheno_columns:
        raise ValueError("no phenology columns available for plotting")

    plt = _pyplot()
    fig, axes = plt.subplots(
        len(pheno_columns),
        1,
        figsize=kwargs.get("figsize", (7, max(3, 2.8 * len(pheno_columns)))),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, column in zip(axes_flat, pheno_columns):
        result = make_pheno_trend_plot(weather_data, df[["Year", column]].rename(columns={column: "pheno"}))
        plot_df = result["data"]
        ax.scatter(plot_df["Year"], plot_df["pheno"], color="black")
        if plot_df["trend"].notna().any():
            trend = plot_df.loc[plot_df["trend"].notna()].sort_values("Year")
            ax.plot(trend["Year"], trend["trend"], color="tab:red")
        ax.set_xlabel("Year")
        ax.set_ylabel(column)
        plt.close(result["figure"])
    fig.tight_layout()
    return _plot_result(df, fig, axes_flat)


def plot_scenarios(
    scenario_data: dict[str, Any] | list[dict[str, Any]],
    metric: str | None = None,
    add_historic: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Plot historic and future climate scenarios as matplotlib boxplots.

    Translates R ``plot_scenarios``.
    """
    scenario_list = scenario_data if isinstance(scenario_data, list) else [scenario_data]
    if not scenario_list:
        raise ValueError("scenario_data must contain at least one scenario")
    if metric is None:
        raise ValueError("metric must be specified")

    rows: list[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(scenario_list):
        if "data" not in scenario or "caption" not in scenario:
            raise ValueError("each scenario must contain 'data' and 'caption'")
        caption_values = list(np.atleast_1d(scenario["caption"]))
        caption = " ".join(str(c) for c in caption_values)
        data_items = _as_frame_list(scenario["data"], name="scenario data")
        labels = scenario.get("labels")
        if labels is None or (not isinstance(labels, Sequence) or isinstance(labels, str)):
            labels = [name for name, _ in data_items]
        for index, (name, frame) in enumerate(data_items):
            if metric not in frame.columns:
                raise ValueError(f"metric '{metric}' is not included in scenario data")
            part = pd.DataFrame(
                {
                    "Scenario": caption,
                    "Source": str(labels[index]) if index < len(labels) else name,
                    "Metric": pd.to_numeric(frame[metric], errors="coerce"),
                    "ScenarioIndex": scenario_index,
                }
            )
            rows.append(part.dropna(subset=["Metric"]))
        if add_historic and scenario.get("historic_data") is not None:
            historic = pd.DataFrame(scenario["historic_data"])
            if metric in historic.columns:
                rows.append(
                    pd.DataFrame(
                        {
                            "Scenario": "Observed",
                            "Source": historic.get("End_year", np.arange(len(historic))).astype(str),
                            "Metric": pd.to_numeric(historic[metric], errors="coerce"),
                            "ScenarioIndex": -1,
                        }
                    ).dropna(subset=["Metric"])
                )
    plot_data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Scenario", "Source", "Metric"])
    if plot_data.empty:
        raise ValueError("no numeric scenario data available for plotting")

    groups = list(plot_data.groupby(["Scenario", "Source"], sort=False))
    labels = [f"{scenario}\n{source}" for (scenario, source), _ in groups]
    values = [group["Metric"].to_numpy(dtype=float) for _, group in groups]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (max(7, len(values) * 1.2), 4)))
    ax.boxplot(values, tick_labels=labels, patch_artist=True)
    ax.set_ylabel(kwargs.get("y_axis_name", f"Cumulative response in {metric}"))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _plot_result(plot_data, fig, ax)


def plot_climate_scenarios(
    climate_scenario_list: list[dict[str, Any]],
    metric: str | None = None,
    metric_label: str | None = None,
    year_name: str = "End_year",
    reference_line: Sequence[float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Plot climate scenario panels with matplotlib boxplots.

    Translates R ``plot_climate_scenarios``.
    """
    if metric is None:
        raise ValueError("metric must be specified")
    if not climate_scenario_list:
        raise ValueError("climate_scenario_list must not be empty")

    panel_data: list[dict[str, Any]] = []
    legends: list[Any] = []
    for scenario_index, scenario in enumerate(climate_scenario_list):
        data_items = _as_frame_list(scenario.get("data"), name="scenario data")
        labels = scenario.get("labels")
        if labels is None or (not isinstance(labels, Sequence) or isinstance(labels, str)):
            labels = [name for name, _ in data_items]
        caption = scenario.get("caption", [f"Scenario {scenario_index + 1}"])
        caption_text = "\n".join(str(c) for c in np.atleast_1d(caption))
        for index, (name, frame) in enumerate(data_items):
            if metric not in frame.columns:
                raise ValueError(f"scenario {index} in panel {scenario_index} lacks column {metric}")
            panel_data.append(
                {
                    "panel": caption_text,
                    "label": labels[index] if index < len(labels) else name,
                    "values": pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(dtype=float),
                    "time_series": bool(scenario.get("time_series", False)),
                }
            )
        legends.append(
            pd.DataFrame({"code": range(1, len(labels) + 1), "Label": labels})
            if labels is not None
            else "no adequate labels provided"
        )

    panels = list(dict.fromkeys(item["panel"] for item in panel_data))
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(panels), figsize=kwargs.get("figsize", (max(5, 4 * len(panels)), 4)), squeeze=False)
    axes_flat = axes.ravel()
    for ax, panel in zip(axes_flat, panels):
        items = [item for item in panel_data if item["panel"] == panel]
        values = [item["values"] for item in items]
        labels = [str(item["label"]) for item in items]
        ax.boxplot(values, tick_labels=labels, patch_artist=True)
        if reference_line is not None:
            ref = sorted([float(v) for v in reference_line])
            if len(ref) in {2, 3}:
                ax.axhspan(ref[0], ref[-1], color="tab:blue", alpha=0.15)
            if len(ref) in {1, 3}:
                ax.axhline(ref[len(ref) // 2], color="tab:blue", linewidth=1.5)
        ax.set_title(panel)
        ax.set_ylabel(metric_label or metric)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    data = pd.DataFrame(
        [
            {"panel": item["panel"], "label": item["label"], "value": value}
            for item in panel_data
            for value in item["values"]
        ]
    )
    return _plot_result(data, fig, axes_flat, legend=legends)


def plot_pls(
    pls_results: dict[str, Any],
    file_path: str | Path | None = None,
    vip_threshold: float = 0.8,
    **kwargs: Any,
) -> dict[str, Any]:
    """Plot PLS coefficient and VIP summaries.

    Translates R ``plot_pls``.
    """
    if "object_type" in pls_results:
        prepared = prepare_pls_plot_data(pls_results, vip_threshold=vip_threshold, **kwargs)
        summary = prepared["panels"]
    elif "PLS_summary" in pls_results:
        summary = pd.DataFrame(pls_results["PLS_summary"]).copy()
        _require_columns(summary, ["VIP", "Coef"], name="PLS_summary")
    else:
        raise ValueError("not a PLS result object")

    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=kwargs.get("figsize", (9, 6)), sharex=True)
    x = np.arange(len(summary))
    axes[0].bar(x, pd.to_numeric(summary["VIP"], errors="coerce"), color="0.35")
    axes[0].axhline(vip_threshold, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("VIP")
    coef = pd.to_numeric(summary["Coef"], errors="coerce")
    axes[1].bar(x, coef, color=np.where(coef < 0, "tab:blue", "tab:green"))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Coefficient")
    axes[1].set_xlabel("Time step")
    fig.tight_layout()
    if file_path is not None:
        fig.savefig(file_path)
    return _plot_result(summary, fig, axes)


def _phase_days(phase: Sequence[int]) -> list[int]:
    if len(phase) != 2:
        raise ValueError("phase definitions must contain start and end Julian days")
    start, end = int(phase[0]), int(phase[1])
    if start > end:
        return list(range(start, 367)) + list(range(1, end + 1))
    return list(range(start, end + 1))


def _phase_temperature_data(
    pheno_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    *,
    split_month: int,
    chilling_phase: Sequence[int],
    forcing_phase: Sequence[int],
) -> pd.DataFrame:
    pheno = pheno_data.copy()
    weather = weather_data.copy()
    _require_columns(pheno, ["Year", "pheno"], name="pheno_data")
    _require_columns(weather, ["Year", "Month", "Day", "Tmin", "Tmax"], name="weather_data")
    for column in ["Year", "Month", "Day", "Tmin", "Tmax"]:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    pheno["Year"] = pd.to_numeric(pheno["Year"], errors="coerce")
    pheno["pheno"] = pd.to_numeric(pheno["pheno"], errors="coerce")
    if not 1 <= int(split_month) <= 12:
        raise ValueError("split_month must be between 1 and 12")

    dates = pd.to_datetime(
        {"year": weather["Year"], "month": weather["Month"], "day": weather["Day"]},
        errors="coerce",
    )
    weather = weather.loc[dates.notna()].copy()
    weather["JDay"] = dates.loc[dates.notna()].dt.dayofyear.to_numpy()
    weather["Season"] = np.where(weather["Month"] <= split_month, weather["Year"], weather["Year"] + 1)
    weather["Tmean"] = (weather["Tmin"] + weather["Tmax"]) / 2

    chill_days = set(_phase_days(chilling_phase))
    force_days = set(_phase_days(forcing_phase))
    chill = (
        weather.loc[weather["JDay"].isin(chill_days)]
        .groupby("Season", as_index=False)["Tmean"]
        .mean()
        .rename(columns={"Tmean": "Tmean_chilling_period"})
    )
    force = (
        weather.loc[weather["JDay"].isin(force_days)]
        .groupby("Season", as_index=False)["Tmean"]
        .mean()
        .rename(columns={"Tmean": "Tmean_forcing_period"})
    )
    merged = chill.merge(force, on="Season", how="inner").merge(
        pheno.rename(columns={"Year": "Season"}),
        on="Season",
        how="inner",
    )
    return merged.dropna(subset=["Tmean_chilling_period", "Tmean_forcing_period", "pheno"])


def plot_phenology_trends(
    pheno_data: pd.DataFrame,
    weather_data: pd.DataFrame | None = None,
    split_month: int = 6,
    chilling_phase: Sequence[int] | None = None,
    forcing_phase: Sequence[int] | None = None,
    x_axis_name: str | None = None,
    y_axis_name: str | None = None,
    legend_name: str | None = None,
    contour_line_color: str | None = "black",
    point_color: str | None = "black",
    point_shape: Any = "o",
    legend_colors: Sequence[str] | None = None,
    base_size: float = 11,
    **kwargs: Any,
) -> dict[str, Any]:
    """Plot phenology trends or chilling/forcing response surfaces.

    Translates R ``plot_phenology_trends``.
    """
    if weather_data is None:
        return make_pheno_trend_plot(pd.DataFrame(), pheno_data, **kwargs)
    if chilling_phase is None or forcing_phase is None:
        raise ValueError("chilling_phase and forcing_phase are required when weather_data is provided")

    df = _phase_temperature_data(
        pheno_data,
        weather_data,
        split_month=split_month,
        chilling_phase=chilling_phase,
        forcing_phase=forcing_phase,
    )
    if df.empty:
        raise ValueError("no overlapping phenology/weather records available for plotting")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 5)))
    x = df["Tmean_chilling_period"].to_numpy(dtype=float)
    y = df["Tmean_forcing_period"].to_numpy(dtype=float)
    z = df["pheno"].to_numpy(dtype=float)
    cmap = None
    if legend_colors is not None:
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("phenology_trend", list(legend_colors))

    if len(df) >= 3 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
        try:
            fill = ax.tricontourf(x, y, z, levels=12, cmap=cmap)
            if contour_line_color is not None:
                ax.tricontour(x, y, z, levels=8, colors=contour_line_color, linewidths=max(base_size / 25, 0.4))
        except RuntimeError:
            fill = ax.scatter(x, y, c=z, cmap=cmap)
    else:
        fill = ax.scatter(x, y, c=z, cmap=cmap)
    if point_color is not None:
        ax.scatter(x, y, color=point_color, marker=point_shape, edgecolors="black", linewidths=0.4)
    colorbar = fig.colorbar(fill, ax=ax)
    colorbar.set_label(legend_name or "Bloom date (DOY)")
    ax.set_xlabel(x_axis_name or "Mean temperature during the chilling phase (deg C)")
    ax.set_ylabel(y_axis_name or "Mean temperature during the forcing phase (deg C)")
    fig.tight_layout()
    return _plot_result(df, fig, ax, colorbar=colorbar)


plot_PLS = plot_pls
