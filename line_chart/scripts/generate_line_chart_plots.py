"""
UPDATED: Line chart blog post – plot generator (4-table version)
---------------------------------------------------------------
Inputs (4 CSVs):
1) daily_platform_metrics_clean.csv   -> main story (no outage visual noise)
2) kpi_rollups_monthly.csv            -> discrete KPI monthly rollups
3) events_and_forecast.csv            -> annotations + forecast band (no outage event)
4) dip_story_timeseries.csv           -> separate dip/missing-data lesson only

Outputs:
A folder ./plots_line_chart_post/ containing PNGs for all figures in the post.

No seaborn. Pure matplotlib.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import warnings

# Set font to Inter for all plots with increased font sizes
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['font.size'] = 15  # Base font size (increased from 11)
    plt.rcParams['axes.titlesize'] = 18  # Title font size (increased from 13)
    plt.rcParams['axes.labelsize'] = 15  # Axis label font size (increased from 11)
    plt.rcParams['xtick.labelsize'] = 14  # X-axis tick font size (increased from 10)
    plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick font size (increased from 10)
    plt.rcParams['legend.fontsize'] = 14  # Legend font size (increased from 10)

# -----------------------------
# Theme Colors (data story theme with beige background)
# -----------------------------
# Dumbbell plot theme colors
COLOR_PRIMARY = "#292F36"  # Dark Gray / Charcoal (from dumbbell theme)
COLOR_SECONDARY = "#A41F13"  # Dark Red / Reddish-Brown (from dumbbell theme - was orange)
COLOR_ACCENT = "#FAC846"  # Bright Sun (Yellow)
COLOR_GREEN = "#A0C382"  # Olivine (Green)
COLOR_TEAL = "#5F9B8C"  # Patina (Teal)
COLOR_HIGHLIGHT = "#A41F13"  # Dark Red from dumbbell theme - for highlights
COLOR_LINE = "#E0DBD8"  # Light Gray / Beige - for connecting lines and axes
COLOR_BG = "#FAF5F1"  # Off-White / Cream - for backgrounds (beige)
COLOR_TEXT = "#292F36"  # Dark Gray / Charcoal - for text (from dumbbell theme)
COLOR_GREY = "#CCCCCC"  # Grey for de-emphasized lines

# Legacy names for compatibility
COLOR_2024 = COLOR_PRIMARY
COLOR_2025 = COLOR_SECONDARY
COLOR_TARGET = "#8F7A6E"  # Medium Brown / Taupe (from dumbbell theme)

# -----------------------------
# CONFIG
# -----------------------------
SCRIPT_DIR = Path(__file__).parent  # scripts folder
DATA_DIR = SCRIPT_DIR.parent / "data"  # data folder
OUTDIR = SCRIPT_DIR.parent / "images"  # images folder
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_DAILY_CLEAN = DATA_DIR / "daily_platform_metrics_clean.csv"
DATA_MONTHLY = DATA_DIR / "kpi_rollups_monthly.csv"
DATA_EVENTS = DATA_DIR / "events_and_forecast.csv"
DATA_DIP = DATA_DIR / "dip_story_timeseries.csv"
DATA_LOG = DATA_DIR / "log_scale_story_timeseries.csv"
DATA_SCALE_DEMO = DATA_DIR / "scale_demo_timeseries.csv"
DATA_FORECAST = DATA_DIR / "forecast_plot_table.csv"

HIGHLIGHT_INDUSTRY = "Finance"


# -----------------------------
# HELPERS
# -----------------------------
def savefig(name: str):
    path = OUTDIR / name
    plt.tight_layout()
    # Ensure background color is preserved when saving
    fig = plt.gcf()
    if fig.get_facecolor()[0] == 1.0:  # If default white, set to COLOR_BG
        fig.patch.set_facecolor(COLOR_BG)
    plt.savefig(str(path), dpi=200, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"Saved: {path}")


def style_plot(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent styling to a plot"""
    ax.set_title(title, color=COLOR_TEXT, fontsize=18)
    ax.set_xlabel(xlabel, color=COLOR_TEXT, fontsize=15)
    ax.set_ylabel(ylabel, color=COLOR_TEXT, fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    ax.tick_params(colors=COLOR_TEXT, labelsize=11)


def read_inputs():
    daily = pd.read_csv(str(DATA_DAILY_CLEAN))
    monthly = pd.read_csv(str(DATA_MONTHLY))
    events = pd.read_csv(str(DATA_EVENTS))
    dip = pd.read_csv(str(DATA_DIP))

    daily["date"] = pd.to_datetime(daily["date"])
    events["date"] = pd.to_datetime(events["date"])
    dip["date"] = pd.to_datetime(dip["date"])
    
    # Change dates from 2026 to 2025
    daily["date"] = daily["date"] - pd.DateOffset(years=1)
    events["date"] = events["date"] - pd.DateOffset(years=1)
    dip["date"] = dip["date"] - pd.DateOffset(years=1)
    
    # Also update monthly dates (format: "2026-01" -> "2025-01")
    monthly["month"] = monthly["month"].str.replace("2026", "2025")

    # numeric coercions
    daily_num_cols = [
        "api_requests", "p95_latency_ms", "new_enterprise_customers_daily",
        "revenue_usd", "marketing_spend_usd", "gross_margin_pct"
    ]
    for c in daily_num_cols:
        if c in daily.columns:
            daily[c] = pd.to_numeric(daily[c], errors="coerce")

    for c in ["forecast_requests_mean", "forecast_requests_p10", "forecast_requests_p90"]:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors="coerce")

    dip["api_requests_total_observed"] = pd.to_numeric(dip["api_requests_total_observed"], errors="coerce")

    return daily, monthly, events, dip


def series_total(daily: pd.DataFrame, col: str) -> pd.Series:
    """Sum across industries per day (ignores NaNs)."""
    s = daily.groupby("date")[col].sum(min_count=1).sort_index()
    return s


def make_ordinal_spacing(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.sort_values(date_col).copy()
    out["ordinal_x"] = np.arange(len(out))
    return out


def annotate_events(ax, events_df: pd.DataFrame, y_at: pd.Series | None = None):
    # Only annotate INCIDENT events (latency incident)
    sub = events_df[events_df["label"].fillna("").astype(str).str.len() > 0].copy()
    sub = sub[sub["event_type"] == "INCIDENT"]

    for _, r in sub.iterrows():
        x = r["date"]
        label = r["label"]
        if y_at is not None and x in y_at.index:
            y = float(y_at.loc[x])
        else:
            y = ax.get_ylim()[0] + 0.65 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        # Split label into two rows if it contains ":"
        if ":" in label:
            parts = label.split(":", 1)
            label_formatted = f"{parts[0]}:\n{parts[1].strip()}"
        else:
            label_formatted = label

        ax.annotate(
            label_formatted,
            xy=(x, y),
            xytext=(-10, -30),  # Move to the left and lower (negative y offset)
            textcoords="offset points",
            fontsize=14,
            arrowprops=dict(arrowstyle="->", linewidth=1, alpha=0.6, color=COLOR_2024),
            alpha=0.95,
            color=COLOR_2024,
            ha="right",  # Right-align text when positioned to the left
        )


# -----------------------------
# FIGURE SET 1: Discrete KPI continuity lie
# -----------------------------
def plot_discrete_kpi_continuity(monthly: pd.DataFrame):
    df = monthly.copy()
    # monthly["month"] is already processed in read_inputs, but we need to convert back to datetime
    df["month"] = pd.to_datetime(df["month"] + "-01")

    # BAD: line chart implies smooth continuity between month buckets
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(df["month"], df["new_enterprise_customers_monthly"], marker="o", color=COLOR_PRIMARY, linewidth=2)
    style_plot(ax, "BAD: Monthly totals plotted as a continuous trend", "Month", "New enterprise customers (monthly total)")
    savefig("01_bad_monthly_totals_line.png")

    # GOOD: bar chart enforces discreteness
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    # Calculate bar width so gap is exactly half the bar width
    # For monthly data, calculate spacing between months
    if len(df) > 1:
        spacing_days = (df["month"].iloc[1] - df["month"].iloc[0]).days
        # If gap = width/2, then spacing = width + gap = width + width/2 = 1.5*width
        # So width = spacing / 1.5
        bar_width_days = spacing_days / 1.5
    else:
        bar_width_days = 20  # fallback
    ax.bar(df["month"], df["new_enterprise_customers_monthly"], width=bar_width_days, color=COLOR_SECONDARY, edgecolor=COLOR_SECONDARY)
    style_plot(ax, "GOOD: Monthly totals shown as discrete buckets", "Month", "New enterprise customers (monthly total)")
    savefig("02_good_monthly_totals_bars.png")

    # GOOD alternative: points-only
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.scatter(df["month"], df["new_enterprise_customers_monthly"], color=COLOR_SECONDARY, s=50, edgecolors=COLOR_SECONDARY)
    style_plot(ax, "GOOD: Points-only (no implied path between months)", "Month", "New enterprise customers (monthly total)")
    savefig("03_good_monthly_totals_points.png")


# -----------------------------
# FIGURE SET 2: Missing-data dip story ONLY (bridged vs zeroed vs gap)
# -----------------------------
def plot_dip_story(dip: pd.DataFrame):
    s = dip.set_index("date")["api_requests_total_observed"].sort_index()

    # BAD A: treat missing as zero (common in Excel/Tableau defaults after fill)
    zeroed = s.copy().fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(zeroed.index, zeroed.values, color=COLOR_PRIMARY, linewidth=2)
    style_plot(ax, "BAD: Missing treated as zero (fake catastrophic dip)", "Date", "Total API requests (daily)")
    savefig("04_bad_missing_as_zero.png")

    # BAD B: bridge gaps via interpolation (false stability)
    bridged = s.copy().interpolate(method="time")

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(bridged.index, bridged.values, color=COLOR_PRIMARY, linewidth=2)
    style_plot(ax, "BAD: Missing silently bridged (interpolated)", "Date", "Total API requests (daily)")
    savefig("05_bad_missing_bridged.png")

    # GOOD: show visible gap + dashed connection + "No data" label
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # Split data into segments before and after missing period
    missing_mask = s.isna()
    if missing_mask.any():
        # Find consecutive missing periods
        missing_groups = (missing_mask != missing_mask.shift()).cumsum()
        missing_periods = []
        
        for group_id in missing_groups[missing_mask].unique():
            period_indices = s[missing_groups == group_id].index
            if len(period_indices) > 0:
                missing_periods.append((period_indices[0], period_indices[-1]))
        
        # Use the first missing period
        if missing_periods:
            missing_start_idx = missing_periods[0][0]
            missing_end_idx = missing_periods[0][1]
            
            # Data before missing period
            before_missing = s[s.index < missing_start_idx]
            # Data after missing period
            after_missing = s[s.index > missing_end_idx]
            
            # Plot solid lines for segments with data
            if len(before_missing) > 0:
                ax.plot(before_missing.index, before_missing.values, marker="o", markersize=3, 
                       linewidth=2, color=COLOR_PRIMARY, markerfacecolor=COLOR_PRIMARY, zorder=3)
            
            if len(after_missing) > 0:
                ax.plot(after_missing.index, after_missing.values, marker="o", markersize=3, 
                       linewidth=2, color=COLOR_PRIMARY, markerfacecolor=COLOR_PRIMARY, zorder=3)
            
            # Connect with dashed line across the gap
            if len(before_missing) > 0 and len(after_missing) > 0:
                last_before_val = before_missing.iloc[-1]
                first_after_val = after_missing.iloc[0]
                last_before_date = before_missing.index[-1]
                first_after_date = after_missing.index[0]
                
                ax.plot([last_before_date, first_after_date], [last_before_val, first_after_val],
                       linestyle="--", linewidth=2, color=COLOR_PRIMARY, alpha=0.6, zorder=2)
            
            # Add "No data" label without bounding box (centered in the gap)
            mid_missing_date = missing_start_idx + (missing_end_idx - missing_start_idx) / 2
            # Set y position higher up in the y-axis range (moved up from 0.5 to 0.7)
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_mid = ax.get_ylim()[0] + 0.7 * y_range
            ax.text(mid_missing_date, y_mid, "No data", ha="center", va="center",
                   fontsize=15, color=COLOR_TEXT, zorder=4)
    else:
        # No missing data, plot normally
        ax.plot(s.index, s.values, marker="o", markersize=3, linewidth=2, 
               color=COLOR_PRIMARY, markerfacecolor=COLOR_PRIMARY)
    
    style_plot(ax, "GOOD: Missing shown as a visible gap (truthful)", "Date", "Total API requests (daily)")
    savefig("06_good_missing_gap.png")


# -----------------------------
# FIGURE SET 3: Irregular-interval spacing trap (ordinal vs datetime)
# -----------------------------
def plot_irregular_spacing_trap(events: pd.DataFrame):
    # Use narrative events as sparse irregular points
    sparse = events[events["event_type"].isin(["LAUNCH", "CAMPAIGN", "PRICING", "INCIDENT"])].copy()
    sparse = sparse.sort_values("date")
    sparse["metric"] = np.arange(len(sparse)) + 1  # just to visualize spacing

    # BAD: ordinal spacing (equal gaps)
    bad = make_ordinal_spacing(sparse, "date")
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(bad["ordinal_x"], bad["metric"], marker="o", color=COLOR_PRIMARY, linewidth=2, markerfacecolor=COLOR_PRIMARY)
    ax.set_title("BAD: Irregular events plotted at equal spacing (ordinal x)", color=COLOR_TEXT, fontsize=18)
    ax.set_xlabel("Event order (not time)", color=COLOR_TEXT, fontsize=15)
    ax.set_ylabel("Metric", color=COLOR_TEXT, fontsize=15)
    ax.set_xticks(bad["ordinal_x"])
    ax.set_xticklabels(bad["date"].dt.strftime("%b %d"), rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    savefig("07_bad_irregular_equal_spacing.png")

    # GOOD: true datetime axis
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(sparse["date"], sparse["metric"], marker="o", color=COLOR_SECONDARY, linewidth=2, markerfacecolor=COLOR_SECONDARY)
    style_plot(ax, "GOOD: Irregular events plotted on a true datetime axis", "Date", "Metric")
    savefig("08_good_irregular_datetime_axis.png")


# -----------------------------
# FIGURE SET 4: Spaghetti chart (bad) vs remedies (main clean data)
# -----------------------------
def plot_spaghetti_and_fixes(daily: pd.DataFrame):
    pivot = daily.pivot_table(index="date", columns="industry", values="api_requests", aggfunc="sum").sort_index()

    # BAD spaghetti
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    # Use different colors for each line
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TARGET, COLOR_TEAL, COLOR_GREEN, COLOR_ACCENT]
    for i, col in enumerate(pivot.columns):
        color = colors[i % len(colors)]  # Cycle through colors
        ax.plot(pivot.index, pivot[col].values, alpha=0.9, color=color, linewidth=1.5)
    style_plot(ax, "BAD: Spaghetti chart (too many lines)", "Date", "API requests (daily)")
    savefig("09_bad_spaghetti.png")

    # GOOD: focus + context
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    for col in pivot.columns:
        if col == HIGHLIGHT_INDUSTRY:
            ax.plot(pivot.index, pivot[col].values, linewidth=2.5, color=COLOR_HIGHLIGHT)
        else:
            ax.plot(pivot.index, pivot[col].values, alpha=0.7, linewidth=1.5, color=COLOR_GREY)
    
    # Add annotation for Finance line at the end
    if HIGHLIGHT_INDUSTRY in pivot.columns:
        finance_data = pivot[HIGHLIGHT_INDUSTRY]
        # Use the last point of the data
        annotate_date = finance_data.index[-1]
        annotate_value = finance_data.iloc[-1]
        
        ax.annotate(
            HIGHLIGHT_INDUSTRY,
            xy=(annotate_date, annotate_value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=15,
            color=COLOR_HIGHLIGHT,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color=COLOR_HIGHLIGHT, alpha=0.8)
        )
    
    ax.set_title(f"GOOD: Focus + context (highlight {HIGHLIGHT_INDUSTRY})", color=COLOR_TEXT, fontsize=18)
    ax.set_xlabel("Date", color=COLOR_TEXT, fontsize=15)
    ax.set_ylabel("API requests (daily)", color=COLOR_TEXT, fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    savefig("10_good_focus_context.png")

    # GOOD: small multiples
    cols = list(pivot.columns)
    n = len(cols)
    ncols = 2
    nrows = math.ceil(n / ncols)

    # Calculate global y-axis range across all columns
    y_min = pivot[cols].min().min()
    y_max = pivot[cols].max().max()
    # Add small padding
    y_range = y_max - y_min
    y_padding = y_range * 0.05
    y_lim = (y_min - y_padding, y_max + y_padding)

    fig = plt.figure(figsize=(12, 3 * nrows), facecolor=COLOR_BG)
    for i, col in enumerate(cols, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        ax.set_facecolor(COLOR_BG)
        
        # Plot all lines: highlight the line matching this panel's title, de-emphasize others
        for plot_col in cols:
            if plot_col == col:  # Highlight the line that matches this panel's title
                ax.plot(pivot.index, pivot[plot_col].values, linewidth=1, color=COLOR_HIGHLIGHT, zorder=3)
            else:
                # De-emphasize other lines with more transparency and thinner lines
                context_color = "#CCCCCC"  # Lighter grey
                ax.plot(pivot.index, pivot[plot_col].values, alpha=0.3, linewidth=0.7, color=context_color, zorder=2)
        
        ax.set_ylim(y_lim)  # Set same y-axis scale for all panels
        ax.set_title(col, color=COLOR_TEXT, fontsize=18)
        ax.set_xlabel("Date", color=COLOR_TEXT, fontsize=10)
        ax.set_ylabel("API requests", color=COLOR_TEXT, fontsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_LINE)
        ax.spines["bottom"].set_color(COLOR_LINE)
        ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    fig.suptitle("GOOD: Small multiples (each series gets its own space)", y=1.02, fontsize=20, color=COLOR_TEXT)
    savefig("11_good_small_multiples.png")


# -----------------------------
# FIGURE SET 5: Dual-axis trap (bad) vs stacked + indexed (main clean data)
# -----------------------------
def plot_dual_axis_trap(daily: pd.DataFrame):
    spend = series_total(daily, "marketing_spend_usd")
    margin = daily.groupby("date")["gross_margin_pct"].mean().sort_index()

    # BAD: dual axis
    fig, ax1 = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax1.set_facecolor(COLOR_BG)
    ax1.plot(spend.index, spend.values, color=COLOR_PRIMARY, linewidth=2)
    ax1.set_xlabel("Date", color=COLOR_TEXT, fontsize=15)
    ax1.set_ylabel("Marketing spend (USD)", color=COLOR_TEXT, fontsize=15)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_color(COLOR_LINE)
    ax1.spines["bottom"].set_color(COLOR_LINE)
    ax1.tick_params(colors=COLOR_TEXT, labelsize=11)
    ax2 = ax1.twinx()
    ax2.plot(margin.index, margin.values, color=COLOR_SECONDARY, linewidth=2)
    ax2.set_ylabel("Gross margin (%)", color=COLOR_TEXT)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLOR_LINE)
    ax2.tick_params(colors=COLOR_TEXT, labelsize=11)
    fig.suptitle("BAD: Dual y-axes suggest correlation by geometry", color=COLOR_TEXT, fontsize=18)
    savefig("12_bad_dual_axis.png")

    # GOOD: stacked panels
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor=COLOR_BG)
    ax_top.set_facecolor(COLOR_BG)
    ax_bot.set_facecolor(COLOR_BG)
    ax_top.plot(spend.index, spend.values, color=COLOR_SECONDARY, linewidth=2)
    ax_top.set_ylabel("Marketing spend (USD)", color=COLOR_TEXT, fontsize=15)
    ax_top.set_title("GOOD: Separate panels, shared x-axis", color=COLOR_TEXT, fontsize=18)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_color(COLOR_LINE)
    ax_top.spines["bottom"].set_color(COLOR_LINE)
    ax_top.tick_params(colors=COLOR_TEXT, labelsize=11)
    ax_bot.plot(margin.index, margin.values, color=COLOR_SECONDARY, linewidth=2)
    ax_bot.set_ylabel("Gross margin (%)", color=COLOR_TEXT)
    ax_bot.set_xlabel("Date", color=COLOR_TEXT)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.spines["left"].set_color(COLOR_LINE)
    ax_bot.spines["bottom"].set_color(COLOR_LINE)
    ax_bot.tick_params(colors=COLOR_TEXT, labelsize=11)
    savefig("13_good_stacked_panels.png")

    # GOOD: indexed-to-100
    idx_spend = 100 * (spend / spend.iloc[0])
    idx_margin = 100 * (margin / margin.iloc[0])

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(idx_spend.index, idx_spend.values, label="Marketing spend (index=100)", color=COLOR_PRIMARY, linewidth=2)
    ax.plot(idx_margin.index, idx_margin.values, label="Gross margin (index=100)", color=COLOR_SECONDARY, linewidth=2)
    ax.set_title("GOOD: Index both series (single axis, relative change)", color=COLOR_TEXT, fontsize=18)
    ax.set_xlabel("Date", color=COLOR_TEXT, fontsize=15)
    ax.set_ylabel("Index (start=100)", color=COLOR_TEXT, fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    ax.legend(frameon=False)
    savefig("14_good_indexed_single_axis.png")


# -----------------------------
# FIGURE SET 6: Smoothing trap (main clean data)
# -----------------------------
def plot_smoothing_trap(daily: pd.DataFrame):
    margin = daily.groupby("date")["gross_margin_pct"].mean().sort_index()

    # BAD: heavy rolling average (cosmetic calm)
    bad = margin.rolling(21, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(bad.index, bad.values, color=COLOR_PRIMARY, linewidth=2)
    style_plot(ax, "BAD: Heavy smoothing (21-day rolling average) hides real swings", "Date", "Gross margin (%)")
    savefig("15_bad_heavy_smoothing.png")

    # GOOD: raw + clearly-labeled trend overlay
    trend = margin.rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(margin.index, margin.values, marker="o", markersize=2, linewidth=1, alpha=0.9, label="Raw", color=COLOR_PRIMARY)
    ax.plot(trend.index, trend.values, linewidth=2, label="7-day rolling mean", color=COLOR_SECONDARY)
    style_plot(ax, "GOOD: Raw series + explicit trend overlay", "Date", "Gross margin (%)")
    ax.legend(frameon=False)
    savefig("16_good_raw_plus_trend.png")


# -----------------------------
# FIGURE SET 7: Linear vs log scale + indexed (main clean data)
# -----------------------------
def fmt_millions(x, pos):
    """Format numbers as millions (1.5M) or thousands (500K)"""
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{x:.0f}"


def add_event_markers(ax, ev, y_series):
    """Add event markers at the closest weekly point with annotations"""
    for _, r in ev.iterrows():
        d = r["date"]
        # Find nearest index in y_series
        idx = y_series.index.get_indexer([d], method="nearest")[0]
        xd = y_series.index[idx]
        yd = float(y_series.iloc[idx])

        ax.scatter([xd], [yd], s=35, color=COLOR_SECONDARY, zorder=5, edgecolors=COLOR_PRIMARY, linewidths=1)
        ax.annotate(
            r["event_type"],
            xy=(xd, yd),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=14,
            alpha=0.9,
            color=COLOR_TEXT
        )


def plot_linear_vs_log(daily: pd.DataFrame = None):
    # Use scale demo timeseries data (monthly)
    df = pd.read_csv(str(DATA_SCALE_DEMO))
    df["month"] = pd.to_datetime(df["month"])
    
    # Change dates from 2024-2025 to 2025-2026
    df["month"] = df["month"] - pd.DateOffset(years=1)
    
    # Calculate indexed series
    df["indexed_100"] = 100 * df["api_requests_total"] / df["api_requests_total"].iloc[0]

    # Linear
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(df["month"], df["api_requests_total"], marker="o", linewidth=2, 
            color=COLOR_PRIMARY, markerfacecolor=COLOR_PRIMARY, markeredgecolor=COLOR_PRIMARY)
    style_plot(ax, "Linear scale: growth becomes a hockey stick", "Month", "Total API requests")
    savefig("17_linear_scale_hockey_stick.png")

    # Log
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(df["month"], df["api_requests_total"], marker="o", linewidth=2,
            color=COLOR_PRIMARY, markerfacecolor=COLOR_PRIMARY, markeredgecolor=COLOR_PRIMARY)
    ax.set_yscale("log")
    style_plot(ax, "Log scale: reveals relative growth and early volatility", "Month", "Total API requests (log scale)")
    savefig("18_log_scale.png")

    # Indexed
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(df["month"], df["indexed_100"], marker="o", linewidth=2,
            color=COLOR_PRIMARY, markerfacecolor=COLOR_PRIMARY, markeredgecolor=COLOR_PRIMARY)
    style_plot(ax, "Indexed (start=100): audience-friendly relative growth view", "Month", "Index (start=100)")
    savefig("19_indexed_growth.png")


# -----------------------------
# FIGURE SET 8: Honest template (main clean data + forecast + annotations)
# -----------------------------
def plot_honest_template(daily: pd.DataFrame, events: pd.DataFrame):
    s = series_total(daily, "api_requests")

    # Load forecast data from CSV (matches reference implementation)
    forecast_start_date = pd.Timestamp("2025-04-01")  # Adjusted to 2025 to match script's date adjustment
    
    try:
        forecast_df = pd.read_csv(str(DATA_FORECAST))
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        # Adjust dates to 2025 to match the script's date adjustment (subtract 1 year)
        forecast_df["date"] = forecast_df["date"] - pd.DateOffset(years=1)
        
        # Create combined plot table (like reference code)
        plot_df = pd.DataFrame({
            "date": pd.date_range(s.index.min(), forecast_df["date"].max(), freq="D")
        })
        
        # Add observed data
        s_df = s.reset_index().rename(columns={"api_requests": "observed_total_requests"})
        plot_df = plot_df.merge(s_df, on="date", how="left")
        
        # Add forecast data
        plot_df = plot_df.merge(forecast_df, on="date", how="left")
        
        # Ensure numeric types
        for c in ["observed_total_requests", "forecast_mean", "forecast_p10", "forecast_p90"]:
            if c in plot_df.columns:
                plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    except FileNotFoundError:
        # Fallback: use old method
        plot_df = None

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    if plot_df is not None and len(plot_df) > 0:
        # Observed (points + thin line) - with gap for missing data period
        obs = plot_df.loc[plot_df["date"] < forecast_start_date].copy()
        obs = obs.dropna(subset=["observed_total_requests"])
        
        # Define missing data period: 2025-02-01 to 2025-02-05
        gap_start = pd.Timestamp("2025-02-01")
        gap_end = pd.Timestamp("2025-02-05")
        
        if len(obs) > 0:
            # Split data into segments before and after gap
            before_gap = obs[obs["date"] < gap_start].copy()
            after_gap = obs[obs["date"] > gap_end].copy()
            
            # Plot solid lines for segments with data
            if len(before_gap) > 0:
                ax.plot(
                    before_gap["date"],
                    before_gap["observed_total_requests"],
                    marker="o",
                    markersize=2,
                    linewidth=1,
                    alpha=0.95,
                    color=COLOR_SECONDARY,
                    markerfacecolor=COLOR_SECONDARY,
                    zorder=3
                )
            
            if len(after_gap) > 0:
                ax.plot(
                    after_gap["date"],
                    after_gap["observed_total_requests"],
                    marker="o",
                    markersize=2,
                    linewidth=1,
                    alpha=0.95,
                    color=COLOR_SECONDARY,
                    markerfacecolor=COLOR_SECONDARY,
                    zorder=3
                )
            
            # Connect with dashed line across the gap
            if len(before_gap) > 0 and len(after_gap) > 0:
                last_before_val = before_gap["observed_total_requests"].iloc[-1]
                first_after_val = after_gap["observed_total_requests"].iloc[0]
                last_before_date = before_gap["date"].iloc[-1]
                first_after_date = after_gap["date"].iloc[0]
                
                ax.plot(
                    [last_before_date, first_after_date],
                    [last_before_val, first_after_val],
                    linestyle="--",
                    linewidth=2,
                    color=COLOR_SECONDARY,
                    alpha=0.6,
                    zorder=2
                )
            
            # Add "No data" label without bounding box (centered in the gap)
            mid_gap_date = gap_start + (gap_end - gap_start) / 2
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_mid = ax.get_ylim()[0] + 0.7 * y_range
            ax.text(
                mid_gap_date,
                y_mid,
                "No data",
                ha="center",
                va="center",
                fontsize=15,
                color=COLOR_TEXT,
                zorder=4
            )

        # Forecast (shaded band + mean) - matches reference code exactly
        fc = plot_df.loc[plot_df["date"] >= forecast_start_date].copy()
        fc = fc.dropna(subset=["forecast_mean"])
        
        if len(fc) > 0:
            # IMPORTANT: fill_between sometimes fails on datetime arrays.
            # Convert x to Matplotlib date numbers (float days) for fill_between
            x = mdates.date2num(np.array(fc["date"].dt.to_pydatetime()))
            
            ax.fill_between(
                x,
                fc["forecast_p10"].values,
                fc["forecast_p90"].values,
                alpha=0.25,
                linewidth=0,
                label="Forecast uncertainty (p10–p90)",
                color=COLOR_TARGET
            )
            ax.plot(
                fc["date"],
                fc["forecast_mean"],
                linewidth=2,
                label="Forecast mean",
                color=COLOR_TARGET
            )

            # Forecast boundary line + label
            ax.axvline(forecast_start_date, linewidth=1.5, alpha=0.7, color=COLOR_TARGET)
            ax.annotate(
                "Forecast starts",
                xy=(forecast_start_date, ax.get_ylim()[0] + 0.9 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=14,
                alpha=0.95,
                color=COLOR_PRIMARY
            )

    annotate_events(ax, events, y_at=s)

    # Set x-axis limit to June 2025
    ax.set_xlim(right=pd.to_datetime("2025-04-30"))
    
    # Set y-axis limit to 8 million
    ax.set_ylim(top=8000000)

    ax.set_title("The honest line chart: observed points, forecast uncertainty, annotated events", color=COLOR_TEXT, fontsize=18)
    ax.set_xlabel("Date", color=COLOR_TEXT, fontsize=15)
    ax.set_ylabel("Total API requests (daily)", color=COLOR_TEXT, fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    ax.legend(frameon=False)
    savefig("20_honest_line_template.png")


# -----------------------------
# FIGURE SET 9 (Optional): Detrended deviations (main clean data)
# -----------------------------
def plot_detrended_view(daily: pd.DataFrame):
    s = series_total(daily, "api_requests")
    trend = s.rolling(28, min_periods=7).mean()
    detrended = s / trend

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(detrended.index, detrended.values, color=COLOR_SECONDARY, linewidth=2)
    ax.axhline(1.0, linewidth=1, alpha=0.6, color=COLOR_TARGET, linestyle="--")
    style_plot(ax, "Detrended view: deviations from long-run trend", "Date", "Observed / Trend")
    savefig("21_detrended_deviations.png")


# -----------------------------
# FIGURE SET 0: Connection principle (scatter vs line)
# -----------------------------
def plot_connection_principle():
    # Create sample data points that show a clear trend when connected
    # Using a different wave-like pattern that becomes clear when connected
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([3.5, 2.2, 3.8, 2.5, 4.2, 3.0, 4.5, 3.8])
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), facecolor=COLOR_BG)
    ax_left.set_facecolor(COLOR_BG)
    ax_right.set_facecolor(COLOR_BG)
    
    # Left panel: Scatter plot (no lines)
    ax_left.scatter(x, y, s=80, color=COLOR_PRIMARY, edgecolors=COLOR_PRIMARY, linewidths=1.5, zorder=3)
    ax_left.set_xlim(0.5, 10.5)
    ax_left.set_ylim(1.0, 5.5)
    ax_left.set_title("Scatter plot: disconnected points", color=COLOR_TEXT, fontsize=18)
    ax_left.set_xlabel("Time", color=COLOR_TEXT, fontsize=15)
    ax_left.set_ylabel("Value", color=COLOR_TEXT, fontsize=15)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines["left"].set_color(COLOR_LINE)
    ax_left.spines["bottom"].set_color(COLOR_LINE)
    ax_left.tick_params(colors=COLOR_TEXT, labelsize=11)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    
    # Right panel: Line graph (connected points)
    ax_right.plot(x, y, marker="o", markersize=8, linewidth=2, color=COLOR_PRIMARY, markerfacecolor=COLOR_PRIMARY, markeredgecolor=COLOR_PRIMARY, zorder=3)
    ax_right.set_xlim(0.5, 10.5)
    ax_right.set_ylim(1.0, 5.5)
    ax_right.set_title("Line graph: connected points reveal order", color=COLOR_TEXT, fontsize=18)
    ax_right.set_xlabel("Time", color=COLOR_TEXT, fontsize=15)
    ax_right.set_ylabel("Value", color=COLOR_TEXT, fontsize=15)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["left"].set_color(COLOR_LINE)
    ax_right.spines["bottom"].set_color(COLOR_LINE)
    ax_right.tick_params(colors=COLOR_TEXT, labelsize=11)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    
    fig.suptitle("Lines connect the dots", color=COLOR_TEXT, fontsize=20, y=0.98)
    # Ensure figure background is set correctly
    fig.patch.set_facecolor(COLOR_BG)
    savefig("00_connection_principle.png")


# -----------------------------
# MAIN
# -----------------------------
def main():
    daily, monthly, events, dip = read_inputs()

    plot_connection_principle()
    plot_discrete_kpi_continuity(monthly)
    plot_dip_story(dip)  # dip story is now isolated here
    plot_irregular_spacing_trap(events)
    plot_spaghetti_and_fixes(daily)
    plot_dual_axis_trap(daily)
    plot_smoothing_trap(daily)
    plot_linear_vs_log(daily)
    plot_honest_template(daily, events)
    plot_detrended_view(daily)

    print("\nAll plots generated.")
    print(f"Output folder: {OUTDIR}")


if __name__ == "__main__":
    main()
