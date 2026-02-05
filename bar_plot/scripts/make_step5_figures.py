"""
Generate Step 5 bar chart figures (sorting is a narrative choice, not a default).

Outputs:
- ../images/step5A_sort_for_ranking.png
- ../images/step5B_sort_alphabetical.png
- ../images/step5C_time_order_correct.png
- ../images/step5D_time_order_wrong_sorted.png

Run:
python make_step5_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set Inter font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / "draft" / "dataset.csv"
DATA_STEP5_PATH = Path(__file__).parent.parent.parent / "draft" / "dataset_step5.csv"
OUT_DIR = Path(__file__).parent.parent / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLOR_PALETTE = {
    "Desert": "#B35A20",      # Deep reddish-brown
    "Zest": "#E8891D",         # Vibrant orange
    "Sea Mist": "#BFD5C9",     # Light blue-green
    "Niagara": "#05A3A4",      # Bright teal
    "Mosque": "#006373",       # Dark teal
}

# -----------------------------
# Helpers
# -----------------------------
def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Lightweight, readable styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in ("x", "y"):
        ax.grid(True, axis=grid_axis, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}", axis: str = "y", pad: float = 0.01) -> None:
    """
    Add value labels to bars.
    axis='y' for vertical bars, axis='x' for horizontal bars.
    pad is fraction of axis range.
    """
    if axis == "y":
        y0, y1 = ax.get_ylim()
        rng = y1 - y0 if y1 != y0 else 1
        for p in ax.patches:
            h = p.get_height()
            if np.isnan(h):
                continue
            x = p.get_x() + p.get_width() / 2
            ax.text(x, h + pad * rng, fmt.format(h), ha="center", va="bottom", fontsize=9)
    else:
        x0, x1 = ax.get_xlim()
        rng = x1 - x0 if x1 != x0 else 1
        for p in ax.patches:
            w = p.get_width()
            if np.isnan(w):
                continue
            y = p.get_y() + p.get_height() / 2
            ax.text(w + pad * rng, y, fmt.format(w), ha="left", va="center", fontsize=9)


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG."""
    png_path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Load + derive columns
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Safer types
df["week"] = df["week"].astype(int)
df["week_start"] = pd.to_datetime(df["week_start"])
df["period"] = pd.Categorical(df["period"], categories=["Before", "After"], ordered=True)

# Rates
df["tickets_per_1k_customers"] = df["tickets_total"] / df["customers_active"] * 1000.0
df["tickets_per_10k_orders"] = df["tickets_total"] / df["orders"] * 10000.0

REGION_ORDER = ["Metro City", "Suburban Belt", "Remote Communities", "Rural Counties", "Small Towns"]
df["region"] = pd.Categorical(df["region"], categories=REGION_ORDER, ordered=True)


def main() -> None:
    # ============================================================
    # Step 5: Sorting is a narrative choice, not a default
    # ============================================================
    region_rates = (
        df.groupby("region", observed=True)
        .apply(lambda g: g["tickets_total"].sum() / g["customers_active"].sum() * 1000.0, include_groups=False)
        .reindex(REGION_ORDER)
    )
    
    # 5A: Sort when the goal is ranking (regions have no natural order)
    fig, ax = plt.subplots(figsize=(7, 5.0))
    rank = region_rates.sort_values(ascending=True)
    bar_height = 0.667  # height such that spacing = height/2
    ax.barh(rank.index.astype(str), rank.values, color="#05A3A4", height=bar_height)
    ax.set_title("")
    ax.set_xlabel("Tickets per 1,000 customers (8 weeks)")
    style_axes(ax, "x")
    annotate_bars(ax, fmt="{:.2f}", axis="x", pad=0.01)
    save_fig(fig, "step5A_sort_for_ranking")

    # 5B: Sort alphabetically (not meaningful for regions)
    fig, ax = plt.subplots(figsize=(7, 5.0))
    region_rates_alpha = region_rates.sort_index()  # Sort alphabetically by region name
    bar_height = 0.667  # height such that spacing = height/2
    ax.barh(region_rates_alpha.index.astype(str), region_rates_alpha.values, color="#05A3A4", height=bar_height)
    ax.set_title("")
    ax.set_xlabel("Tickets per 1,000 customers (8 weeks)")
    style_axes(ax, "x")
    annotate_bars(ax, fmt="{:.2f}", axis="x", pad=0.01)
    save_fig(fig, "step5B_sort_alphabetical")

    # 5C: Do not sort when order is meaningful (weeks are time)
    # Load dataset with random weekly values for step 5C and 5D
    df_step5 = pd.read_csv(DATA_STEP5_PATH)
    df_step5["week"] = df_step5["week"].astype(int)
    
    weekly = (
        df_step5.groupby("week", observed=True)["tickets_total"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5.0))
    bar_width = 0.667  # width such that spacing = width/2
    ax.bar(weekly.index.astype(str), weekly.values, color="#05A3A4", width=bar_width)
    ax.set_title("")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total tickets (all regions)")
    style_axes(ax, "y")
    save_fig(fig, "step5C_time_order_correct")

    # 5D: What goes wrong if you sort time (destroys the story)
    fig, ax = plt.subplots(figsize=(7, 5.0))
    weekly_sorted = weekly.sort_values(ascending=False)
    ax.bar(weekly_sorted.index.astype(str), weekly_sorted.values, color="#05A3A4", width=bar_width)
    ax.set_title("")
    ax.set_xlabel("Week (sorted, not time)")
    ax.set_ylabel("Total tickets (all regions)")
    style_axes(ax, "y")
    save_fig(fig, "step5D_time_order_wrong_sorted")

    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
