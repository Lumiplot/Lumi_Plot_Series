"""
Generate Step 0 bar chart figures (counts vs rates by region).

Outputs:
- ../images/step0A_counts_by_region.png
- ../images/step0B_rates_by_region.png

Run:
python make_step0_figures.py
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
    # Step 0: Decide what "more" means (counts vs rates)
    # ============================================================
    region_counts = (
        df.groupby("region", observed=True)["tickets_total"]
        .sum()
        .reindex(REGION_ORDER)
    )

    region_rates = (
        df.groupby("region", observed=True)
        .apply(lambda g: g["tickets_total"].sum() / g["customers_active"].sum() * 1000.0, include_groups=False)
        .reindex(REGION_ORDER)
    )

    # Create single-word labels for x-axis
    region_labels_short = {
        "Metro City": "Metro",
        "Suburban Belt": "Suburban",
        "Remote Communities": "Remote",
        "Rural Counties": "Rural",
        "Small Towns": "Small",
    }

    # 0A: Counts by region
    # Reduced width, smaller text labels
    fig_width = 4.5
    fig_height = 4.8  # Keep original height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bar_width = 0.667  # width such that spacing = width/2
    
    x_labels_0a = [region_labels_short.get(str(r), str(r)) for r in region_counts.index]
    
    ax.bar(x_labels_0a, region_counts.values, color="#05A3A4", width=bar_width)
    ax.set_title("")  # No title
    ax.set_ylabel("Total tickets (8 weeks)", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    style_axes(ax, "y")
    # Annotate bars with reduced spacing
    y0, y1 = ax.get_ylim()
    rng = y1 - y0 if y1 != y0 else 1
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            x = p.get_x() + p.get_width() / 2
            ax.text(x, h + 0.005 * rng, f"{h:.0f}", ha="center", va="bottom", fontsize=7)
    save_fig(fig, "step0A_counts_by_region")

    # 0B: Rates by region
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    x_labels_0b = [region_labels_short.get(str(r), str(r)) for r in region_rates.index]
    ax.bar(x_labels_0b, region_rates.values, color="#05A3A4", width=bar_width)
    ax.set_title("")  # No title
    ax.set_ylabel("Tickets per 1,000 customers (8 weeks)", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    style_axes(ax, "y")
    # Annotate bars with reduced spacing
    y0, y1 = ax.get_ylim()
    rng = y1 - y0 if y1 != y0 else 1
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            x = p.get_x() + p.get_width() / 2
            ax.text(x, h + 0.005 * rng, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    save_fig(fig, "step0B_rates_by_region")

    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
