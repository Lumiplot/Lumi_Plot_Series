"""
Generate Step 4 bar chart figures (replace legends with direct answers - color and text labels).

Outputs:
- ../images/step4_direct_answer_color_and_labels.png
- ../images/step4A_different_colors.png
- ../images/step4B_single_color.png

Run:
python make_step4_figures.py
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
# Neutral gray for non-emphasized elements
NEUTRAL_COLOR = "#bdbdbd"

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
    # Step 4: Replace legends with direct answers (color and text labels)
    # ============================================================
    region_rates = (
        df.groupby("region", observed=True)
        .apply(lambda g: g["tickets_total"].sum() / g["customers_active"].sum() * 1000.0, include_groups=False)
        .reindex(REGION_ORDER)
    )
    
    # Sort by values from small to large
    region_rates_sorted = region_rates.sort_values(ascending=True)
    
    # Highlight the highest rate region
    top_region = region_rates_sorted.idxmax()

    fig, ax = plt.subplots(figsize=(7, 5.0))
    labels = region_rates_sorted.index.astype(str).tolist()
    vals = region_rates_sorted.values

    colors = [NEUTRAL_COLOR if r != top_region else COLOR_PALETTE["Zest"] for r in region_rates_sorted.index]  # neutral + emphasis
    bar_height = 0.667  # gap = 0.5 * bar_height
    ax.barh(labels, vals, color=colors, height=bar_height)
    ax.set_title("")
    ax.set_xlabel("Tickets per 1,000 customers (8 weeks)")
    style_axes(ax, "x")

    # Direct labels
    for i, (lab, v) in enumerate(zip(labels, vals)):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9)
    save_fig(fig, "step4_direct_answer_color_and_labels")

    # 4A: Each bar uses different color
    fig, ax = plt.subplots(figsize=(7, 5.0))
    labels_4a = region_rates_sorted.index.astype(str).tolist()
    vals_4a = region_rates_sorted.values
    
    # Use different colors from palette for each bar
    palette_colors = list(COLOR_PALETTE.values())
    colors_4a = [palette_colors[i % len(palette_colors)] for i in range(len(labels_4a))]
    
    bar_height = 0.667  # gap = 0.5 * bar_height
    ax.barh(labels_4a, vals_4a, color=colors_4a, height=bar_height)
    ax.set_title("")
    ax.set_xlabel("Tickets per 1,000 customers (8 weeks)")
    style_axes(ax, "x")

    # Direct labels
    for i, (lab, v) in enumerate(zip(labels_4a, vals_4a)):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9)
    save_fig(fig, "step4A_different_colors")

    # 4B: Single color #05a3a4
    fig, ax = plt.subplots(figsize=(7, 5.0))
    labels_4b = region_rates_sorted.index.astype(str).tolist()
    vals_4b = region_rates_sorted.values
    
    # Use single color for all bars
    bar_height = 0.667  # gap = 0.5 * bar_height
    ax.barh(labels_4b, vals_4b, color="#05A3A4", height=bar_height)
    ax.set_title("")
    ax.set_xlabel("Tickets per 1,000 customers (8 weeks)")
    style_axes(ax, "x")

    # Direct labels
    for i, (lab, v) in enumerate(zip(labels_4b, vals_4b)):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9)
    save_fig(fig, "step4B_single_color")

    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
