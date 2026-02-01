"""
Generate Step 3C/D/E bar chart figures (spacing examples).

Outputs:
- ../images/step3C_spacing_too_thin.png
- ../images/step3D_spacing_just_right.png
- ../images/step3E_spacing_too_narrow.png

Run:
python make_step3_cde_figures.py
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

REGION_SHORT = {
    "Metro City": "Metro",
    "Suburban Belt": "Suburb",
    "Small Towns": "Towns",
    "Rural Counties": "Rural",
    "Remote Communities": "Remote",
}


def main() -> None:
    # ============================================================
    # Step 3C/D/E: Spacing (gap) demo using bar width
    # ============================================================
    region_rates = (
        df.groupby("region", observed=True)
        .apply(lambda g: g["tickets_total"].sum() / g["customers_active"].sum() * 1000.0, include_groups=False)
        .reindex(REGION_ORDER)
    )
    
    demo = region_rates.copy()  # use rates for a compact example
    
    # Sort by y-value (descending order)
    demo = demo.sort_values(ascending=False)
    demo_labels = [REGION_SHORT.get(str(r), str(r)) for r in demo.index]

    # 3C: Too thin
    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    ax.bar(demo_labels, demo.values, width=0.25, color="#05A3A4")
    ax.set_title("")
    ax.set_ylabel("Tickets per 1,000 customers")
    style_axes(ax, "y")
    save_fig(fig, "step3C_spacing_too_thin")

    # 3D: Just right
    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    bar_width = 0.667  # width such that spacing = width/2
    ax.bar(demo_labels, demo.values, width=bar_width, color="#05A3A4")
    ax.set_title("")
    ax.set_ylabel("Tickets per 1,000 customers")
    style_axes(ax, "y")
    save_fig(fig, "step3D_spacing_just_right")

    # 3E: Too narrow (bars too wide, spacing too small)
    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    bar_width = 0.95  # bars very wide, spacing very narrow
    ax.bar(demo_labels, demo.values, width=bar_width, color="#05A3A4")
    ax.set_title("")
    ax.set_ylabel("Tickets per 1,000 customers")
    style_axes(ax, "y")
    save_fig(fig, "step3E_spacing_too_narrow")

    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
