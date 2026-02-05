"""
Generate Step 1 bar chart figures (protect the baseline - scale is the trust anchor).

Outputs:
- ../images/step1A_csat_start_at_zero.png
- ../images/step1B_csat_truncated_axis.png
- ../images/step1C_customers_linear.png
- ../images/step1D_customers_log.png

Run:
python make_step1_figures.py
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


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    return float(np.sum(v * w) / np.sum(w))


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
    # Step 1: Protect the baseline (scale is the trust anchor)
    #   Use the same bars twice: honest (start at 0) vs misleading (truncated)
    # ============================================================
    # Weighted CSAT by period (weights = customers_active)
    csat_by_period = (
        df.groupby(["period", "week"], observed=True)
        .apply(lambda g: weighted_mean(g["csat_avg_1_to_5"], g["customers_active"]), include_groups=False)
        .reset_index(name="csat_weighted")
        .groupby("period", observed=True)["csat_weighted"]
        .mean()
        .reindex(["Before", "After"])
    )

    # 1A: Baseline at zero (honest for bar length)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bar_width = 0.667  # width such that spacing = width/2
    ax.bar(csat_by_period.index.astype(str), csat_by_period.values, color="#05A3A4", width=bar_width)
    ax.set_title("")
    ax.set_ylabel("Average CSAT (1 to 5)")
    ax.set_ylim(0, 5)
    style_axes(ax, "y")
    save_fig(fig, "step1A_csat_start_at_zero")

    # 1B: Truncated axis (makes small differences look huge)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.bar(csat_by_period.index.astype(str), csat_by_period.values, color="#05A3A4", width=bar_width)
    ax.set_title("")
    ax.set_ylabel("Average CSAT (1 to 5)")
    ymin = float(csat_by_period.min()) - 0.05
    ymax = float(csat_by_period.max()) + 0.05
    ax.set_ylim(ymin, ymax)
    style_axes(ax, "y")
    save_fig(fig, "step1B_csat_truncated_axis")

    # 1C: Log scale example (only when story is multiplicative)
    # Customers active spans ~300x across regions.
    customers_by_region = (
        df.groupby("region", observed=True)["customers_active"]
        .first()
        .reindex(REGION_ORDER)
    )

    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    bar_width = 0.667  # width such that spacing = width/2
    x_labels_1c = [REGION_SHORT.get(str(r), str(r)) for r in customers_by_region.index]
    ax.bar(x_labels_1c, customers_by_region.values, color="#05A3A4", width=bar_width)
    ax.set_title("")
    ax.set_ylabel("Customers active")
    style_axes(ax, "y")
    save_fig(fig, "step1C_customers_linear")

    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    x_labels_1d = [REGION_SHORT.get(str(r), str(r)) for r in customers_by_region.index]
    ax.bar(x_labels_1d, customers_by_region.values, color="#05A3A4", width=bar_width)
    ax.set_title("")
    ax.set_ylabel("Customers active (log)")
    ax.set_yscale("log")
    style_axes(ax, "y")
    save_fig(fig, "step1D_customers_log")

    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
