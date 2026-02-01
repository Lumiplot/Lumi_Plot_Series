"""
Generate Step 2 bar chart figures (make it readable at a glance - labels and orientation).

Outputs:
- ../images/step2A_vertical_long_labels.png
- ../images/step2B_horizontal_long_labels.png

Run:
python make_step2_figures.py
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

TICKET_TYPE_MAP = {
    "tickets_login": "Cannot log in or reset password",
    "tickets_payment": "Payment fails at checkout",
    "tickets_delivery": "Delivery delayed or missing",
    "tickets_wrong_item": "Wrong item received or damaged",
    "tickets_promo": "Promo code not applying correctly",
    "tickets_refund": "Billed twice or refund issue",
}


def main() -> None:
    # ============================================================
    # Step 2: Make it readable at a glance (labels and orientation)
    # ============================================================
    ticket_totals = df[list(TICKET_TYPE_MAP.keys())].sum().rename(index=TICKET_TYPE_MAP)
    ticket_totals = ticket_totals.sort_values(ascending=False)

    # 2A: Vertical bars with long labels (hard to read, tempting to rotate)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bar_width = 0.667  # width such that spacing = width/2
    ax.bar(ticket_totals.index, ticket_totals.values, color="#05A3A4", width=bar_width)
    ax.set_title("")
    ax.set_ylabel("Tickets (8 weeks, all regions)")
    ax.tick_params(axis="x", labelrotation=35)
    style_axes(ax, "y")
    save_fig(fig, "step2A_vertical_long_labels")

    # 2B: Horizontal bars for long labels (usually better)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bar_height = 0.667  # height such that spacing = height/2
    ax.barh(ticket_totals.index[::-1], ticket_totals.values[::-1], color="#05A3A4", height=bar_height)
    ax.set_title("")
    ax.set_xlabel("Tickets (8 weeks, all regions)")
    style_axes(ax, "x")
    annotate_bars(ax, fmt="{:.0f}", axis="x", pad=0.01)
    save_fig(fig, "step2B_horizontal_long_labels")

    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
