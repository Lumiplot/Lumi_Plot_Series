"""
Generate Step 3 bar chart figures (remove chartjunk - 3D and spacing).

Outputs:
- ../images/step3A_bad_3d_grouped.png
- ../images/step3B_good_2d_grouped.png

Run:
python make_step3_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    # Step 3: Remove chartjunk (3D and spacing)
    # ============================================================
    grouped = (
        df.groupby(["region", "period"], observed=True)["tickets_total"]
        .sum()
        .unstack("period")
        .reindex(REGION_ORDER)
    )

    # 3A: 3D grouped bars (example of what not to do)
    fig = plt.figure(figsize=(6, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    xlabels = [REGION_SHORT.get(str(r), str(r)) for r in grouped.index]
    ylabels = grouped.columns.astype(str).tolist()

    xpos = np.arange(len(xlabels))
    ypos = np.arange(len(ylabels))
    xposM, yposM = np.meshgrid(xpos, ypos, indexing="ij")

    x = xposM.ravel()
    y = yposM.ravel()
    z = np.zeros_like(x)

    dx = 0.6
    dy = 0.6
    dz = grouped.to_numpy().ravel()

    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color="#05A3A4")
    ax.set_title("")
    ax.set_zlabel("Total tickets (8 weeks)")
    ax.set_xticks(xpos + dx / 2)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_yticks(ypos + dy / 2)
    ax.set_yticklabels(ylabels)
    save_fig(fig, "step3A_bad_3d_grouped")

    # 3B: 2D grouped bars (better)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    w = 0.38
    x = np.arange(len(grouped.index))
    ax.bar(x - w / 2, grouped["Before"].values, width=w, label="Before", color="#E8891D")
    ax.bar(x + w / 2, grouped["After"].values, width=w, label="After", color="#05A3A4")
    ax.set_title("")
    ax.set_ylabel("Total tickets (8 weeks)")
    ax.set_xticks(x)
    xlabels_3b = [REGION_SHORT.get(str(r), str(r)) for r in grouped.index]
    ax.set_xticklabels(xlabels_3b, rotation=0)
    style_axes(ax, "y")
    ax.legend(frameon=False)
    save_fig(fig, "step3B_good_2d_grouped")

    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
