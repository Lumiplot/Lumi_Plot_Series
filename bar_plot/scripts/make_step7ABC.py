"""
Generate step 7A, 7B, and 7C bar chart figures for labeling strategies.

Outputs:
- step7A_best_label_directly.png
- step7B_good_axis_gridlines.png
- step7C_no_gridlines_no_labels.png

Run:
python make_step7ABC.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Set font to Inter
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# Paths
SCRIPT_DIR = Path(__file__).parent  # scripts folder
DATA_PATH = SCRIPT_DIR.parent.parent / "draft" / "dataset.csv"  # Dataset is in draft folder
OUT_DIR = SCRIPT_DIR.parent / "images"  # images folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKET_TYPE_MAP = {
    "tickets_login": "Cannot log in or reset password",
    "tickets_payment": "Payment fails at checkout",
    "tickets_delivery": "Delivery delayed or missing",
    "tickets_wrong_item": "Wrong item received or damaged",
    "tickets_promo": "Promo code not applying correctly",
    "tickets_refund": "Billed twice or refund issue",
}


# -----------------------------
# Helpers
# -----------------------------
def style_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
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


def step7_prepare_ticket_totals(df: pd.DataFrame) -> pd.Series:
    """Prepare ticket totals sorted by value."""
    totals = df[list(TICKET_TYPE_MAP.keys())].sum()
    totals.index = [TICKET_TYPE_MAP[c] for c in totals.index]
    totals = totals.sort_values(ascending=False)
    return totals


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)
ticket_totals = step7_prepare_ticket_totals(df)

# Common settings
bar_height = 0.667  # gap = 0.5 * bar_height
figsize = (9, 3.5)

# -----------------------------
# Step 7A: Best - Label directly (no axis lookup)
# -----------------------------
fig, ax = plt.subplots(figsize=figsize)
ax.barh(ticket_totals.index[::-1], ticket_totals.values[::-1], color="#05A3A4", height=bar_height)

ax.set_xlabel("Tickets (8 weeks, all regions)")
ax.set_ylabel("")

style_axes(ax, grid_axis=None)

# Set font for tick labels
ax.tick_params(axis='both', labelsize=9)
for label in ax.get_xticklabels():
    label.set_fontfamily('Inter')
for label in ax.get_yticklabels():
    label.set_fontfamily('Inter')

# Direct value labels at end of bars
x0, x1 = ax.get_xlim()
rng = (x1 - x0) if (x1 != x0) else 1.0
pad = 0.005 * rng  # Reduced padding to bring labels closer to bars

for p in ax.patches:
    w = p.get_width()
    y = p.get_y() + p.get_height() / 2
    ax.text(w + pad, y, f"{w:.0f}", va="center", ha="left", fontsize=9, fontfamily='Inter')

save_fig(fig, "step7A_best_label_directly")

# -----------------------------
# Step 7B: Good - Axis + gridlines (no labels)
# -----------------------------
fig, ax = plt.subplots(figsize=figsize)
ax.barh(ticket_totals.index[::-1], ticket_totals.values[::-1], color="#05A3A4", height=bar_height)

ax.set_xlabel("Tickets (8 weeks, all regions)")
ax.set_ylabel("")

# Keep x-axis ticks and add x-gridlines
style_axes(ax, grid_axis="x")

# Set font for tick labels
ax.tick_params(axis='both', labelsize=9)
for label in ax.get_xticklabels():
    label.set_fontfamily('Inter')
for label in ax.get_yticklabels():
    label.set_fontfamily('Inter')

save_fig(fig, "step7B_good_axis_gridlines")

# -----------------------------
# Step 7C: No gridlines, no labels
# -----------------------------
fig, ax = plt.subplots(figsize=figsize)
ax.barh(ticket_totals.index[::-1], ticket_totals.values[::-1], color="#05A3A4", height=bar_height)

ax.set_xlabel("Tickets (8 weeks, all regions)")
ax.set_ylabel("")

# No gridlines
style_axes(ax, grid_axis=None)

# Set font for tick labels
ax.tick_params(axis='both', labelsize=9)
for label in ax.get_xticklabels():
    label.set_fontfamily('Inter')
for label in ax.get_yticklabels():
    label.set_fontfamily('Inter')

# No data labels on bars

save_fig(fig, "step7C_no_gridlines_no_labels")

print(f"Done. Figures saved to: {OUT_DIR.resolve()}")
print(f"  - step7A_best_label_directly.png")
print(f"  - step7B_good_axis_gridlines.png")
print(f"  - step7C_no_gridlines_no_labels.png")
