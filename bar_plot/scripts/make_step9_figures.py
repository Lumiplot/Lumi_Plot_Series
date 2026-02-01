"""
Generate step 9 bar chart figure with custom labels.

Outputs:
- step9_custom_labels.png

Run:
python make_step9_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set font to Inter
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# Optional (better CI for small n). If missing, we fall back to 1.96.
try:
    import scipy.stats as st  # type: ignore
except Exception:
    st = None

# Paths
SCRIPT_DIR = Path(__file__).parent  # scripts folder
DATA_PATH = SCRIPT_DIR.parent.parent / "draft" / "dataset.csv"  # Dataset is in draft folder (go up: scripts -> final -> bars -> draft)
OUT_DIR = SCRIPT_DIR.parent / "images"  # images folder (go up: scripts -> final -> images)
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_ORDER = ["Metro City", "Suburban Belt", "Remote Communities", "Rural Counties", "Small Towns"]


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
# Load + derive columns (same as step 6C)
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Safer types
df["week"] = df["week"].astype(int)
df["week_start"] = pd.to_datetime(df["week_start"])
df["period"] = pd.Categorical(df["period"], categories=["Before", "After"], ordered=True)

# Rates
df["tickets_per_1k_customers"] = df["tickets_total"] / df["customers_active"] * 1000.0
df["tickets_per_10k_orders"] = df["tickets_total"] / df["orders"] * 10000.0

df["region"] = pd.Categorical(df["region"], categories=REGION_ORDER, ordered=True)

# -----------------------------
# Step 9: Same data as step 6C but with custom labels
# -----------------------------
rate_weekly = (
    df.groupby(["region", "week"], observed=True)["tickets_per_1k_customers"]
    .mean()
    .reset_index()
)

# Use same filtering logic as step 6B to get the same 3 weeks per region
rate_weekly_6b_filtered = []
for region in REGION_ORDER:
    region_data = rate_weekly[rate_weekly["region"] == region].copy()
    # Find min and max weeks
    min_idx = region_data["tickets_per_1k_customers"].idxmin()
    max_idx = region_data["tickets_per_1k_customers"].idxmax()
    min_val = region_data.loc[min_idx, "tickets_per_1k_customers"]
    max_val = region_data.loc[max_idx, "tickets_per_1k_customers"]
    min_week = region_data.loc[min_idx, "week"]
    max_week = region_data.loc[max_idx, "week"]
    
    # Find third point: the one furthest from the mean (excluding min and max)
    mean_val = region_data["tickets_per_1k_customers"].mean()
    remaining_data = region_data[~region_data["week"].isin([min_week, max_week])].copy()
    if len(remaining_data) > 0:
        remaining_data["dist_from_mean"] = abs(remaining_data["tickets_per_1k_customers"] - mean_val)
        third_idx = remaining_data["dist_from_mean"].idxmax()
        third_week = remaining_data.loc[third_idx, "week"]
        third_val = remaining_data.loc[third_idx, "tickets_per_1k_customers"]
        region_data = region_data[region_data["week"].isin([min_week, max_week, third_week])].copy()
    else:
        # If only 2 weeks available, use both
        region_data = region_data[region_data["week"].isin([min_week, max_week])].copy()
        third_val = None
    
    # Modify values to create variance: expand the range by 1.3x
    current_range = max_val - min_val
    current_mean = region_data["tickets_per_1k_customers"].mean()
    
    # Expand range by 1.3x
    expanded_range = current_range * 1.3
    new_min = current_mean - expanded_range / 2
    new_max = current_mean + expanded_range / 2
    
    # Ensure values don't go negative
    new_min = max(0.1, new_min)
    
    # Update min and max values
    region_data.loc[region_data["week"] == min_week, "tickets_per_1k_customers"] = new_min
    region_data.loc[region_data["week"] == max_week, "tickets_per_1k_customers"] = new_max
    
    # Update third value
    if third_val is not None:
        if third_val < current_mean:
            new_third = new_min + (new_max - new_min) * 0.25
        else:
            new_third = new_min + (new_max - new_min) * 0.75
        region_data.loc[region_data["week"] == third_week, "tickets_per_1k_customers"] = new_third
    
    rate_weekly_6b_filtered.append(region_data)

rate_weekly_6b = pd.concat(rate_weekly_6b_filtered, ignore_index=True)

stats_6b = rate_weekly_6b.groupby("region", observed=True)["tickets_per_1k_customers"].agg(["mean", "std", "count"])
stats_6b["se"] = stats_6b["std"] / np.sqrt(stats_6b["count"])

if st is not None:
    tcrit_6b = stats_6b["count"].apply(lambda n: st.t.ppf(0.975, df=max(1, n-1)) if n > 1 else 1.96)
else:
    tcrit_6b = pd.Series(1.96, index=stats_6b.index)

stats_6b["ci95_half"] = tcrit_6b * stats_6b["se"]
# Sort by mean rate from high to low
stats_6b = stats_6b.sort_values("mean", ascending=False)

# Calculate y-axis limits (same as step 6C)
y_max_6b = (stats_6b["mean"] + stats_6b["ci95_half"]).max()
y_min_6b = max(0, (stats_6b["mean"] - stats_6b["ci95_half"]).min())
y_range_6b = y_max_6b - y_min_6b
y_lim_6b = (max(0, y_min_6b - y_range_6b * 0.01), y_max_6b + y_range_6b * 0.1)

# Step 9: Custom labels - x labels are a, b, c, d, e; y label is "values"
fig, ax = plt.subplots(figsize=(6, 5.2))
bar_width = 0.667  # width such that spacing = width/2
x_labels_9 = ["a", "b", "c", "d", "e"]  # Custom x-axis labels
ax.bar(
    x_labels_9,
    stats_6b["mean"].values,
    color="#bfd5c9",  # Custom bar color
    width=bar_width,
)
ax.set_title("")
ax.set_ylabel("values")  # Custom y-axis label
ax.set_ylim(y_lim_6b)  # Use same y-axis scale as step 6C
style_axes(ax, "y")
save_fig(fig, "step9_custom_labels")

print(f"Done. Figure saved to: {OUT_DIR.resolve()}")
print(f"  - step9_custom_labels.png")
