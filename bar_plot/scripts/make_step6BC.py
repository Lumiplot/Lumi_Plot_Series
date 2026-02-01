"""
Generate step 6B and 6C bar chart figures with error bars.

Outputs:
- step6B_large_error_bars_small_n.png
- step6C_no_error_bars.png

Run:
python make_step6BC.py
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

REGION_SHORT = {
    "Metro City": "Metro",
    "Suburban Belt": "Suburb",
    "Small Towns": "Towns",
    "Rural Counties": "Rural",
    "Remote Communities": "Remote",
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

df["region"] = pd.Categorical(df["region"], categories=REGION_ORDER, ordered=True)

# -----------------------------
# Step 6B and 6C: Error bars with small n
# -----------------------------
rate_weekly = (
    df.groupby(["region", "week"], observed=True)["tickets_per_1k_customers"]
    .mean()
    .reset_index()
)

# 6B: Large error bars due to small n (n=3 for all regions - using weeks with largest difference)
# For ALL regions, use 3 weeks: min, max, and the point furthest from mean (to maximize difference)
# Then modify values to create variance
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
    # Keep the mean roughly the same but increase spread slightly
    current_range = max_val - min_val
    current_mean = region_data["tickets_per_1k_customers"].mean()
    
    # Expand range by 1.3x for smaller variance (reduced from 2.5x)
    expanded_range = current_range * 1.3
    new_min = current_mean - expanded_range / 2
    new_max = current_mean + expanded_range / 2
    
    # Ensure values don't go negative
    new_min = max(0.1, new_min)
    
    # Update min and max values
    region_data.loc[region_data["week"] == min_week, "tickets_per_1k_customers"] = new_min
    region_data.loc[region_data["week"] == max_week, "tickets_per_1k_customers"] = new_max
    
    # Update third value to be less extreme (reduce variance a bit)
    if third_val is not None:
        if third_val < current_mean:
            # Make it lower but not as extreme
            new_third = new_min + (new_max - new_min) * 0.25  # Place it closer to min but less extreme
        else:
            # Make it higher but not as extreme
            new_third = new_min + (new_max - new_min) * 0.75  # Place it closer to max but less extreme
        region_data.loc[region_data["week"] == third_week, "tickets_per_1k_customers"] = new_third
    
    rate_weekly_6b_filtered.append(region_data)

rate_weekly_6b = pd.concat(rate_weekly_6b_filtered, ignore_index=True)

stats_6b = rate_weekly_6b.groupby("region", observed=True)["tickets_per_1k_customers"].agg(["mean", "std", "count"])
stats_6b["se"] = stats_6b["std"] / np.sqrt(stats_6b["count"])

if st is not None:
    # Use t-critical for n-1 degrees of freedom (n=3 means df=2)
    tcrit_6b = stats_6b["count"].apply(lambda n: st.t.ppf(0.975, df=max(1, n-1)) if n > 1 else 1.96)
else:
    tcrit_6b = pd.Series(1.96, index=stats_6b.index)  # fallback

stats_6b["ci95_half"] = tcrit_6b * stats_6b["se"]
# Sort by mean rate from high to low
stats_6b = stats_6b.sort_values("mean", ascending=False)

# Calculate y-axis limits based on 6B (including error bars)
y_max_6b = (stats_6b["mean"] + stats_6b["ci95_half"]).max()
y_min_6b = max(0, (stats_6b["mean"] - stats_6b["ci95_half"]).min())
y_range_6b = y_max_6b - y_min_6b
# Reduced bottom padding (0.01 instead of 0.1) to minimize space between 0 and x-axis
y_lim_6b = (max(0, y_min_6b - y_range_6b * 0.01), y_max_6b + y_range_6b * 0.1)  # Minimal bottom padding, 10% top padding

# Step 6B: With error bars
fig, ax = plt.subplots(figsize=(6, 5.2))
bar_width = 0.667  # width such that spacing = width/2
x_labels_6b = [REGION_SHORT.get(str(r), str(r)) for r in stats_6b.index]
ax.bar(
    x_labels_6b,
    stats_6b["mean"].values,
    yerr=stats_6b["ci95_half"].values,
    capsize=6,
    error_kw={"elinewidth": 0.8},
    color="#05A3A4",
    width=bar_width,
)
ax.set_title("")
ax.set_ylabel("Tickets per 1,000 customers")
ax.set_ylim(y_lim_6b)
style_axes(ax, "y")

# Create label showing n=3 for all regions
ax.text(
    0.0,
    1.02,
    "Error bars = 95% CI across weeks (n=3 per region)",
    transform=ax.transAxes,
    fontsize=10,
    va="bottom",
)
save_fig(fig, "step6B_large_error_bars_small_n")

# Step 6C: Same as 6B but without error bars, using same y-axis scale
fig, ax = plt.subplots(figsize=(6, 5.2))
bar_width = 0.667  # width such that spacing = width/2
x_labels_6c = [REGION_SHORT.get(str(r), str(r)) for r in stats_6b.index]
ax.bar(
    x_labels_6c,
    stats_6b["mean"].values,
    color="#05A3A4",
    width=bar_width,
)
ax.set_title("")
ax.set_ylabel("Tickets per 1,000 customers")
ax.set_ylim(y_lim_6b)  # Use same y-axis scale as 6B
style_axes(ax, "y")
save_fig(fig, "step6C_no_error_bars")

print(f"Done. Figures saved to: {OUT_DIR.resolve()}")
print(f"  - step6B_large_error_bars_small_n.png")
print(f"  - step6C_no_error_bars.png")
