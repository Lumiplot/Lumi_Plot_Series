import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# Set font to Inter
plt.rcParams['font.family'] = 'Inter'
plt.rcParams['font.sans-serif'] = ['Inter', 'sans-serif']

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# Load the synthetic dataset
script_dir = Path(__file__).parent
data_path = script_dir.parent / "data" / "synthetic_scatter_master_dataset.csv"
df = pd.read_csv(data_path)

# Subset: week8 + moderate severity (gives a clearer positive slope)
wk8 = df[df["visit"] == "week8"].copy()
d = wk8[wk8["severity_band"] == "moderate"].copy()

# Create more "country-like" units by splitting each site into 4 pseudo-countries (deterministic)
d["country_bucket"] = (d["subject_id"] % 4).astype(int)
d["country"] = d["site_id"] + "-" + d["country_bucket"].astype(str)

# Aggregate to one dot per pseudo-country
g = (
    d.groupby(["region", "country"], as_index=False)
     .agg(
         n=("subject_id", "nunique"),
         exposure_p90=("exposure_auc", lambda x: np.quantile(x, 0.90)),
         response_mean=("response", "mean"),
         cost_p90=("cost_usd", lambda x: np.quantile(x, 0.90)),
         adverse_rate=("adverse_event", "mean"),
     )
)

# Keep small groups too (to increase point count) but avoid singletons
g = g[g["n"] >= 3].copy().reset_index(drop=True)

# X / Y
x = g["exposure_p90"].to_numpy()
y = g["response_mean"].to_numpy()

# Bubble size: extreme scaling based on n and adverse rate (to create a few monsters)
n = g["n"].to_numpy().astype(float)
s_raw = (n**3.1) * (1.0 + 1.2 * g["adverse_rate"].to_numpy())
s = 10 + 4200 * (s_raw - s_raw.min()) / (s_raw.max() - s_raw.min() + 1e-9)
s = np.clip(s, 10, 5200)

# Color: region
regions = g["region"].astype("category")
region_codes = regions.cat.codes.to_numpy()

# Trend: y ~ a + b*log10(x)
lx = np.log10(x + 1e-6)
b, a = np.polyfit(lx, y, 1)
lx_line = np.linspace(lx.min(), lx.max(), 280)
x_line = 10**lx_line
y_line = a + b * lx_line

# Outliers: residuals + extremes
resid = y - (a + b * lx)
abs_resid = np.abs(resid)

k = 10
out_idx = np.argsort(abs_resid)[-k:]
out_mask = np.zeros_like(abs_resid, dtype=bool)
out_mask[out_idx] = True
out_mask[int(np.argmax(x))] = True
out_mask[int(np.argmin(y))] = True
out_mask[int(np.argmax(y))] = True

labels = g["country"].to_numpy()

# Exclude "11-1" from being highlighted as an outlier
for idx in range(len(labels)):
    if "11-1" in labels[idx]:
        out_mask[idx] = False

# Plot
plt.figure(figsize=(13.2, 6.6))

# Map region codes to colors
scatter_colors = np.array([COLORS[i % len(COLORS)] for i in region_codes])

# Non-outliers
plt.scatter(
    x[~out_mask], y[~out_mask],
    s=s[~out_mask],
    c=scatter_colors[~out_mask],
    alpha=0.55,
    linewidths=0
)

# Outliers emphasized
plt.scatter(
    x[out_mask], y[out_mask],
    s=s[out_mask] * 1.20,
    c=scatter_colors[out_mask],
    alpha=0.95,
    linewidths=1.1,
    edgecolors="black"
)

# Trend line (thicker)
plt.plot(x_line, y_line, linewidth=3.0, color=COLORS[0])

# Axes styling
plt.xscale("log")
plt.xlabel("Exposure (AUC p90 per country, log scale)")
plt.ylabel("Mean outcome score (week8, moderate subgroup)")
plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.28)

# Adjust x-axis limits to accommodate labels on the right side
x_min, x_max = x.min(), x.max()
# Add extra space on the right for labels (about 20% more)
x_max_expanded = x_max * 1.2
plt.xlim(x_min * 0.9, x_max_expanded)

# Label outliers (positioned to the right of data points)
for idx in np.where(out_mask)[0]:
    # Skip labeling "EU-02-01" (or similar pattern)
    if "EU-02-01" in labels[idx] or "EU-02-1" in labels[idx]:
        continue
    # Skip labeling "11-1"
    if "11-1" in labels[idx]:
        continue
    # Position labels to the right with small proportional offset (works with log scale)
    # Multiply x-value by a small factor for consistent visual spacing on log scale
    offset = x[idx] * 0.02  # 2% of x-value for very close, consistent spacing
    plt.text(x[idx] + offset, y[idx], labels[idx], fontsize=9.5, 
             verticalalignment='center', horizontalalignment='left')

# Region legend (using custom colors)
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", linestyle="", 
                  markerfacecolor=COLORS[i % len(COLORS)], markeredgecolor="none",
                  markersize=8) for i in range(len(regions.cat.categories))]
plt.legend(handles, regions.cat.categories, title="Region", frameon=True, 
          fontsize=8, title_fontsize=8, loc="upper left")

plt.title("More dots (pseudo-countries) + stronger slope + clearly highlighted outliers")

plt.tight_layout()

# Save the plot BEFORE showing
images_dir = Path("/Users/dullmanatee/PycharmProjects/scatter/Scatter/images")
output_path1 = images_dir / "figure_1_bubble_scatter.png"
output_path2 = images_dir / "bubble_scatter_more_data_outliers_clear_slope.png"
plt.savefig(output_path1, dpi=220, facecolor="white")
plt.savefig(output_path2, dpi=220, facecolor="white")
print(f"Saved: {output_path1}")
print(f"Saved: {output_path2}")

plt.show()
plt.close()