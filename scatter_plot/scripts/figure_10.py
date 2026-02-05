import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

rng = np.random.default_rng(777)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# -----------------------------
# New story dataset (paired): factory pass rate before vs after automation upgrade
# Story: most plants increase pass rates after upgrading inspection + process control,
# but a minority get worse (bad rollout / changeover issues).
# Use log scales because pass rates span orders of magnitude.
# -----------------------------
n = 180
factories = [f"Plant {i:03d}" for i in range(1, n+1)]
region = rng.choice(["Americas", "Europe", "Asia"], size=n, p=[0.30, 0.25, 0.45])

employees = rng.integers(120, 2200, size=n)
automation_level = np.clip(rng.normal(0.55, 0.22, size=n), 0.05, 0.98)  # 0..1

# Defects before (ppm): wide spread
log10_before = rng.normal(loc=2.6, scale=0.55, size=n)  # ~400 ppm median-ish
defects_before = 10 ** log10_before
defects_before = np.clip(defects_before, 8, 25000)

# Improvement factor depends on automation_level (higher automation -> more reduction)
base_log10_factor = -0.35 - 0.55*(automation_level - 0.5) + rng.normal(0, 0.18, size=n)

# Inject rollout failures (~12%) that get worse after the upgrade
fail_idx = rng.choice(n, size=int(0.12*n), replace=False)
base_log10_factor[fail_idx] = rng.normal(loc=0.12, scale=0.16, size=fail_idx.size)

factor = 10 ** base_log10_factor

# Mild regression-to-the-mean: extremely clean plants improve less
factor = factor * (1 + (1200/defects_before))**0.10

defects_after = defects_before * factor
defects_after = defects_after * (10 ** rng.normal(0, 0.04, size=n))  # measurement noise
defects_after = np.clip(defects_after, 4, 30000)

# Convert defect rates to pass rates: use "parts per million passing"
# To get a wide range for log scale, we'll use: pass_rate = 1,000,000 / (defects + 1)
# This gives us values that span orders of magnitude (similar to how defects work)
# Higher pass_rate = better (more parts passing per defect)
PPM_MAX = 1_000_000
pass_rate_before = PPM_MAX / (defects_before + 1)
pass_rate_after = PPM_MAX / (defects_after + 1)

df = pd.DataFrame({
    "factory": factories,
    "region": region,
    "employees": employees,
    "automation_level": np.round(automation_level, 3),
    "pass_rate_before": np.round(pass_rate_before, 1),
    "pass_rate_after": np.round(pass_rate_after, 1),
})
df["improvement_ratio_after_over_before"] = np.round(df["pass_rate_after"] / df["pass_rate_before"], 3)
df["improved"] = df["pass_rate_after"] > df["pass_rate_before"]

# -----------------------------
# Plot: paired scatter with x=y diagonal (log-log)
# -----------------------------
x = df["pass_rate_before"].to_numpy()
y = df["pass_rate_after"].to_numpy()

# Set limits - use the same range for both axes so diagonal line aligns properly
xmin_data = x.min()
xmax_data = x.max()
ymin_data = y.min()
ymax_data = y.max()

# Use the union of both ranges with padding
all_min = min(xmin_data, ymin_data)
all_max = max(xmax_data, ymax_data)

# Add padding on log scale (multiply/divide by a factor)
xmin = all_min / 1.1
xmax = all_max * 1.1
# Set both axes to same limits
ymin = xmin
ymax = xmax

fig = plt.figure(figsize=(6.8, 5.8), dpi=220)
ax = fig.add_axes([0.14, 0.14, 0.80, 0.78])

# Set log scale and limits FIRST - both axes use same limits
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Color by region using custom palette
reg_cat = pd.Categorical(df["region"])
codes = reg_cat.codes
scatter_colors = [COLORS[i % len(COLORS)] for i in codes]
sc = ax.scatter(x, y, s=16, alpha=0.80, c=scatter_colors)

# Diagonal x=y line - since both axes have same limits, diagonal spans full range
diag = np.logspace(np.log10(xmin), np.log10(xmax), 240)
ax.plot(diag, diag, linewidth=0.7, color='gray', alpha=0.5, linestyle='--')

ax.set_xlabel("Pass rate before upgrade (ppm, log scale)")
ax.set_ylabel("Pass rate after upgrade (ppm, log scale)")
ax.set_title(
    "Before vs after (paired): pass rates per plant\n"
    "Diagonal is x = y (no change); above diagonal = improvement",
    fontsize=10
)
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)

improved_pct = 100 * df["improved"].mean()
# Position text in upper left area
ax.text(xmin*1.15, ymax/1.7, f"{improved_pct:.0f}% above diagonal\n(improved)", fontsize=9)

# Legend that matches the scatter colors
handles, labels = [], []
for i, name in enumerate(reg_cat.categories):
    handles.append(Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=COLORS[i % len(COLORS)], markeredgecolor="none",
                          markersize=7))
    labels.append(name)

ax.legend(handles, labels, title="Region", frameon=True, fontsize=8, title_fontsize=8, loc="lower right")

# Save plot
output_path = Path("../images/figure_10.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, facecolor="white", dpi=220)
print(f"Figure 10 saved to {output_path.absolute()}")

plt.close(fig)

