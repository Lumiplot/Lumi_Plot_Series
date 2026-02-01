import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

rng = np.random.default_rng(20251213)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# ------------------------------
# Synthetic dataset: "Average adult BMI by country (men vs women)"
# ------------------------------
regions = [
    "North America", "Latin America", "Europe", "Africa", "Middle East",
    "South Asia", "East Asia", "Southeast Asia", "Oceania"
]
region_sizes = [12, 14, 18, 18, 10, 12, 12, 12, 8]  # total 116

def make_names(prefix, n):
    syll = ["land", "ia", "stan", "terra", "vale", "nia", "ford", "mark", "ton", "ria"]
    return [f"{prefix}-{i+1:02d}{syll[i % len(syll)]}" for i in range(n)]

country_rows = []
for reg, n in zip(regions, region_sizes):
    if reg == "North America":
        mu_m, mu_f, sd = 28.8, 27.2, 1.6
    elif reg == "Latin America":
        mu_m, mu_f, sd = 27.4, 26.6, 1.7
    elif reg == "Europe":
        mu_m, mu_f, sd = 26.6, 25.6, 1.6
    elif reg == "Africa":
        mu_m, mu_f, sd = 23.8, 24.6, 1.9
    elif reg == "Middle East":
        mu_m, mu_f, sd = 28.6, 29.2, 1.8
    elif reg == "South Asia":
        mu_m, mu_f, sd = 23.2, 23.8, 1.4
    elif reg == "East Asia":
        mu_m, mu_f, sd = 23.6, 22.8, 1.2
    elif reg == "Southeast Asia":
        mu_m, mu_f, sd = 24.0, 23.6, 1.3
    else:  # Oceania
        mu_m, mu_f, sd = 29.4, 28.2, 1.8

    # correlated male/female BMI within each region
    male = rng.normal(mu_m, sd, n)
    female = male - rng.normal(1.0, 0.9, n) + rng.normal(mu_f - mu_m, 0.6, n)

    male = np.clip(male, 18.0, 35.5)
    female = np.clip(female, 18.0, 35.5)

    names = make_names(reg.split()[0], n)
    for i in range(n):
        country_rows.append({
            "country": names[i],
            "region": reg,
            "bmi_men": float(np.round(male[i], 1)),
            "bmi_women": float(np.round(female[i], 1)),
        })

df = pd.DataFrame(country_rows)

# Intentional annotatable "outliers" (synthetic)
extras = pd.DataFrame([
    {"country": "Gulf-terra",   "region": "Middle East", "bmi_men": 34.8, "bmi_women": 33.6},
    {"country": "Island-mark",  "region": "Oceania",     "bmi_men": 33.8, "bmi_women": 31.0},
    {"country": "Coastal-nia",  "region": "Africa",      "bmi_men": 21.0, "bmi_women": 28.8},
    {"country": "Metro-ia",     "region": "Europe",      "bmi_men": 25.1, "bmi_women": 19.6},
    {"country": "Riceford",     "region": "East Asia",   "bmi_men": 20.4, "bmi_women": 20.0},
])
df = pd.concat([df, extras], ignore_index=True)

script_dir = Path(__file__).parent
csv_path = script_dir.parent / "data" / "bmi_men_women_synthetic.csv"
df.to_csv(csv_path, index=False)

# ------------------------------
# Plot: Two-panel figure (A: scatter only, B: annotated)
# ------------------------------
# Scatter colored by region (custom palette)
reg_cat = pd.Categorical(df["region"], categories=regions, ordered=True)
codes = reg_cat.codes
# Map codes to colors from palette
scatter_colors = [COLORS[i % len(COLORS)] for i in codes]

# Legend (regions) - create once for both panels
handles, labels = [], []
for i, r in enumerate(regions):
    handles.append(Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=COLORS[i % len(COLORS)], markeredgecolor="none",
                          markersize=6, alpha=0.9))
    labels.append(r)

# Create two-panel figure
fig = plt.figure(figsize=(14.8, 6.6), dpi=220)

# Panel A: Scatter only (before annotation)
ax1 = fig.add_axes([0.06, 0.12, 0.40, 0.78])
ax1.scatter(df["bmi_men"], df["bmi_women"], s=22, alpha=0.75, c=scatter_colors, zorder=3)
ax1.set_xlim(18, 36)
ax1.set_ylim(18, 36)
ax1.set_xlabel("Men: average BMI")
ax1.set_ylabel("Women: average BMI")
ax1.set_title("Men vs women BMI by country", fontsize=11)
ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
ax1.legend(handles, labels, title="Region", frameon=True, fontsize=6, title_fontsize=7,
          loc="lower right", bbox_to_anchor=(0.98, 0.05))

# Panel B: Annotated version (after annotation)
ax2 = fig.add_axes([0.54, 0.12, 0.40, 0.78])

# BMI cutoffs: 25, 30
x0, x1, x2, x3 = 18, 25, 30, 36
y0, y1, y2, y3 = 18, 25, 30, 36

# 3x3 regime blocks (soft fill like your reference)
fills = {
    (0,0): (0.75, 0.95, 0.75, 0.35),  # Green area - more saturated and visible
    (1,0): (0.95, 0.95, 0.85, 0.22),
    (2,0): (0.98, 0.92, 0.80, 0.22),
    (0,1): (0.95, 0.95, 0.85, 0.22),
    (1,1): (0.98, 0.93, 0.80, 0.22),
    (2,1): (0.98, 0.88, 0.75, 0.22),
    (0,2): (0.98, 0.92, 0.80, 0.22),
    (1,2): (0.98, 0.88, 0.75, 0.22),
    (2,2): (0.95, 0.70, 0.65, 0.35),  # Red area - more saturated and visible
}
xs = [(x0, x1), (x1, x2), (x2, x3)]
ys = [(y0, y1), (y1, y2), (y2, y3)]
for ix, (xa, xb) in enumerate(xs):
    for iy, (ya, yb) in enumerate(ys):
        ax2.add_patch(Rectangle((xa, ya), xb-xa, yb-ya,
                               facecolor=fills[(ix,iy)], edgecolor="none", zorder=0))

# Threshold lines
for v in [25, 30]:
    ax2.axvline(v, linestyle="--", linewidth=0.6, alpha=0.3, zorder=1, color="gray")
    ax2.axhline(v, linestyle="--", linewidth=0.6, alpha=0.3, zorder=1, color="gray")

# x=y line
g = np.linspace(18, 36, 200)
ax2.plot(g, g, linewidth=1.6, alpha=0.85, zorder=2)

# Scatter colored by region
ax2.scatter(df["bmi_men"], df["bmi_women"], s=22, alpha=0.75, c=scatter_colors, zorder=3)

ax2.set_xlim(18, 36)
ax2.set_ylim(18, 36)
ax2.set_xlabel("Men: average BMI")
ax2.set_ylabel("Women: average BMI")
ax2.set_title("Men vs women BMI by country\n(x=y line + BMI categories as background)", fontsize=11)
ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)

# Regime labels on left side
ax2.text(18.2, 33.2, "OBESE", fontsize=9, alpha=0.8)
ax2.text(18.2, 27.6, "OVERWEIGHT", fontsize=9, alpha=0.8)
ax2.text(18.2, 21.8, "NORMAL\nWEIGHT", fontsize=9, alpha=0.8, va="center")

# Legend (regions)
ax2.legend(handles, labels, title="Region", frameon=True, fontsize=6, title_fontsize=7,
          loc="lower right", bbox_to_anchor=(0.98, 0.05))

# Annotate outliers
to_annotate = ["Gulf-terra", "Coastal-nia", "Metro-ia", "Riceford"]
for name in to_annotate:
    r = df[df["country"] == name].iloc[0]
    # Get the color for this point's region
    region_idx = regions.index(r["region"])
    point_color = COLORS[region_idx % len(COLORS)]
    ax2.scatter([r["bmi_men"]], [r["bmi_women"]], s=60, alpha=0.95, c=point_color, zorder=4)
    # Shorter arrow for Gulf-terra
    if name == "Gulf-terra":
        xytext_x = r["bmi_men"] + (0.4 if r["bmi_men"] < 30 else -1.5)
        xytext_y = r["bmi_women"] + (0.4 if r["bmi_women"] < 30 else -1.0)
    elif name == "Riceford":
        # Place Riceford label at bottom
        xytext_x = r["bmi_men"]
        xytext_y = r["bmi_women"] - 1.5
    else:
        xytext_x = r["bmi_men"] + (0.7 if r["bmi_men"] < 30 else -3.2)
        xytext_y = r["bmi_women"] + (0.8 if r["bmi_women"] < 30 else -2.0)
    ax2.annotate(
        name,
        xy=(r["bmi_men"], r["bmi_women"]),
        xytext=(xytext_x, xytext_y),
        arrowprops=dict(arrowstyle="->", lw=0.9, alpha=0.8),
        fontsize=8,
        zorder=5
    )

output_path = Path("../images/figure_14.png")
fig.savefig(output_path, facecolor="white", dpi=220)
plt.close(fig)

print("Saved:", csv_path, output_path)

