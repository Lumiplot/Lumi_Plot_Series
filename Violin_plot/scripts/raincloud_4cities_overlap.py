import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde  # pip install scipy if needed

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
df = pd.read_csv("../data/delivery_wait_times.csv")

cities = ["Berlin", "London", "New York", "Singapore"]
methods = ["Bike", "Car"]
colors = {"Bike": "#1f77b4", "Car": "#ff7f0e"}  # blue / orange

# global x-range so all densities line up
x_min = df["wait_time"].min() - 5
x_max = df["wait_time"].max() + 5
xs = np.linspace(x_min, x_max, 400)

# -------------------------------------------------
# 2. Build raincloud plot with 4 rows (one per city)
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

group_spacing = 3.0        # vertical distance between cities
cloud_height = 1.0         # max height of each density "cloud"
gap = 0.5                  # equal spacing between cloud, boxplot, and raw dots

for row_idx, city in enumerate(cities[::-1]):  # top to bottom
    y0 = row_idx * group_spacing  # baseline for this city

    for j, method in enumerate(methods):
        color = colors[method]
        data = df[(df["city"] == city) & (df["method"] == method)]["wait_time"].to_numpy()

        # ---------- KDE "cloud" (full overlap like your example) ----------
        kde = gaussian_kde(data)
        dens = kde(xs)
        dens = dens / dens.max() * cloud_height  # normalize height

        # fill density above baseline (both methods share same baseline y0)
        ax.fill_between(xs, y0, y0 + dens, color=color, alpha=0.35, linewidth=0.0)
        ax.plot(xs, y0 + dens, color=color, lw=1.5)

        # ---------- Boxplot for this method ----------
        # Equal spacing: boxplot is one gap below cloud baseline
        box_y = y0 - gap + j * 0.3     # Bike ~ y0-gap, Car ~ y0-gap+0.3
        bp = ax.boxplot(
            data,
            positions=[box_y],
            vert=False,
            widths=0.3,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,  # hide outlier points
            showcaps=False     # hide whisker end caps
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.7)
        for median in bp["medians"]:
            median.set(color="white", linewidth=1.2)
        # Explicitly remove whisker end caps (the horizontal lines at the ends)
        for cap in bp["caps"]:
            cap.set_visible(False)

        # ---------- Jittered raw points ("rain") ----------
        # Equal spacing: raw dots are one gap below boxplot
        rain_y = y0 - 2*gap - j * 0.15    # two slightly separated bands
        y_points = np.random.normal(loc=rain_y, scale=0.06, size=len(data))
        ax.scatter(
            data,
            y_points,
            s=12,
            alpha=0.5,
            color=color,
            edgecolor="none"
        )

# -------------------------------------------------
# 3. Cosmetics
# -------------------------------------------------
# y-axis labels at the center of each city's band
yticks = [i * group_spacing for i in range(len(cities))]
ax.set_yticks(yticks)
ax.set_yticklabels(cities[::-1])

ax.set_xlabel("Wait time (minutes)")
ax.set_title("Raincloud Plot: Bike vs Car Wait Times in Four Cities")

ax.set_xlim(x_min, x_max)
ax.set_ylim(-2, (len(cities) - 1) * group_spacing + cloud_height + 1)

ax.grid(axis="x", alpha=0.2)

# Legend
legend_handles = [
    plt.Line2D([0], [0], color=colors["Bike"], lw=6, label="Bike"),
    plt.Line2D([0], [0], color=colors["Car"],  lw=6, label="Car"),
]
ax.legend(handles=legend_handles, loc="lower right")

plt.tight_layout()
plt.savefig("../images/raincloud_4cities_overlap.png", dpi=150, bbox_inches="tight")
print("Raincloud plot saved as '../images/raincloud_4cities_overlap.png'")
plt.show()

