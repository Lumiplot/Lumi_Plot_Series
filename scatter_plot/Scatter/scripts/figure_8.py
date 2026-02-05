import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(7)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# ----------------------------
# Synthetic dataset story:
# Laptop battery life vs laptop weight, split by laptop class.
#
# Variables:
# - weight_kg (x)
# - battery_life_hr (y)
# - battery_capacity_wh (bubble size)
#
# Story: At the same weight, Ultrabooks tend to get longer battery life (more efficient)
# than Gaming laptops (power hungry), even though Gaming often has bigger batteries.
# ----------------------------
n = 123
laptop_class = rng.choice(["Ultrabook", "Gaming"], size=n, p=[0.55, 0.45])

# Weight (kg): ultrabooks lighter, gaming heavier
weight = rng.normal(1.55, 0.22, n)
weight += np.where(laptop_class == "Gaming",
                   rng.normal(0.65, 0.18, n),
                   rng.normal(-0.05, 0.12, n))
weight = np.clip(weight, 1.0, 3.6)

# Battery capacity (Wh): correlated with weight, gaming a bit higher
capacity = 30 + 18 * (weight - 1.0) + rng.normal(0, 6.5, n) + np.where(laptop_class == "Gaming", 10, 0)
capacity = np.clip(capacity, 35, 105)

# Battery life (hours): increases with capacity but class affects efficiency
eff = np.where(laptop_class == "Ultrabook",
               rng.normal(0.14, 0.015, n),
               rng.normal(0.10, 0.018, n))
battery_life = 1.2 + eff * capacity + rng.normal(0, 0.55, n)
battery_life = np.clip(battery_life, 3.0, 18.5)

df = pd.DataFrame({
    "model_id": [f"L{i:03d}" for i in range(1, n + 1)],
    "class": laptop_class,
    "weight_kg": np.round(weight, 2),
    "battery_capacity_wh": np.round(capacity, 0).astype(int),
    "battery_life_hr": np.round(battery_life, 1),
})

# Two "interesting" outliers (optional, for annotation in your tutorial)
extra = pd.DataFrame([
    {"model_id": "L_out_all_day_gaming", "class": "Gaming",   "weight_kg": 3.10, "battery_capacity_wh": 99, "battery_life_hr": 16.8},
    {"model_id": "L_out_bad_ultrabook",  "class": "Ultrabook","weight_kg": 1.35, "battery_capacity_wh": 55, "battery_life_hr": 4.2},
])
df = pd.concat([df, extra], ignore_index=True)

def style_axes(ax):
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

# Calculate x limits from actual data with small padding
x_data_min = df["weight_kg"].min()
x_data_max = df["weight_kg"].max()
x_range = x_data_max - x_data_min
xmin = x_data_min - 0.05 * x_range
xmax = x_data_max + 0.05 * x_range

# Calculate y limits from actual data with small padding
y_data_min = df["battery_life_hr"].min()
y_data_max = df["battery_life_hr"].max()
y_range = y_data_max - y_data_min
ymin = y_data_min - 0.05 * y_range
ymax = y_data_max + 0.05 * y_range

color_map = {"Ultrabook": COLORS[0], "Gaming": COLORS[1]}

# ----------------------------
# Bubble plots - combined with color different only panel
# ----------------------------
def bubble_size(cap_wh):
    cap_wh = np.asarray(cap_wh)
    return ((cap_wh - 30) / 2.2) ** 1.7  # gentle area scaling

fig = plt.figure(figsize=(12.5, 5.5), dpi=220)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.20)

# Panel 1: Color different only
ax1 = fig.add_subplot(gs[0, 0])
for label in ["Ultrabook", "Gaming"]:
    d = df[df["class"] == label]
    ax1.scatter(d["weight_kg"], d["battery_life_hr"], s=16, alpha=0.85, 
               label=label, color=color_map[label])
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_xlabel("Laptop weight (kg)")
ax1.set_ylabel("Battery life (hours)")
ax1.legend(frameon=False, ncols=2, loc="upper left", fontsize=8, handletextpad=0.4, columnspacing=1.0)
style_axes(ax1)
ax1.set_title("Color different only", fontsize=11, pad=8)

# Panel 2: Bubble plot
ax2 = fig.add_subplot(gs[0, 1])
for label in ["Ultrabook", "Gaming"]:
    d = df[df["class"] == label]
    ax2.scatter(
        d["weight_kg"], d["battery_life_hr"],
        s=bubble_size(d["battery_capacity_wh"]),
        alpha=0.80,
        color=color_map[label],
        label=label
    )

ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel("Weight (kg)")
ax2.set_ylabel("Battery life (hours)")
legend1 = ax2.legend(frameon=False, loc="upper left", fontsize=9)
style_axes(ax2)
ax2.set_title("Bubble plot: both groups in one panel", fontsize=11, pad=8)

# Bubble legend (capacity) - add as second legend
legend_caps = [40, 60, 80, 100]
handles2 = [plt.scatter([], [], s=bubble_size(c), alpha=0.7, color="gray") for c in legend_caps]
labels2 = [f"{c}" for c in legend_caps]
legend2 = ax2.legend(handles2, labels2, title="Battery (Wh)", frameon=False,
            loc="lower right", fontsize=8, title_fontsize=8)
# Add the first legend back
ax2.add_artist(legend1)

# Save the figure
output_path = Path("../images/figure_8.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, facecolor="white", dpi=220)
print(f"Figure 8 saved to {output_path.absolute()}")

plt.close(fig)

