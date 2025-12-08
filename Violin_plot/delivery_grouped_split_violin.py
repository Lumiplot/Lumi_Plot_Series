"""
Create grouped and split violin plots for delivery wait times dataset.

This script demonstrates two types of violin plots:
1. Grouped violin plots - side-by-side violins for Bike and Car within each city
2. Split violin plots - half violins showing Bike and Car within each city
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv("delivery_wait_times.csv")

# Set style
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Color palette (matching raincloud plots)
palette = {"Bike": "#1f77b4", "Car": "#ff7f0e"}  # blue for bike, orange for car

# -------------------- (1) Grouped Violin Plot -------------------
# Side-by-side violins for each method within each city
violin_parts1 = sns.violinplot(
    data=df,
    x="city",
    y="wait_time",
    hue="method",   # creates side-by-side violins per city
    split=False,    # grouped (not split)
    palette=palette,
    inner="box",  # show box plots inside violins
    bw_method="silverman",
    density_norm="area",  # fair shape comparison
    linewidth=0.5,
    inner_kws={
        "linewidth": 0.8,
        "color": "white",
        "alpha": 0.6,     # 0 = transparent, 1 = opaque
    },
    ax=axes[0]
)

# Set opacity for violins
for pc in violin_parts1.collections:
    pc.set_alpha(1.0)

axes[0].set_title("Grouped Violin Plot\n(Side-by-side violins)", 
                  fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel("Wait time (minutes)", fontsize=12)
axes[0].set_xlabel("City", fontsize=12)
axes[0].legend(title="Delivery Method", title_fontsize=11, fontsize=10, 
               loc='upper right', framealpha=0.9)

# -------------------- (2) Split Violin Plot -------------------
# Half violins showing both methods within each city
violin_parts2 = sns.violinplot(
    data=df,
    x="city",
    y="wait_time",
    hue="method",
    split=True,     # split violins (half violins)
    palette=palette,
    inner="quartile",  # show quartiles inside violins
    bw_method="silverman",
    density_norm="area",  # fair shape comparison
    linewidth=0.5,
    inner_kws={"linewidth": 0.5, "color": "white"},
    ax=axes[1]
)

# Set opacity for split violins
for pc in violin_parts2.collections:
    pc.set_alpha(1.0)

axes[1].set_title("Split Violin Plot\n(Half violins)", 
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel("Wait time (minutes)", fontsize=12)
axes[1].set_xlabel("City", fontsize=12)
axes[1].legend(title="Delivery Method", title_fontsize=11, fontsize=10, 
               loc='upper right', framealpha=0.9)

plt.suptitle("Delivery Wait Times: Grouped vs Split Violin Plots", 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # Increase horizontal spacing between plots

# Save the plot
plt.savefig("delivery_grouped_split_violin.png", dpi=150, bbox_inches="tight")
print("Plot saved as 'delivery_grouped_split_violin.png'")
plt.show()

