"""
Generate conceptual dumbbell plot with annotations explaining components.

Outputs:
- conceptual_dumbbell_plot.png

Run:
python make_conceptual_dumbbell.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set font to Inter
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# Paths
SCRIPT_DIR = Path(__file__).parent  # scripts folder
OUT_DIR = SCRIPT_DIR.parent / "images"  # images folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Data for conceptual plot
# -----------------------------
categories = ['A', 'B', 'C', 'D']
# Each category has two values: [open_circle_value, filled_circle_value]
dumbbell_data = {
    'A': [4.0, 5.0],
    'B': [3.5, 4.5],
    'C': [1.5, 2.5],
    'D': [0.7, 2.8]
}

# -----------------------------
# Create figure
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Set up y-axis (categories)
y_positions = list(range(len(categories)))
ax.set_yticks(y_positions)
ax.set_yticklabels(categories)
# Set y-axis limits with padding, inverted so A is at top
ax.set_ylim(len(categories) - 0.5, -0.5)

# Set up x-axis (values)
ax.set_xlim(0, 6)
ax.set_xticks(range(7))  # 0, 1, 2, 3, 4, 5, 6
ax.set_xlabel("VALUE AXIS", fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel("CATEGORIES", fontsize=11, fontweight='bold', labelpad=10)

# Draw dumbbells
for i, cat in enumerate(categories):
    val1, val2 = dumbbell_data[cat]
    
    # Draw connecting line
    ax.plot([val1, val2], [i, i], 'k-', linewidth=2, zorder=1)
    
    # Draw open circle (hollow)
    ax.scatter([val1], [i], s=200, facecolors='white', edgecolors='black', 
               linewidths=2, zorder=3)
    
    # Draw filled circle
    ax.scatter([val2], [i], s=200, facecolors='black', edgecolors='black', 
               linewidths=2, zorder=3)


# Style axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.3, axis='x')
ax.set_axisbelow(True)

plt.tight_layout(pad=1.5)

# Save figure
output_path = OUT_DIR / "conceptual_dumbbell_plot.png"
fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

print(f"Done. Figure saved to: {output_path.resolve()}")
