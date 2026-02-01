import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(123)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# ---------- Top row data (same data in all 3 panels) ----------
n1 = 70
x1 = rng.uniform(0, 10, n1)
y1 = 0.75 * x1 + rng.normal(0, 1.2, n1) + 1.0
x1 = np.clip(x1, 0, 10)
y1 = np.clip(y1, 0, 10)

# Fit line (same for all top panels)
b1, a1 = np.polyfit(x1, y1, 1)
x1_line = np.linspace(0, 10, 200)
y1_line = a1 + b1 * x1_line

# ---------- Build figure with MANUALLY CONTROLLED top-panel shapes ----------
fig = plt.figure(figsize=(13.5, 3.8), dpi=220)

# Top row: make plot areas obviously different shapes
# axes coords: [left, bottom, width, height] in figure fraction
# Increased spacing between panels
ax_t1 = fig.add_axes([0.06, 0.20, 0.30, 0.60])  # wide + short
ax_t2 = fig.add_axes([0.40, 0.15, 0.18, 0.70])  # square-ish
ax_t3 = fig.add_axes([0.63, 0.10, 0.12, 0.80])  # tall + narrow

top_axes = [ax_t1, ax_t2, ax_t3]

for ax in top_axes:
    ax.scatter(x1, y1, s=14, alpha=0.75, color=COLORS[0], edgecolors='none')
    ax.plot(x1_line, y1_line, linewidth=2.6, color=COLORS[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

# Save the figure
output_path = Path("../images/figure_5.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, facecolor="white", dpi=220, bbox_inches="tight")
print(f"Figure 5 saved to {output_path.absolute()}")

plt.close(fig)

