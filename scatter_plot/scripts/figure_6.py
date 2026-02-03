import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(123)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# ---------- Bottom row data (log scale effects) ----------
n2 = 140
x2 = 10 ** rng.uniform(0.2, 5.0, n2)  # ~1.6 to 100k
true_y2 = 2.2 * (x2 ** 0.22)
y2 = true_y2 * np.exp(rng.normal(0, 0.25, n2))  # multiplicative noise
y2 = np.clip(y2, 0.2, None)

# Lines for each scaling choice
# Fit power law: y = a * x^b using log-log fit
lx = np.log10(x2)
ly = np.log10(y2)
b_power, a_power_log = np.polyfit(lx, ly, 1)  # log(y) = a_power_log + b_power*log(x)
a_power = 10 ** a_power_log  # Convert back: y = a_power * x^b_power

# Generate x values for plotting
x2_line = np.linspace(x2.min(), x2.max(), 300)
lx_line = np.linspace(lx.min(), lx.max(), 300)
x2_line_logx = 10 ** lx_line

# Bottom-left: power law fit on linear-linear axes
y2_line_power = a_power * (x2_line ** b_power)

# Bottom-middle: power law fit on log-X, linear-Y axes
y2_line_power_logx = a_power * (x2_line_logx ** b_power)

# Bottom-right: log-log fit (already correct for power law)
ly_line = a_power_log + b_power * lx_line
y2_line_loglog = 10 ** ly_line

# ---------- Build figure with three equal-size panels ----------
fig = plt.figure(figsize=(13.5, 3.8), dpi=220)

# Bottom row: three equal-size panels
ax_b1 = fig.add_axes([0.06, 0.20, 0.27, 0.60])
ax_b2 = fig.add_axes([0.38, 0.20, 0.27, 0.60])
ax_b3 = fig.add_axes([0.70, 0.20, 0.27, 0.60])

# Bottom-left: power law fit on linear-linear axes
ax_b1.scatter(x2, y2, s=14, alpha=0.70, color=COLORS[1], edgecolors='none')
ax_b1.plot(x2_line, y2_line_power, linewidth=2.3, color=COLORS[1])
ax_b1.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
ax_b1.set_title("Linear scale\n(points bunch on the left)", fontsize=10)
ax_b1.set_xlabel("X")
ax_b1.set_ylabel("Y")

# Bottom-middle: power law fit on log-X, linear-Y axes
ax_b2.scatter(x2, y2, s=14, alpha=0.70, color=COLORS[1], edgecolors='none')
ax_b2.plot(x2_line_logx, y2_line_power_logx, linewidth=2.3, color=COLORS[1])
ax_b2.set_xscale("log")
ax_b2.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
ax_b2.set_title("Log X\n(spread X evenly)", fontsize=10)
ax_b2.set_xlabel("X (log)")
ax_b2.set_ylabel("Y")

# Bottom-right: log X + log Y (power law fit)
ax_b3.scatter(x2, y2, s=14, alpha=0.70, color=COLORS[1], edgecolors='none')
ax_b3.plot(x2_line_logx, y2_line_loglog, linewidth=2.3, color=COLORS[1])
ax_b3.set_xscale("log")
ax_b3.set_yscale("log")
ax_b3.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
ax_b3.set_title("Log X + Log Y\n(power laws look linear)", fontsize=10)
ax_b3.set_xlabel("X (log)")
ax_b3.set_ylabel("Y (log)")

fig.suptitle(
    "Log scaling changes patterns",
    fontsize=12,
    y=0.95,
)

# Save the figure
output_path = Path("../images/figure_6.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, facecolor="white", dpi=220, bbox_inches="tight")
print(f"Figure 6 saved to {output_path.absolute()}")

plt.close(fig)

