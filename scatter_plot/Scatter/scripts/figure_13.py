import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys

# Import theme colors from parent directory
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent.parent / "scatter plot"))
from theme_color import COLOR_DICT

# Set font to Inter, with fallback to sans-serif if not available
plt.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

# Load the dataset
data_path = script_dir.parent.parent / "scatter plot" / "data" / "support_cost_scatter_dataset.csv"
df = pd.read_csv(data_path)

avg = df["support_cost_per_ticket_usd"].mean()
mask_above_avg = df["support_cost_per_ticket_usd"] > avg

# Create figure with square subplots
fig = plt.figure(figsize=(12, 5), dpi=220)

# Calculate aspect ratio for square plotting area
# aspect = (y_range / x_range) * (figure_width / figure_height) * (axes_width / axes_height)
# For square axes (width = height), this simplifies
x_range = 200 - 50  # 150
y_range = 3.5 - 0.5  # 3.0
fig_width, fig_height = 12, 5
panel_size = 0.32  # Size of each square panel
aspect_ratio = (y_range / x_range) * (fig_width / fig_height)

# Position axes to be square: [left, bottom, width, height] in figure coordinates
# Each panel should be square, so width = height
left_margin = 0.08
bottom_margin = 0.15
spacing = 0.15  # Spacing to prevent y-axis overlap

ax1 = fig.add_axes([left_margin, bottom_margin, panel_size, panel_size])
ax2 = fig.add_axes([left_margin + panel_size + spacing, bottom_margin, panel_size, panel_size])

# First panel: All points same color, no reference line, no annotation
ax1.scatter(df["tickets_per_week"], df["support_cost_per_ticket_usd"], 
           alpha=0.75, s=45, color=COLOR_DICT["Dark Blue"])
ax1.set_xlabel("Tickets handled per week")
ax1.set_ylabel("Support cost per ticket (USD)")
ax1.set_xlim(50, 200)
ax1.set_ylim(0.5, 3.5)
# Lock axes position to keep them square - axes are already square (width = height)
# The plotting area will be square because the axes box is square
ax1.set_position([left_margin, bottom_margin, panel_size, panel_size])
ax1.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

# Second panel: Colored points with reference line and annotation
ax2.scatter(df.loc[~mask_above_avg, "tickets_per_week"], df.loc[~mask_above_avg, "support_cost_per_ticket_usd"], 
           alpha=0.75, s=45, color=COLOR_DICT["Dark Blue"], label='Below average')
ax2.scatter(df.loc[mask_above_avg, "tickets_per_week"], df.loc[mask_above_avg, "support_cost_per_ticket_usd"], 
           alpha=0.9, s=55, color=COLOR_DICT["Crusta"], label='Above average')

# Subtle average line: medium gray, thin, dashed
ax2.axhline(avg, linestyle="--", linewidth=1.0, color='gray', alpha=0.5)
ax2.text(55, avg + 0.05, "AVG", fontsize=10, va="bottom", color='gray', alpha=0.6)

ax2.set_xlabel("Tickets handled per week")
# Remove ylabel from second panel to prevent overlap
ax2.set_xlim(50, 200)
ax2.set_ylim(0.5, 3.5)
# Lock axes position to keep them square - axes are already square (width = height)
# The plotting area will be square because the axes box is square
ax2.set_position([left_margin + panel_size + spacing, bottom_margin, panel_size, panel_size])
ax2.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

output_path = Path("../images/figure_13.png")
fig.savefig(output_path, facecolor="white", dpi=220)
plt.close(fig)

print("Saved:", output_path)

