import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tempfile
import os

rng = np.random.default_rng(11)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

def make_panel(kind: str, title: str, out_path: Path):
    if kind == "curve_up":
        x = np.linspace(1.2, 8.8, 18) + rng.normal(0, 0.25, 18)
        # Upward curve (convex)
        y = 0.14*(x**2) + 0.35*x + rng.normal(0, 0.8, size=x.size) + 0.6
        # Calculate y range and set appropriate limits with generous padding
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        y_padding = max(y_range * 0.2, 1.0)  # At least 20% padding or 1.0 unit
        y_lim_low = max(0, y_min - y_padding)
        y_lim_high = y_max + y_padding  # Remove the cap to show all points

    elif kind == "cluster":
        x1 = rng.normal(3.0, 0.55, 10)
        y1 = rng.normal(7.3, 0.55, 10)
        x2 = rng.normal(7.1, 0.55, 10)
        y2 = rng.normal(3.0, 0.55, 10)
        x1 = np.clip(x1, 0.7, 9.3)
        y1 = np.clip(y1, 0.7, 9.3)
        x2 = np.clip(x2, 0.7, 9.3)
        y2 = np.clip(y2, 0.7, 9.3)

    elif kind == "outlier":
        x = rng.uniform(1, 9, 16)
        y = 0.65 * x + rng.normal(0, 0.6, size=x.size) + 2.0
        # One very obvious outlier (far below the trend)
        x_outlier = 8.7
        y_outlier = 1.2

    else:
        raise ValueError("Unknown kind")

    fig = plt.figure(figsize=(2.35, 2.1), dpi=220)
    ax = fig.add_axes([0.18, 0.18, 0.75, 0.70])

    if kind == "curve_up":
        ax.scatter(x, y, s=11, color=COLORS[0])
        ax.set_xlim(0, 10)
        ax.set_ylim(y_lim_low, y_lim_high)
    elif kind == "cluster":
        ax.scatter(x1, y1, s=11, color=COLORS[0])
        ax.scatter(x2, y2, s=11, color=COLORS[1])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    elif kind == "outlier":
        ax.scatter(x, y, s=11, color=COLORS[0])
        ax.scatter([x_outlier], [y_outlier], s=11, color=COLORS[1])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    ax.set_xticks(np.arange(0, 11, 2))
    if kind == "curve_up":
        # Set y ticks based on dynamic y limits
        y_range = y_lim_high - y_lim_low
        y_tick_step = max(1, round(y_range / 5))
        y_ticks = np.arange(np.ceil(y_lim_low), np.floor(y_lim_high) + 1, y_tick_step)
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks(np.arange(0, 11, 2))
    
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    # Get actual axis limits for arrow positioning
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Origin point where axes should cross (exact same point for both)
    x_origin = xlim[0]
    y_origin = ylim[0]
    
    # Hide top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Hide bottom and left spines - we'll draw them manually to ensure they cross
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    # Draw both axes as lines first to ensure they cross at origin
    # X-axis: horizontal line from origin to right edge
    ax.plot([x_origin, xlim[1]], [y_origin, y_origin], 'k-', lw=1.0, clip_on=False)
    # Y-axis: vertical line from origin to top edge  
    ax.plot([x_origin, x_origin], [y_origin, ylim[1]], 'k-', lw=1.0, clip_on=False)
    
    # Add arrowheads at the ends (using small offsets so they don't overlap the line ends)
    # Calculate arrowhead size as a small fraction of axis range
    arrow_size_x = (xlim[1] - xlim[0]) * 0.02
    arrow_size_y = (ylim[1] - ylim[0]) * 0.02
    
    # X-axis arrowhead pointing right
    ax.annotate("", xy=(xlim[1], y_origin), 
                xytext=(xlim[1] - arrow_size_x, y_origin),
                arrowprops=dict(arrowstyle="->", lw=1.0), 
                clip_on=False)
    
    # Y-axis arrowhead pointing up
    ax.annotate("", xy=(x_origin, ylim[1]), 
                xytext=(x_origin, ylim[1] - arrow_size_y),
                arrowprops=dict(arrowstyle="->", lw=1.0), 
                clip_on=False)

    fig.suptitle(title, fontsize=9, y=0.98)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# Create temporary directory for panel images
with tempfile.TemporaryDirectory() as tmpdir:
    paths = [
        Path(tmpdir) / "panel_curve_up.png",
        Path(tmpdir) / "panel_cluster.png",
        Path(tmpdir) / "panel_outlier.png",
    ]

    make_panel("curve_up", "Upward curve", paths[0])
    make_panel("cluster", "Clustering", paths[1])
    make_panel("outlier", "Outlier", paths[2])

    # Stitch into one image
    imgs = [Image.open(p).convert("RGBA") for p in paths]
    spacing = 18
    w = sum(im.width for im in imgs) + spacing * (len(imgs) - 1)
    h = max(im.height for im in imgs)

    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    xoff = 0
    for im in imgs:
        yoff = (h - im.height) // 2
        canvas.paste(im, (xoff, yoff))
        xoff += im.width + spacing

    # Save final figure
    output_path = Path("../images/figure_2.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, "PNG")
    print(f"Figure 2 saved to {output_path.absolute()}")

