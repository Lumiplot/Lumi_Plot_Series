# Paired-plots gallery: (1) paired mean + SEM with subject lines, (2) paired boxplot + points + connectors
# No seaborn. Just numpy/pandas/matplotlib.
# Following theme from draft folder: colors, fonts, aspect ratio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
import os

# Set font to Inter for all plots (matching draft folder theme)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# -------------------------
# Theme Colors (matching draft folder)
# -------------------------
COLOR_2024 = "#292F36"  # Dark Gray / Charcoal
COLOR_2025 = "#A41F13"  # Reddish-Brown / Burnt Orange
COLOR_TARGET = "#8F7A6E"  # Medium Brown / Taupe
COLOR_LINE = "#E0DBD8"  # Light Gray / Beige - for connecting lines
COLOR_BG = "#FAF5F1"  # Off-White / Cream - for backgrounds

rng = np.random.default_rng(7)

# -----------------------------
# Load survey data
# -----------------------------
# Data is in the parent directory (slope_dumbbell folder)
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "survey_data.csv")
df_survey = pd.read_csv(data_path)

# Remove any rows with missing data
df_survey = df_survey.dropna()

# Rename columns for easier use
df_survey = df_survey.rename(columns={
    "2024 Q4": "before",
    "2025 Q4": "after"
})

# Use survey data for both plots
df_left = df_survey.copy()
df_right = df_survey.copy()


# -----------------------------
# Figure 5a: Paired lines + mean/SEM
# -----------------------------
def plot_paired_mean_sem(df, a="Bars", b="Dots", ylabel="mean ΔF/F0 (%)",
                         a_label=None, b_label=None, outfile="paired_mean_sem.png"):
    x0, x1 = 0, 1
    
    # Use provided labels or fall back to column names
    if a_label is None:
        a_label = a
    if b_label is None:
        b_label = b

    a_vals = df[a].to_numpy()
    b_vals = df[b].to_numpy()

    # summary stats
    mean_a = a_vals.mean()
    mean_b = b_vals.mean()
    sem_a = a_vals.std(ddof=1) / np.sqrt(len(a_vals))
    sem_b = b_vals.std(ddof=1) / np.sqrt(len(b_vals))

    # Aspect ratio matching draft folder (slightly wider than tall)
    fig, ax = plt.subplots(figsize=(3.64, 2.94))

    # subject lines (thin) - using theme COLOR_LINE
    for i in range(len(df)):
        ax.plot([x0, x1], [a_vals[i], b_vals[i]], color=COLOR_LINE, alpha=0.65, linewidth=0.5, zorder=1)

    # mean line (thicker) - 2024 in dark gray, 2025 in red
    ax.plot([x0, (x0+x1)/2], [mean_a, (mean_a+mean_b)/2], color=COLOR_2024, linewidth=1.5, zorder=2)
    ax.plot([(x0+x1)/2, x1], [(mean_a+mean_b)/2, mean_b], color=COLOR_2025, linewidth=1.5, zorder=2)

    # mean +/- SEM error bars - 2024 in dark gray, 2025 in red
    ax.errorbar([x0], [mean_a], yerr=[sem_a], fmt="o", color=COLOR_2024,
                capsize=6, markersize=4, linewidth=1.0, zorder=3)
    ax.errorbar([x1], [mean_b], yerr=[sem_b], fmt="o", color=COLOR_2025,
                capsize=6, markersize=4, linewidth=1.0, zorder=3)

    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels([a_label, b_label], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Employee satisfaction survey results", fontsize=12)
    
    # Clean look - matching draft folder style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_2024)
    ax.spines["bottom"].set_color(COLOR_2024)
    ax.tick_params(colors=COLOR_2024)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


plot_paired_mean_sem(df_left, a="before", b="after",
                     a_label="2024 Q4", b_label="2025 Q4",
                     ylabel="% of employees agree",
                     outfile=os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_5a.png"))


# -----------------------------
# Figure 5b: Paired boxplots + points + connectors
# -----------------------------
def plot_paired_boxpoints(df, a="OJ", b="WC",
                          a_label=None, b_label=None, outfile="paired_box_points_connectors.png"):
    x0, x1 = 0, 1
    
    # Use provided labels or fall back to column names
    if a_label is None:
        a_label = a
    if b_label is None:
        b_label = b
    
    a_vals = df[a].to_numpy()
    b_vals = df[b].to_numpy()

    # Aspect ratio matching draft folder
    fig, ax = plt.subplots(figsize=(4.62, 2.94))

    # jittered points - using theme colors (more subtle)
    # Use same jitter for both x positions so lines connect properly
    jitter = 0.06
    jitter_x0 = rng.normal(0, jitter, len(df))
    jitter_x1 = rng.normal(0, jitter, len(df))
    x0_jittered = np.full(len(df), x0) + jitter_x0
    x1_jittered = np.full(len(df), x1) + jitter_x1
    
    # grey connector lines - connecting the jittered dot positions
    for i in range(len(df)):
        ax.plot([x0_jittered[i], x1_jittered[i]], [a_vals[i], b_vals[i]], 
                color=COLOR_LINE, alpha=0.45, linewidth=1.0, zorder=1)

    # Plot jittered points
    ax.scatter(x0_jittered, a_vals,
               s=14, color=COLOR_2024, alpha=0.35, zorder=3, edgecolors='none')
    ax.scatter(x1_jittered, b_vals,
               s=14, color=COLOR_2025, alpha=0.35, zorder=3, edgecolors='none')

    # boxplots (outlined, no fill) - using theme colors
    bp = ax.boxplot([a_vals, b_vals],
                    positions=[x0, x1],
                    widths=0.45,
                    patch_artist=True,
                    showfliers=False)

    # style boxes - using theme colors
    colors_box = [COLOR_2024, COLOR_2025]
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor("none")
        patch.set_edgecolor(c)
        patch.set_linewidth(2)

    for med, c in zip(bp["medians"], colors_box):
        med.set_color(c)
        med.set_linewidth(2)

    # Whiskers: first 2 are for 2024 Q4 (black), last 2 are for 2025 Q4 (red)
    for i, whisk in enumerate(bp["whiskers"]):
        if i < 2:  # 2024 Q4
            whisk.set_color(COLOR_2024)
        else:  # 2025 Q4
            whisk.set_color(COLOR_2025)
        whisk.set_linewidth(1.5)

    # Caps: first 2 are for 2024 Q4 (black), last 2 are for 2025 Q4 (red)
    for i, cap in enumerate(bp["caps"]):
        if i < 2:  # 2024 Q4
            cap.set_color(COLOR_2024)
        else:  # 2025 Q4
            cap.set_color(COLOR_2025)
        cap.set_linewidth(1.5)

    ax.set_xticks([x0, x1])
    ax.set_xticklabels([a_label, b_label], fontsize=12)
    ax.set_ylabel("% of employees agree", fontsize=11)
    ax.set_title("Employee satisfaction survey results", fontsize=12)
    
    # Clean look - matching draft folder style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_2024)
    ax.spines["bottom"].set_color(COLOR_2024)
    ax.tick_params(colors=COLOR_2024)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


plot_paired_boxpoints(df_right, a="before", b="after",
                      a_label="2024 Q4", b_label="2025 Q4",
                      outfile=os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_5b.png"))


print("Saved: figure_5a.png, figure_5b.png")
