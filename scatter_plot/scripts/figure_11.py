import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

rng = np.random.default_rng(20251213)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# =========================
# New dataset with clearer, more obvious relationships
# Topic: "Wellbeing + Productivity study" (remote workers)
# =========================
n = 280
program = rng.choice(["Baseline", "Mindfulness", "Fitness"], size=n, p=[0.34, 0.33, 0.33])
age = rng.integers(22, 56, size=n)

# Latent factors (z-scores)
recovery = rng.normal(0, 1.0, size=n)
strain   = rng.normal(0, 1.0, size=n)
activity = rng.normal(0, 1.0, size=n)

# Program effects (stronger separation)
recovery += np.where(program == "Mindfulness", 0.6, 0.0) + np.where(program == "Fitness", 0.25, 0.0)
strain   += np.where(program == "Mindfulness", -0.6, 0.0) + np.where(program == "Fitness", -0.25, 0.0)
activity += np.where(program == "Fitness", 0.75, 0.0)

# Observables with intentionally strong correlations
sleep_hours = 7.1 + 0.75*recovery - 0.35*strain + rng.normal(0, 0.25, size=n)
sleep_hours = np.clip(sleep_hours, 4.5, 9.8)

stress_score = 55 + 9.5*strain - 7.5*recovery + 0.10*(age-35) + rng.normal(0, 3.0, size=n)
stress_score = np.clip(stress_score, 15, 95)

caffeine_mg = 220 + 75*strain - 55*recovery + rng.normal(0, 35, size=n)
caffeine_mg = np.clip(caffeine_mg, 0, 700)

steps = 8200 + 2300*activity - 550*strain + 280*recovery + rng.normal(0, 900, size=n)
steps = np.clip(steps, 1200, 20000)

hrv_ms = 62 + 10.5*recovery - 9.0*strain + 0.00012*(steps-8000) - 0.22*(age-35) + rng.normal(0, 3.2, size=n)
hrv_ms = np.clip(hrv_ms, 18, 130)

resting_hr_bpm = 68 + 4.8*strain - 3.8*recovery - 0.00028*(steps-8000) + 0.09*(age-35) + rng.normal(0, 1.8, size=n)
resting_hr_bpm = np.clip(resting_hr_bpm, 45, 95)

caff_effect = 7.0 * (1 - np.exp(-caffeine_mg/220))
focus_score = (
    65
    + 6.5*(sleep_hours-7.0)
    + 0.22*(hrv_ms-60)
    + 0.00035*(steps-8000)
    - 0.34*(stress_score-55)
    - 0.18*(resting_hr_bpm-68)
    + 0.35*caff_effect
    + rng.normal(0, 3.0, size=n)
)
focus_score = np.clip(focus_score, 15, 100)

df = pd.DataFrame({
    "program": program,
    "age": age,
    "sleep_hours": np.round(sleep_hours, 2),
    "stress_score": np.round(stress_score, 1),
    "caffeine_mg": np.round(caffeine_mg, 0).astype(int),
    "steps": np.round(steps, 0).astype(int),
    "hrv_ms": np.round(hrv_ms, 1),
    "resting_hr_bpm": np.round(resting_hr_bpm, 1),
    "focus_score": np.round(focus_score, 1),
})

# =========================
# Combined figure with two panels
# =========================
vars_sm = ["sleep_hours", "stress_score", "caffeine_mg", "hrv_ms", "steps", "focus_score"]
X = df[vars_sm]
k = len(vars_sm)

# Create figure with two panels side by side
fig = plt.figure(figsize=(22, 10), dpi=220)

# =========================
# Panel 1: Scatterplot matrix with fit line + Pearson r
# =========================
# Position the scatterplot matrix in the left half
left_panel_left = 0.02
left_panel_bottom = 0.05
left_panel_width = 0.48
left_panel_height = 0.90

# Calculate subplot positions within the left panel
left, bottom, right, top = 0.07, 0.07, 0.98, 0.94
w = (right - left) / k
h = (top - bottom) / k

for i in range(k):
    for j in range(k):
        # Calculate absolute position within the figure
        abs_left = left_panel_left + left_panel_width * (left + j * w)
        abs_bottom = left_panel_bottom + left_panel_height * (bottom + (k-1-i) * h)
        abs_width = left_panel_width * w
        abs_height = left_panel_height * h
        
        ax_sub = fig.add_axes([abs_left, abs_bottom, abs_width, abs_height])
        ax_sub.set_facecolor("white")

        if i == j:
            # KDE instead of histogram
            data = X.iloc[:, j].to_numpy(dtype=float)
            kde = gaussian_kde(data)
            x_min, x_max = data.min(), data.max()
            x_range = x_max - x_min
            x_grid = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
            density = kde(x_grid)
            ax_sub.fill_between(x_grid, 0, density, alpha=0.15, color=COLORS[0])
            ax_sub.plot(x_grid, density, linewidth=1.2, alpha=0.9, color=COLORS[0])
        else:
            x = X.iloc[:, j].to_numpy(dtype=float)
            y = X.iloc[:, i].to_numpy(dtype=float)
            ax_sub.scatter(x, y, s=8, alpha=0.18, linewidths=0)

            # Fit line
            m, b = np.polyfit(x, y, 1)
            xg = np.linspace(np.min(x), np.max(x), 100)
            ax_sub.plot(xg, m*xg + b, linewidth=1.0, color=COLORS[0], alpha=0.6)

            # Pearson r
            r = np.corrcoef(x, y)[0, 1]
            ax_sub.text(0.05, 0.90, f"r = {r:+.2f}", transform=ax_sub.transAxes, fontsize=8)

        if i < k - 1:
            ax_sub.set_xticklabels([])
        else:
            ax_sub.set_xlabel(vars_sm[j], fontsize=8, rotation=45, ha="right")

        if j > 0:
            ax_sub.set_yticklabels([])
        else:
            ax_sub.set_ylabel(vars_sm[i], fontsize=8)

        ax_sub.grid(True, linestyle=":", linewidth=0.4, alpha=0.25)
        for sp in ["top", "right"]:
            ax_sub.spines[sp].set_visible(False)

# Add title for panel 1
fig.text(0.26, 0.96, "Scatterplot matrix with fit line + r", fontsize=12, ha="center", weight="bold")

# =========================
# Panel 2: Correlogram (upper triangle = values, lower triangle = bubbles)
# =========================
ax2 = fig.add_axes([0.52, 0.15, 0.42, 0.70])
ax2.set_facecolor("white")

corr = X.corr(numeric_only=True).to_numpy()

ax2.set_xlim(-0.5, k-0.5)
ax2.set_ylim(-0.5, k-0.5)
ax2.invert_yaxis()

ax2.set_xticks(range(k))
ax2.set_yticks(range(k))
ax2.set_xticklabels(vars_sm, rotation=45, ha="right", fontsize=8)
ax2.set_yticklabels(vars_sm, fontsize=8)

# Create diverging colormap from palette (Dark Blue -> white -> Crusta)
cmap = LinearSegmentedColormap.from_list("diverging", [COLORS[0], "#FFFFFF", COLORS[1]])

for i in range(k):
    for j in range(k):
        r = corr[i, j]

        # cell border
        ax2.plot([j-0.5, j+0.5], [i-0.5, i-0.5], linewidth=0.5, alpha=0.3)
        ax2.plot([j-0.5, j+0.5], [i+0.5, i+0.5], linewidth=0.5, alpha=0.3)
        ax2.plot([j-0.5, j-0.5], [i-0.5, i+0.5], linewidth=0.5, alpha=0.3)
        ax2.plot([j+0.5, j+0.5], [i-0.5, i+0.5], linewidth=0.5, alpha=0.3)

        if i < j:
            ax2.text(j, i, f"{r:+.2f}", ha="center", va="center", fontsize=8,
                    alpha=0.95 if abs(r) > 0.25 else 0.55)
        elif i > j:
            radius = 0.28 * abs(r)
            color = cmap((r + 1) / 2)
            ax2.add_patch(Circle((j, i), radius=radius, facecolor=color, edgecolor="none", alpha=0.95))
        else:
            ax2.text(j, i, f"{r:+.2f}", ha="center", va="center", fontsize=8)

# colorbar
cax = fig.add_axes([0.95, 0.22, 0.015, 0.50])
norm = plt.Normalize(-1, 1)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label("Pearson r", fontsize=9)

ax2.set_title("Correlation matrix\n(lower: bubbles, upper: values)", fontsize=11, pad=10)

for sp in ["top", "right", "left", "bottom"]:
    ax2.spines[sp].set_visible(False)

# Save combined figure
output_path = Path("../images/figure_11.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, facecolor="white", dpi=220, bbox_inches="tight")
print(f"Figure 11 saved to {output_path.absolute()}")

plt.close(fig)

