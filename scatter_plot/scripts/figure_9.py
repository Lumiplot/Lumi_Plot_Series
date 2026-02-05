import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(31415)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# =============================
# Dataset 1: Summaries with X and Y error bars (calibration / measurement study)
# Story: New assay measured at 4 known concentrations.
# Points are MEANS across replicate runs.
# X error bars = prep tolerance (instrument/prep variation)
# Y error bars = 95% CI of the mean response
# =============================

levels = np.array([1.2, 2.8, 5.4, 10.3])  # "true" concentration
n_rep = np.array([10, 12, 12, 10])

true_resp = 1.8 + 0.55*levels + 0.35*np.log1p(levels)

rows = []
for lvl, nr, tr in zip(levels, n_rep, true_resp):
    x_rep = rng.normal(lvl, 0.10*lvl, size=nr)                # prep variation (x)
    y_rep = rng.normal(tr, 0.35 + 0.08*tr, size=nr)           # measurement noise (y)
    for xr, yr in zip(x_rep, y_rep):
        rows.append({"level_true": lvl, "level_prepared": xr, "response": yr})

cal_raw = pd.DataFrame(rows)

summ = (cal_raw.groupby("level_true", as_index=False)
              .agg(n=("response","size"),
                   x_mean=("level_prepared","mean"),
                   x_sd=("level_prepared","std"),
                   y_mean=("response","mean"),
                   y_sd=("response","std")))

# X error: instrument/prep tolerance proxy
summ["x_err"] = np.minimum(summ["x_sd"].fillna(0), 0.18*summ["x_mean"])

# Y error: 95% CI of mean
summ["y_sem"] = summ["y_sd"] / np.sqrt(summ["n"])
summ["y_ci95"] = 1.96 * summ["y_sem"]

cal_summary = summ[["level_true","x_mean","y_mean","x_err","y_ci95","n","y_sd","y_sem"]].copy()

# =============================
# Dataset 2: Time course with error bars for two groups (dose/rate)
# Story: Mean response over time for High vs Low dose.
# Points are MEANS across subjects, error bars are 95% CI.
# =============================

subjects_per_group = 18
times = np.array([0, 2, 4, 8, 12, 16], dtype=float)

def simulate_group(group_name, growth_scale, curvature, noise_base):
    subj_ids = [f"{group_name}_{i:02d}" for i in range(1, subjects_per_group+1)]
    data = []
    for sid in subj_ids:
        subj_shift = rng.normal(0, 8.0)
        subj_scale = rng.normal(1.0, 0.08)
        for t in times:
            mu = (growth_scale * (t**curvature)) * subj_scale + subj_shift
            sd = noise_base + 0.12*max(mu, 0)
            y = rng.normal(max(mu, 0), sd)
            data.append({"subject_id": sid, "group": group_name, "time": t, "response": max(y, 0)})
    return pd.DataFrame(data)

tc_low = simulate_group("Low",  growth_scale=3.8, curvature=1.25, noise_base=6.0)
tc_high = simulate_group("High", growth_scale=6.0, curvature=1.35, noise_base=6.5)
timecourse_raw = pd.concat([tc_low, tc_high], ignore_index=True)

summ2 = (timecourse_raw.groupby(["group","time"], as_index=False)
                    .agg(n=("response","size"),
                         mean=("response","mean"),
                         sd=("response","std")))
summ2["sem"] = summ2["sd"] / np.sqrt(summ2["n"])
summ2["ci95"] = 1.96 * summ2["sem"]

# Create combined figure with two panels
fig = plt.figure(figsize=(13.0, 5.0), dpi=220)

# Panel 1: X & Y error bars (summary points)
ax1 = fig.add_axes([0.08, 0.15, 0.40, 0.75])

ax1.errorbar(
    cal_summary["x_mean"], cal_summary["y_mean"],
    xerr=cal_summary["x_err"], yerr=cal_summary["y_ci95"],
    fmt="s", markersize=5,
    capsize=5,
    elinewidth=1.0,   # thin error bars
    linewidth=1.2,
    color=COLORS[0]
)

ax1.set_title("Assay calibration: mean response with X and Y error bars\nX error = prep tolerance, Y error = 95% CI of mean", fontsize=10)
ax1.set_xlabel("Prepared concentration (a.u.)")
ax1.set_ylabel("Measured response (a.u.)")
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 10)
ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

# Panel 2: lines + error bars (summary points), thin lines
ax2 = fig.add_axes([0.56, 0.15, 0.40, 0.75])

color_map = {"High": COLORS[1], "Low": COLORS[0]}
for grp in ["High", "Low"]:
    d = summ2[summ2["group"] == grp].sort_values("time")
    ax2.errorbar(
        d["time"], d["mean"],
        yerr=d["ci95"],
        fmt="o",
        capsize=4,
        elinewidth=1.0,
        linewidth=1.2,
        markersize=4,
        label=grp,
        color=color_map[grp]
    )
    ax2.plot(d["time"], d["mean"], linewidth=1.2, color=color_map[grp])

ax2.set_title("Response vs time: group means with 95% CI error bars\nPoints are summaries (not raw), so fewer points stay readable", fontsize=10)
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Response (a.u.)")
ax2.set_xlim(times.min()-0.5, times.max()+0.5)
ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
ax2.legend(frameon=True, fontsize=8, title="Dose", title_fontsize=8, loc="upper left")

# Save combined figure
output_path = Path("../images/figure_9.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, facecolor="white", dpi=220)
print(f"Figure 9 saved to {output_path.absolute()}")

plt.close(fig)

