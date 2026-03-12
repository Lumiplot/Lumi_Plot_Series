from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _setup_inter_font():
    """Register Inter font if available in common paths."""
    font_dirs = [
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path(__file__).parent / "fonts",
        Path("/opt/homebrew/share/fonts"),
        Path("/usr/share/fonts"),
    ]
    for d in font_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("**/*[Ii]nter*.ttf")):
            if "Italic" in f.stem or "Bold" in f.stem:
                continue
            try:
                fm.fontManager.addfont(str(f))
                return
            except Exception:
                pass


_setup_inter_font()

# --------------------------------------------------
# Output (Lumi_Plot_Series format: scripts/ -> ../images/)
# --------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Global config
# --------------------------------------------------
WEEK_ORDER = [f"W{i}" for i in range(1, 9)]
TOP_APPS = ["NovaAI", "TaskPilot", "ChatFlow", "MemoSpark"]
HIGHLIGHT_APPS = ["NovaAI", "ChatFlow", "MemoSpark", "EchoWrite"]

# Color palette: dark blue, red, orange, goldenrod, cream
PALETTE = ["#003049", "#D62828", "#F77F00", "#FCBF49", "#EAE2B7"]

# 16:9 aspect ratio for single-panel figures (width, height)
ASPECT_16_9 = (9, 9 / 16 * 9)  # (9, 5.0625)

# Typography and line scaling
FONT_BASE = 13
FONT_SMALL = 11
FONT_MED = 12
FONT_LARGE = 14
LW_HIGHLIGHT = 3.5
LW_OTHER = 1.8
LW_VALUE = 3.0
LW_ANNOTATE = 1.6

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "font.size": FONT_BASE,
    "axes.titlesize": FONT_LARGE,
    "axes.labelsize": FONT_BASE,
    "xtick.labelsize": FONT_SMALL,
    "ytick.labelsize": FONT_SMALL,
    "legend.fontsize": FONT_SMALL,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Inter Tight", "DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

# --------------------------------------------------
# Base dataset
# --------------------------------------------------
CSV = """week,app,downloads_k
W1,NovaAI,920
W1,TaskPilot,760
W1,ChatFlow,700
W1,MemoSpark,410
W1,PixelNote,380
W1,StudyMate,340
W1,CalmScript,300
W1,TrendLens,280
W1,ClipMind,260
W1,DataMuse,220
W1,PromptPad,200
W1,EchoWrite,
W2,NovaAI,860
W2,TaskPilot,730
W2,ChatFlow,720
W2,MemoSpark,430
W2,PixelNote,390
W2,StudyMate,330
W2,CalmScript,310
W2,TrendLens,290
W2,ClipMind,250
W2,DataMuse,215
W2,PromptPad,195
W2,EchoWrite,
W3,NovaAI,820
W3,TaskPilot,690
W3,ChatFlow,740
W3,MemoSpark,470
W3,PixelNote,400
W3,StudyMate,320
W3,CalmScript,315
W3,TrendLens,305
W3,ClipMind,245
W3,DataMuse,210
W3,PromptPad,180
W3,EchoWrite,
W4,NovaAI,780
W4,TaskPilot,640
W4,ChatFlow,735
W4,MemoSpark,530
W4,PixelNote,410
W4,StudyMate,300
W4,CalmScript,290
W4,TrendLens,320
W4,ClipMind,230
W4,DataMuse,205
W4,PromptPad,170
W4,EchoWrite,120
W5,NovaAI,750
W5,TaskPilot,590
W5,ChatFlow,720
W5,MemoSpark,580
W5,PixelNote,420
W5,StudyMate,250
W5,CalmScript,250
W5,TrendLens,340
W5,ClipMind,210
W5,DataMuse,200
W5,PromptPad,160
W5,EchoWrite,220
W6,NovaAI,720
W6,TaskPilot,550
W6,ChatFlow,700
W6,MemoSpark,620
W6,PixelNote,430
W6,StudyMate,240
W6,CalmScript,255
W6,TrendLens,360
W6,ClipMind,190
W6,DataMuse,195
W6,PromptPad,
W6,EchoWrite,330
W7,NovaAI,700
W7,TaskPilot,510
W7,ChatFlow,680
W7,MemoSpark,640
W7,PixelNote,425
W7,StudyMate,235
W7,CalmScript,260
W7,TrendLens,380
W7,ClipMind,170
W7,DataMuse,190
W7,PromptPad,
W7,EchoWrite,450
W8,NovaAI,680
W8,TaskPilot,470
W8,ChatFlow,650
W8,MemoSpark,660
W8,PixelNote,420
W8,StudyMate,230
W8,CalmScript,250
W8,TrendLens,410
W8,ClipMind,150
W8,DataMuse,185
W8,PromptPad,
W8,EchoWrite,560
"""

# --------------------------------------------------
# Data prep
# --------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(CSV))
    df["downloads_k"] = pd.to_numeric(df["downloads_k"], errors="coerce")
    df["week"] = pd.Categorical(df["week"], categories=WEEK_ORDER, ordered=True)
    df = df.sort_values(["week", "app"]).reset_index(drop=True)

    # Shared-rank version
    df["rank_tied"] = (
        df.groupby("week", observed=False)["downloads_k"]
        .rank(ascending=False, method="min")
    )

    # Unique-rank version for clean plots
    df["rank_unique"] = np.nan
    for _, sub in df.groupby("week", sort=False, observed=False):
        valid = sub[sub["downloads_k"].notna()].sort_values(
            ["downloads_k", "app"], ascending=[False, True]
        )
        df.loc[valid.index, "rank_unique"] = np.arange(1, len(valid) + 1, dtype=float)

    return df


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def save(fig, name: str):
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def app_color(app: str, highlight=True):
    mapping = {
        "NovaAI": PALETTE[0],
        "TaskPilot": PALETTE[1],
        "ChatFlow": PALETTE[2],
        "MemoSpark": PALETTE[3],
        "EchoWrite": "#000000",
        "TrendLens": PALETTE[0],
        "PixelNote": PALETTE[1],
        "StudyMate": PALETTE[2],
        "CalmScript": PALETTE[3],
        "ClipMind": PALETTE[4],
        "DataMuse": PALETTE[0],
        "PromptPad": PALETTE[1],
        "A": PALETTE[0],
        "B": PALETTE[1],
        "C": PALETTE[2],
        "D": PALETTE[3],
        "E": PALETTE[4],
        "F": PALETTE[0],
        "G": PALETTE[1],
        "Riser": PALETTE[2],
    }
    return mapping.get(app, PALETTE[0]) if highlight else "0.75"


def smooth_series(x, y, points_per_segment=30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return x, y

    xs, ys = [], []
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        t = np.linspace(0, 1, points_per_segment, endpoint=False)
        te = (1 - np.cos(np.pi * t)) / 2
        xs.append(x0 + (x1 - x0) * t)
        ys.append(y0 + (y1 - y0) * te)
    xs.append(np.array([x[-1]]))
    ys.append(np.array([y[-1]]))
    return np.concatenate(xs), np.concatenate(ys)


def plot_bump(
    ax,
    df,
    apps=None,
    rank_col="rank_unique",
    title=None,
    direct_labels=False,
    labels_end_only=False,
    label_fontsize=None,
    label_above_apps=None,
    highlight_apps=None,
    invert=True,
    markers=True,
    legend=False,
    jitter_ties=False,
    smooth=False,
    alpha_other=0.60,
    lw_hi=LW_HIGHLIGHT,
    lw_other=LW_OTHER,
):
    if apps is None:
        apps = df["app"].dropna().unique().tolist()
    if highlight_apps is None:
        highlight_apps = []
    if label_fontsize is None:
        label_fontsize = FONT_MED
    if label_above_apps is None:
        label_above_apps = []

    x_map = {w: i for i, w in enumerate(WEEK_ORDER)}

    for app in apps:
        sub = df[df["app"] == app].sort_values("week")
        x = np.array([x_map[w] for w in sub["week"]], dtype=float)
        y = sub[rank_col].to_numpy(dtype=float)

        if jitter_ties and rank_col == "rank_tied":
            y = y + {"StudyMate": -0.12, "CalmScript": 0.12}.get(app, 0.0)

        is_hi = (app in highlight_apps) or (len(highlight_apps) == 0)
        color = app_color(app, is_hi)
        lw = lw_hi if is_hi else lw_other
        alpha = 1.0 if is_hi else alpha_other
        z = 3 if is_hi else 1

        if smooth:
            xs, ys = smooth_series(x, y)
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=z, label=app)
        else:
            ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, zorder=z, label=app)

        if markers:
            ax.plot(x, y, linestyle="None", marker="o", color=color, alpha=alpha, zorder=z + 1)

        if direct_labels:
            valid = sub[sub[rank_col].notna()]
            if not valid.empty:
                first = valid.iloc[0]
                last = valid.iloc[-1]
                if not labels_end_only:
                    ax.text(
                        x_map[first["week"]] - 0.08,
                        float(first[rank_col]),
                        f"{app}  ",
                        ha="right",
                        va="center",
                        fontsize=FONT_SMALL,
                        color=color if is_hi else "0.40",
                        alpha=alpha,
                    )
                last_x = x_map[last["week"]] + 0.08
                last_y = float(last[rank_col])
                if app in label_above_apps:
                    ax.text(last_x - 0.08, last_y - 0.12, app, ha="center", va="bottom",
                            fontsize=label_fontsize, color=color if is_hi else "0.40", alpha=alpha)
                else:
                    ax.text(last_x, last_y, f"  {app}", ha="left", va="center",
                            fontsize=label_fontsize, color=color if is_hi else "0.40", alpha=alpha)

    ax.set_xticks(range(len(WEEK_ORDER)))
    ax.set_xticklabels(WEEK_ORDER)
    ax.set_xlabel("Week")
    ax.set_ylabel("Rank")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    ymax = int(np.nanmax(df[rank_col])) if df[rank_col].notna().any() else 1
    ax.set_ylim(0.5, ymax + 0.8)
    if invert:
        ax.invert_yaxis()

    if title:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")


def plot_value_lines(ax, df, apps, title=None, direct_labels=False, legend=False, legend_inside=False, label_fontsize=None, legend_fontsize=None):
    if label_fontsize is None:
        label_fontsize = FONT_MED
    x_map = {w: i for i, w in enumerate(WEEK_ORDER)}

    for app in apps:
        sub = df[df["app"] == app].sort_values("week")
        x = np.array([x_map[w] for w in sub["week"]], dtype=float)
        y = sub["downloads_k"].to_numpy(dtype=float)
        color = app_color(app, True)

        ax.plot(x, y, marker="o", linewidth=LW_VALUE, label=app, color=color)

        if direct_labels:
            valid = sub[sub["downloads_k"].notna()]
            if not valid.empty:
                last = valid.iloc[-1]
                ax.text(
                    x_map[last["week"]] + 0.08,
                    float(last["downloads_k"]),
                    f"  {app}",
                    ha="left",
                    va="center",
                    fontsize=label_fontsize,
                    color=color,
                )

    ax.set_xticks(range(len(WEEK_ORDER)))
    ax.set_xticklabels(WEEK_ORDER)
    ax.set_xlabel("Week")
    ax.set_ylabel("Value")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    if title:
        ax.set_title(title)
    if legend:
        kwargs = {"frameon": False}
        if legend_fontsize is not None:
            kwargs["fontsize"] = legend_fontsize
        if legend_inside:
            ax.legend(loc="upper right", **kwargs)
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", **kwargs)


def annotate(ax, text, xy, xytext):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        arrowprops=dict(arrowstyle="->", lw=LW_ANNOTATE),
        fontsize=FONT_MED,
    )


# --------------------------------------------------
# Figures
# --------------------------------------------------
def figure_01_line_chart_hides_race(df):
    apps = TOP_APPS + ["PixelNote"]
    fig, ax = plt.subplots(figsize=ASPECT_16_9)
    plot_value_lines(ax, df, apps, title="Standard value lines can hide the race", legend=True, legend_inside=True)
    save(fig, "figure_01_line_chart_hides_race.png")


def figure_02_definition_anatomy(df):
    apps = TOP_APPS
    fig, ax = plt.subplots(figsize=ASPECT_16_9)
    plot_bump(ax, df[df["app"].isin(apps)], apps=apps, direct_labels=True, labels_end_only=True, title="What a bump chart is")
    save(fig, "figure_02_definition_anatomy.png")


def figure_03_ordinal_not_quantitative(df):
    week = "W8"
    sub = df[(df["week"] == week) & (df["downloads_k"].notna())].sort_values("downloads_k", ascending=False).head(5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # ranks
    axes[0].hlines(sub["rank_unique"], 0, sub["downloads_k"], linewidth=0)  # no-op for scale
    colors = [app_color(a, True) for a in sub["app"]]
    axes[0].scatter(np.ones(len(sub)), sub["rank_unique"], s=70, c=colors)
    for _, row in sub.iterrows():
        axes[0].text(1.02, row["rank_unique"], row["app"], va="center", fontsize=FONT_SMALL)
    axes[0].set_xlim(0.9, 1.4)
    axes[0].set_ylim(0.5, 5.5)
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_ylabel("Rank")
    axes[0].set_title("Rank view")
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)

    # values
    bar_colors = [app_color(a, True) for a in sub["app"]]
    axes[1].barh(sub["app"], sub["downloads_k"], color=bar_colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Downloads")
    axes[1].set_title("Value view")
    fig.suptitle("Figure 3. Equal rank spacing does not mean equal value gaps", y=1.03)
    save(fig, "figure_03_ordinal_not_quantitative.png")


def figure_04_how_to_read(df):
    apps = ["NovaAI", "TaskPilot", "ChatFlow", "MemoSpark"]
    fig, ax = plt.subplots(figsize=ASPECT_16_9)
    plot_bump(ax, df[df["app"].isin(apps)], apps=apps, direct_labels=True, labels_end_only=True, title="How to read a bump chart")
    save(fig, "figure_04_how_to_read.png")


def figure_05_crossing_is_event(df):
    apps = ["TaskPilot", "ChatFlow", "MemoSpark"]
    fig, ax = plt.subplots(figsize=ASPECT_16_9)
    plot_bump(ax, df[df["app"].isin(apps)], apps=apps, direct_labels=True, labels_end_only=True, title="The crossing is the event")
    save(fig, "figure_05_crossing_is_event.png")


def figure_06_crossing_not_magnitude(df):
    apps = ["TaskPilot", "ChatFlow"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(wspace=0.35)
    plot_bump(axes[0], df[df["app"].isin(apps)], apps=apps, direct_labels=True, labels_end_only=True, title="Order changes")
    sub = df[df["app"].isin(apps)].copy()
    plot_value_lines(axes[1], sub, apps, title="Values near the crossing")
    fig.suptitle("Figure 6. What a crossing does not mean", y=1.03)
    save(fig, "figure_06_crossing_not_magnitude.png")


def figure_07_hidden_tradeoff(df):
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    plot_bump(
        ax1,
        df,
        apps=sorted(df["app"].unique()),
        direct_labels=True,
        labels_end_only=True,
        highlight_apps=["NovaAI", "ChatFlow"],
        title="Rank view"
    )

    plot_value_lines(ax2, df, ["NovaAI", "ChatFlow"], direct_labels=True, title="Value view")

    fig.suptitle("Figure 7. Stable rank can hide decline", y=0.98)
    save(fig, "figure_07_hidden_tradeoff.png")


def figure_08_history_timeline():
    fig, ax = plt.subplots(figsize=ASPECT_16_9)
    ax.axis("off")

    x1, x2 = 0.2, 0.8
    y = 0.5
    ax.plot([x1, x2], [y, y], linewidth=LW_VALUE, color=PALETTE[0])
    ax.scatter([x1, x2], [y, y], s=90, color=PALETTE[0])

    ax.text(x1, y + 0.12, "1815\nOxford bumps racing", ha="center", va="bottom", fontsize=FONT_BASE)
    ax.text(x1, y - 0.12, "Race by overtaking the boat ahead", ha="center", va="top", fontsize=FONT_MED)

    ax.text(x2, y + 0.12, "1900\nParis Exposition", ha="center", va="bottom", fontsize=FONT_BASE)
    ax.text(x2, y - 0.12, "Du Bois and collaborators use rank-like\ncomparative graphics", ha="center", va="top", fontsize=FONT_MED)

    ax.set_title("Figure 8. A short visual history")
    save(fig, "figure_08_history_timeline.png")


def figure_09_where_bump_shines(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    panels = [
        (["NovaAI", "TaskPilot", "ChatFlow", "MemoSpark"], "Sports standings"),
        (["ChatFlow", "MemoSpark", "TrendLens", "EchoWrite"], "App rankings"),
        (["NovaAI", "ChatFlow", "MemoSpark", "PixelNote"], "Market-share leaders"),
        (["NovaAI", "TaskPilot", "ChatFlow", "EchoWrite"], "Election race positions"),
        (["TrendLens", "PixelNote", "CalmScript", "StudyMate"], "Business units"),
        (["EchoWrite", "PromptPad", "ClipMind", "DataMuse"], "Late entrants and dropouts"),
    ]

    for ax, (apps, title) in zip(axes.flat, panels):
        plot_bump(ax, df[df["app"].isin(apps)], apps=apps, title=title)

    fig.suptitle("Figure 9. Where bump charts shine", y=1.02)
    save(fig, "figure_09_where_bump_shines.png")


def figure_10_when_to_skip(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # magnitude matters
    plot_bump(axes[0, 0], df[df["app"].isin(["NovaAI", "ChatFlow", "MemoSpark"])],
              apps=["NovaAI", "ChatFlow", "MemoSpark"], title="Magnitude matters")
    axes[0, 0].text(2.6, 3.8, "Use a value chart too", fontsize=FONT_MED)

    # too many lines
    plot_bump(axes[0, 1], df, apps=sorted(df["app"].unique()), title="Too many categories")

    # noisy ranks
    noisy = pd.DataFrame({
        "week": np.repeat(WEEK_ORDER, 3),
        "app": ["A", "B", "C"] * len(WEEK_ORDER),
        "downloads_k": [100, 99, 98, 100, 101, 99, 100, 99, 100, 100, 101, 99,
                        99, 100, 101, 100, 99, 100, 101, 100, 99, 100, 99, 100]
    })
    noisy["week"] = pd.Categorical(noisy["week"], categories=WEEK_ORDER, ordered=True)
    noisy["rank_unique"] = np.nan
    for _, sub in noisy.groupby("week", sort=False, observed=False):
        valid = sub.sort_values(["downloads_k", "app"], ascending=[False, True])
        noisy.loc[valid.index, "rank_unique"] = np.arange(1, len(valid) + 1, dtype=float)
    plot_bump(axes[1, 0], noisy, apps=["A", "B", "C"], title="Noisy data can fake drama")

    # ties
    plot_bump(
        axes[1, 1],
        df[df["app"].isin(["StudyMate", "CalmScript", "TrendLens"])],
        apps=["StudyMate", "CalmScript", "TrendLens"],
        rank_col="rank_tied",
        title="Ties need handling"
    )

    fig.suptitle("Figure 10. When to skip or qualify a bump chart", y=1.02)
    save(fig, "figure_10_when_to_skip.png")


def figure_11_line_vs_slope_vs_bump(df):
    apps = TOP_APPS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.subplots_adjust(wspace=0.55)
    plot_value_lines(axes[0], df, apps, legend=True, legend_inside=True, legend_fontsize=9, title="Line chart")

    ax = axes[1]
    for app in apps:
        sub = df[(df["app"] == app) & (df["week"].isin(["W1", "W8"]))].sort_values("week")
        x = np.array([0, 1])
        y = sub["downloads_k"].to_numpy(dtype=float)
        c = app_color(app, True)
        ax.plot(x, y, marker="o", linewidth=LW_VALUE, color=c, label=app)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["W1", "W8"])
    ax.set_ylabel("Value")
    ax.set_title("Slope chart")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    plot_bump(axes[2], df[df["app"].isin(apps)], apps=apps, direct_labels=True, labels_end_only=True, title="Bump chart")
    fig.suptitle("Figure 11. Bump vs line vs slope", y=1.03)
    save(fig, "figure_11_line_vs_slope_vs_bump.png")


def figure_12_field_size_caveat():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Rank 3 of 5
    axes[0].barh(range(5, 0, -1), [1] * 5, color=[PALETTE[i % len(PALETTE)] for i in range(5)])
    axes[0].invert_yaxis()
    axes[0].axhline(3, color=PALETTE[0], linewidth=2.5)
    axes[0].text(1.02, 3, "Rank 3 of 5", va="center")
    axes[0].set_title("Smaller field")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Rank 5 of 20
    axes[1].barh(range(20, 0, -1), [1] * 20, color=[PALETTE[i % len(PALETTE)] for i in range(20)])
    axes[1].invert_yaxis()
    axes[1].axhline(5, color=PALETTE[0], linewidth=2.5)
    axes[1].text(1.02, 5, "Rank 5 of 20", va="center")
    axes[1].set_title("Larger field")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle("Figure 12. Rank means different things in different field sizes", y=1.03)
    save(fig, "figure_12_field_size_caveat.png")


def figure_13_data_prep_pipeline():
    fig, ax = plt.subplots(figsize=ASPECT_16_9)
    ax.axis("off")

    steps = [
        "Raw values",
        "Rank by period",
        "Choose tie rule",
        "Choose missing-data rule",
        "Final plotting table"
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    y = 0.5

    for i, (x, label) in enumerate(zip(xs, steps)):
        rect = plt.Rectangle((x - 0.08, y - 0.13), 0.16, 0.26, fill=False, linewidth=1.8, edgecolor=PALETTE[i % len(PALETTE)])
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=FONT_BASE)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.09, y), xytext=(x + 0.09, y),
                        arrowprops=dict(arrowstyle="->", lw=LW_ANNOTATE, color=PALETTE[0]))
    ax.set_title("Figure 13. Data prep pipeline")
    save(fig, "figure_13_data_prep_pipeline.png")


def figure_14_tie_breaking_rules(df):
    base = df[df["app"].isin(["StudyMate", "CalmScript", "TrendLens"])].copy()
    # Recompute rank_tied within this 3-app subset (not full df)
    base["rank_tied"] = (
        base.groupby("week", observed=False)["downloads_k"]
        .rank(ascending=False, method="min")
    )

    alpha_df = base.copy()
    alpha_df["rank_alpha"] = np.nan
    for _, sub in alpha_df.groupby("week", sort=False, observed=False):
        valid = sub[sub["downloads_k"].notna()].sort_values(["downloads_k", "app"], ascending=[False, True])
        alpha_df.loc[valid.index, "rank_alpha"] = np.arange(1, len(valid) + 1, dtype=float)

    prev_df = base.copy()
    prev_df["rank_prev"] = np.nan
    prev_order = {"StudyMate": 1, "CalmScript": 2, "TrendLens": 3}
    for w, sub in prev_df.groupby("week", sort=False, observed=False):
        valid = sub[sub["downloads_k"].notna()].copy()
        valid["prev"] = valid["app"].map(prev_order)
        valid = valid.sort_values(["downloads_k", "prev"], ascending=[False, True])
        prev_df.loc[valid.index, "rank_prev"] = np.arange(1, len(valid) + 1, dtype=float)
        prev_order = {app: i + 1 for i, app in enumerate(valid["app"].tolist())}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    fig.subplots_adjust(wspace=0.55)
    apps_14 = ["StudyMate", "CalmScript", "TrendLens"]
    plot_bump(axes[0], base, apps=apps_14, rank_col="rank_tied", direct_labels=True, labels_end_only=True, title="Shared rank")
    plot_bump(axes[1], alpha_df, apps=apps_14, rank_col="rank_alpha", direct_labels=True, labels_end_only=True, title="Alphabetical tie-break")
    plot_bump(axes[2], prev_df, apps=apps_14, rank_col="rank_prev", direct_labels=True, labels_end_only=True, title="Previous-rank tie-break")
    for ax in axes:
        ax.set_ylim(0.5, 3.8)
        ax.set_yticks([1, 2, 3])
    fig.suptitle("Figure 14. Tie-breaking rules change the story", y=1.03)
    save(fig, "figure_14_tie_breaking_rules.png")


def figure_15_missing_and_time_spacing(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # missing periods: one series with a real internal gap (W4, W5 missing)
    # Riser: W1=8, W2=7, W3=6, W4-W5=missing, W6=5, W7=4, W8=4 — line breaks in middle
    i = lambda w: list(WEEK_ORDER).index(w)
    gap_data = []
    for w in WEEK_ORDER:
        idx = i(w)
        gap_data.append({"week": w, "app": "Riser", "rank_unique": [8, 7, 6, np.nan, np.nan, 5, 4, 4][idx]})
        for r, app in enumerate(["A", "B", "C", "D", "E", "F", "G"], 1):
            # Fill ranks 1-7; when Riser has 8, use 1-7; when Riser has 7, use 1-6,8; etc.
            if idx <= 2:  # W1-W3
                rank = r if r < [8, 7, 6][idx] else r + 1
            elif idx <= 4:  # W4-W5, Riser missing
                rank = r
            else:  # W6-W8
                rank = r if r < [5, 4, 4][idx - 5] else r + 1
            gap_data.append({"week": w, "app": app, "rank_unique": float(rank)})
    gap_df = pd.DataFrame(gap_data)
    gap_df["week"] = pd.Categorical(gap_df["week"], categories=WEEK_ORDER, ordered=True)
    plot_bump(
        axes[0],
        gap_df,
        apps=["Riser", "A", "B", "C", "D", "E", "F", "G"],
        title="Broken lines show absence honestly",
        direct_labels=True,
        labels_end_only=True,
        highlight_apps=["Riser"],
    )

    # uneven spacing schematic
    ax = axes[1]
    x_real = [0, 1, 4, 7]
    y = [1, 2, 2, 3]
    ax.plot(x_real, y, marker="o", linewidth=LW_VALUE, color=PALETTE[0])
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 3])
    ax.set_ylabel("Rank")
    ax.set_xlabel("Time")
    ax.set_xticks(x_real)
    ax.set_xticklabels(["Jan", "Feb", "Jun", "Dec"])
    ax.set_title("Uneven intervals should look uneven")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("Figure 15. Missing periods and honest time spacing", y=1.03)
    save(fig, "figure_15_missing_and_time_spacing.png")


def figure_16_design_before_after(df):
    apps = sorted(df["app"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))

    plot_bump(axes[0], df, apps=apps, title="Before")
    plot_bump(
        axes[1],
        df,
        apps=apps,
        direct_labels=True,
        labels_end_only=True,
        highlight_apps=HIGHLIGHT_APPS,
        title="After"
    )
    fig.suptitle("Figure 16. Design system: before vs after", y=1.02)
    save(fig, "figure_16_design_before_after.png")


def figure_17_labels_vs_legends(df):
    apps = HIGHLIGHT_APPS
    sub = df[df["app"].isin(apps)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    plot_bump(axes[0], sub, apps=apps, legend=True, title="Legend")
    plot_bump(axes[1], sub, apps=apps, direct_labels=True, title="End labels")
    plot_bump(axes[2], sub, apps=apps, direct_labels=True, title="Both-end labels")
    # add first labels manually for panel 3
    for app in apps:
        ss = sub[sub["app"] == app].sort_values("week")
        first = ss[ss["rank_unique"].notna()].iloc[0]
        axes[2].text(0 - 0.08, float(first["rank_unique"]), f"{app}  ", ha="right", va="center", fontsize=FONT_SMALL, color=app_color(app, True))

    fig.suptitle("Figure 17. Direct labels beat legends", y=1.03)
    save(fig, "figure_17_labels_vs_legends.png")


def figure_18_topn_caveat(df):
    # create top-N only view versus full highlighted view
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.35)

    top4 = df[df["rank_unique"] <= 4].copy()
    plot_bump(axes[0], top4, apps=sorted(top4["app"].dropna().unique()), direct_labels=True, labels_end_only=True,
              label_above_apps=["TaskPilot"], title="Top 4 only")

    plot_bump(
        axes[1],
        df,
        apps=sorted(df["app"].unique()),
        highlight_apps=["EchoWrite", "NovaAI", "TaskPilot", "MemoSpark", "ChatFlow"],
        direct_labels=True,
        labels_end_only=True,
        title="Full field with highlights"
    )

    fig.suptitle("Figure 18. Top N helps, but can hide challengers", y=1.03)
    save(fig, "figure_18_topn_caveat.png")


def figure_19_small_multiples(df):
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.65)

    ax_big = fig.add_subplot(gs[0, :])
    plot_bump(ax_big, df, apps=sorted(df["app"].unique()), direct_labels=True, labels_end_only=True, label_fontsize=FONT_SMALL, title="One tangled chart")

    groups = [
        ["NovaAI", "TaskPilot", "ChatFlow", "MemoSpark"],
        ["EchoWrite", "TrendLens", "PixelNote", "DataMuse"],
        ["StudyMate", "CalmScript", "ClipMind", "PromptPad"],
    ]
    titles = ["Top leaders", "Risers", "Lower table"]

    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        plot_bump(ax, df[df["app"].isin(groups[i])], apps=groups[i], direct_labels=True, labels_end_only=True, label_fontsize=FONT_SMALL, title=titles[i])

    fig.suptitle("Figure 19. Small multiples can beat one tangled chart", y=0.98)
    save(fig, "figure_19_small_multiples.png")


def figure_20_line_shapes(df):
    apps = ["NovaAI", "ChatFlow", "MemoSpark", "EchoWrite"]
    sub = df[df["app"].isin(apps)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_bump(axes[0], sub, apps=apps, smooth=False, markers=True, title="Straight lines")
    plot_bump(axes[1], sub, apps=apps, smooth=True, markers=True, title="Smooth curves")
    fig.suptitle("Figure 20. Straight lines vs smooth curves", y=1.03)
    save(fig, "figure_20_line_shapes.png")


def figure_21_final_hero(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.subplots_adjust(wspace=0.35)
    ax1, ax2 = axes[0], axes[1]

    plot_bump(
        ax1,
        df,
        apps=sorted(df["app"].unique()),
        direct_labels=True,
        labels_end_only=True,
        label_fontsize=FONT_SMALL,
        highlight_apps=HIGHLIGHT_APPS,
        title="Best-practice bump chart"
    )

    plot_value_lines(ax2, df, ["NovaAI", "ChatFlow", "MemoSpark", "EchoWrite"], legend=True, legend_inside=True, title="Companion value chart")

    fig.suptitle("Figure 21. Final best-practice reference figure", y=1.02)
    save(fig, "figure_21_final_hero.png")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    df = load_data()

    figure_01_line_chart_hides_race(df)
    figure_02_definition_anatomy(df)
    figure_03_ordinal_not_quantitative(df)
    figure_04_how_to_read(df)
    figure_05_crossing_is_event(df)
    figure_06_crossing_not_magnitude(df)
    figure_07_hidden_tradeoff(df)
    figure_08_history_timeline()
    figure_09_where_bump_shines(df)
    figure_10_when_to_skip(df)
    figure_11_line_vs_slope_vs_bump(df)
    figure_12_field_size_caveat()
    figure_13_data_prep_pipeline()
    figure_14_tie_breaking_rules(df)
    figure_15_missing_and_time_spacing(df)
    figure_16_design_before_after(df)
    figure_17_labels_vs_legends(df)
    figure_18_topn_caveat(df)
    figure_19_small_multiples(df)
    figure_20_line_shapes(df)
    figure_21_final_hero(df)

    print(f"Saved 21 figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
