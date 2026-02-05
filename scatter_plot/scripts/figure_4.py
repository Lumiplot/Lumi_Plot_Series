import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import tempfile

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

rng = np.random.default_rng(20251213)

def ols_mean_ci(x, y, xgrid, alpha=0.05):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    Xg = sm.add_constant(xgrid)
    pred = model.get_prediction(Xg).summary_frame(alpha=alpha)
    return model, pred["mean"].to_numpy(), pred["mean_ci_lower"].to_numpy(), pred["mean_ci_upper"].to_numpy()

def loess_boot_ci(x, y, xgrid, frac=0.35, B=250, alpha=0.05):
    base = lowess(y, x, frac=frac, it=0, return_sorted=True)
    yhat = np.interp(xgrid, base[:, 0], base[:, 1])

    boots = np.empty((B, xgrid.size), dtype=float)
    n = x.size
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        fitb = lowess(yb, xb, frac=frac, it=0, return_sorted=True)
        boots[b] = np.interp(xgrid, fitb[:, 0], fitb[:, 1], left=np.nan, right=np.nan)

    lower = np.nanpercentile(boots, 100 * (alpha / 2), axis=0)
    upper = np.nanpercentile(boots, 100 * (1 - alpha / 2), axis=0)
    return yhat, lower, upper

def glm_logit_mean_ci(x, y, xgrid, alpha=0.05):
    X = sm.add_constant(x)
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    Xg = sm.add_constant(xgrid)
    sf = model.get_prediction(Xg).summary_frame(alpha=alpha)
    return model, sf["mean"].to_numpy(), sf["mean_ci_lower"].to_numpy(), sf["mean_ci_upper"].to_numpy()

def save_single_panel(figpath: Path, title: str, xlabel: str, ylabel: str, draw_fn):
    fig = plt.figure(figsize=(5.2, 4.2), dpi=220)
    ax = fig.add_axes([0.14, 0.14, 0.82, 0.78])
    draw_fn(ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.savefig(figpath, facecolor="white")
    plt.close(fig)

# Create temporary directory for panel images
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    
    # ---------------- Panel 1: LOESS ----------------
    n = 240
    x = rng.uniform(18, 62, n)

    # Story: outcome increases with X but bends (nonlinear)
    y_true = 1800 + 520*x - 7.8*(x - 40)**2
    y = y_true + rng.normal(0, 1400, n)

    xgrid = np.linspace(x.min(), x.max(), 240)

    panel1 = tmpdir_path / "panel_loess.png"

    def draw_loess(ax):
        ax.scatter(x, y, s=14, alpha=0.7, color=COLORS[0], edgecolors='none')
        yhat, lo, hi = loess_boot_ci(x, y, xgrid, frac=0.28, B=250, alpha=0.05)
        line, = ax.plot(xgrid, yhat, linewidth=2.4, color=COLORS[0])
        ax.fill_between(xgrid, lo, hi, alpha=0.18, color=COLORS[0])

    save_single_panel(panel1,
                      "Same data, different smoother: LOESS",
                      "Exposure proxy (X)",
                      "Outcome (Y)",
                      draw_loess)

    # ---------------- Panel 2: Linear regression ----------------
    panel2 = tmpdir_path / "panel_regression.png"

    def draw_reg(ax):
        ax.scatter(x, y, s=14, alpha=0.7, color=COLORS[0], edgecolors='none')
        _, mean, lo, hi = ols_mean_ci(x, y, xgrid, alpha=0.05)
        line, = ax.plot(xgrid, mean, linewidth=2.4, color=COLORS[0])
        ax.fill_between(xgrid, lo, hi, alpha=0.18, color=COLORS[0])

    save_single_panel(panel2,
                      "Same data, different smoother: Linear regression",
                      "Exposure proxy (X)",
                      "Outcome (Y)",
                      draw_reg)

    # ---------------- Panel 3: Logistic regression ----------------
    k = 220
    x4 = rng.uniform(0, 10, k)

    # Story: event probability rises sharply after a threshold
    p = 1 / (1 + np.exp(-(x4 - 5.2)*1.35))
    y4 = (rng.random(k) < p).astype(int)

    x4g = np.linspace(x4.min(), x4.max(), 240)
    panel3 = tmpdir_path / "panel_logistic.png"

    def draw_logistic(ax):
        yj = y4 + rng.normal(0, 0.03, size=y4.size)  # jitter for visibility
        yj = np.clip(yj, -0.08, 1.08)
        ax.scatter(x4, yj, s=14, alpha=0.7, color=COLORS[0], edgecolors='none')
        _, mean, lo, hi = glm_logit_mean_ci(x4, y4, x4g, alpha=0.05)
        line, = ax.plot(x4g, mean, linewidth=2.4, color=COLORS[0])
        ax.fill_between(x4g, lo, hi, alpha=0.18, color=COLORS[0])
        ax.set_ylim(-0.1, 1.1)

    save_single_panel(panel3,
                      "Logistic regression: probability curve (with CI)",
                      "Score / dose (X)",
                      "Probability of event",
                      draw_logistic)

    # ---------------- Stitch 3 panels into a 1x3 image ----------------
    imgs = [Image.open(p).convert("RGB") for p in [panel1, panel2, panel3]]
    pad = 20
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)

    canvas = Image.new("RGB", (3*w + 2*pad, h), (255, 255, 255))
    positions = [(0, 0), (w + pad, 0), (2*w + 2*pad, 0)]
    for im, (x0, y0) in zip(imgs, positions):
        canvas.paste(im, (x0, y0))

    # Save final figure
    output_path = Path("../images/figure_4.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Figure 4 saved to {output_path.absolute()}")

