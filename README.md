# Lumiplot Plotting & Stats Series

This repo hosts the Lumiplot tutorial series: each folder pairs a published article with all the code, data, and figures used in that post.

The goal: show how to go from real-world research questions → honest, readable, statistically sound plots, with and without Lumiplot.

---

## Who this series is for

- Data scientists and applied researchers
- People who already know basic Python / pandas / plotting
- Anyone who wants better defaults for statistical graphics and a faster path from "raw CSV" to "journal-ready figure"

You don't need to use Lumiplot to get value from the code, but you'll see how it fits into a modern plotting workflow.

---

## Repository structure

Each directory corresponds to one article in the series:

- `Violin_plot/` – Violin plots vs boxplots tutorial
  - `README.md` – detailed article content and explanations
  - `*.py` – Python scripts to generate figures
  - `*.csv` – datasets used in the examples

Future posts will follow a similar pattern:

```
NN-short-descriptive-title/
  README.md
  notebooks/        # optional – Jupyter notebooks for exploration
  *.py              # Python scripts for figure generation
  data/             # or *.csv files in the folder
  figures/          # optional – exported PNG/SVGs
  prompts/          # optional – Lumiplot or LLM prompts
```

---

## Current posts

### 1. What Violin Plots Tell You That Boxplots Hide

**Article:**  
[What Violin Plots Tell You That Boxplots Hide](https://medium.com/p/what-violin-plots-tell-you-that-boxplots-hide-b8b9831f855b)

**Focus:**
- When violin plots are better than boxplots
- When they fail or mislead
- How to layer boxplots, jitter, beeswarms, rugs, and rainclouds without lying with your data
- Practical design choices: width, opacity, inner annotations, and overlays

**Code in this repo:**
- Generate synthetic "delivery wait time" data (London bike deliveries)
- Compare boxplots, violins, and raincloud-style overlays
- Show "good vs bad" styling choices for violin plots
- Example Lumiplot prompts for reproducing the same figures automatically

**Location:** `Violin_plot/`

---

## Using Lumiplot with these examples

Where relevant, you'll find:
- Prompt examples that describe the desired plot in natural language
- Notes on how Lumiplot chooses:
  - Plot type (violin vs boxplot vs histogram…)
  - Statistical tests and overlays
  - Styling defaults (width, opacity, annotations)

The idea is not to hide the code, but to show how Lumiplot can:
- Get you to a good first draft figure in seconds
- Still let you inspect and edit the underlying Python/Plotly code

For violin plots, see the detailed prompts and examples in `Violin_plot/README.md`.

---

## Contributing & feedback

This series is a work in progress. If you:
- Spot a bug
- Have a better example dataset
- Want to see a specific topic (e.g., mixed models, uncertainty ribbons, calibration plots)

…feel free to open an issue or a pull request with:
- A short description of the problem / suggestion
- Any relevant code diffs or links to references

---

## License

Update this section to match how you want to share things. For example:
- Code: MIT
- Text & figures: CC BY 4.0

Until that's finalized, please treat this repo as "look but ask before reusing in commercial products".

