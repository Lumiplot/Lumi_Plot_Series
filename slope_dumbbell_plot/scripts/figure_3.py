import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# Set font to Inter for all plots
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# -------------------------
# Theme Colors
# -------------------------
COLOR_2024 = "#292F36"  # Dark Gray / Charcoal
COLOR_2025 = "#A41F13"  # Reddish-Brown / Burnt Orange
COLOR_TARGET = "#8F7A6E"  # Medium Brown / Taupe
COLOR_LINE = "#E0DBD8"  # Light Gray / Beige - for connecting lines
COLOR_BG = "#FAF5F1"  # Off-White / Cream - for backgrounds
COLOR_GREY = "#CCCCCC"  # Grey for de-emphasized lines

# -------------------------
# Load data from CSV
# -------------------------
# Data is in the parent directory (slope_dumbbell folder)
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "survey_data.csv")
df = pd.read_csv(data_path)

# Rename columns to match script expectations
df = df.rename(columns={
    "Category": "attribute",
    "2024 Q4": "competitor",
    "2025 Q4": "our_company"
})

# Remove any rows with missing data
df = df.dropna()

TARGET = 60  # percent

# Sort data by 2025 Q4 values (descending) - highest values at top
d = df.copy().sort_values("our_company", ascending=False).reset_index(drop=True)

# Add small offsets to stagger overlapping points
def add_jitter(values, jitter_amount=0.8):
    """Add small vertical offsets to overlapping values"""
    jittered = values.copy()
    value_counts = {}
    for idx, val in enumerate(values):
        if val in value_counts:
            value_counts[val].append(idx)
        else:
            value_counts[val] = [idx]
    
    for val, indices in value_counts.items():
        if len(indices) > 1:
            # Stagger overlapping points
            for i, idx in enumerate(indices):
                offset = (i - (len(indices) - 1) / 2) * jitter_amount
                jittered[idx] = val + offset
    return jittered

# Apply jitter to both sides
competitor_jittered = add_jitter(d["competitor"].values)
our_company_jittered = add_jitter(d["our_company"].values)

x0, x1 = 0, 1

# -------------------------
# 1) BAD: All slope lines same color, same before/after colors, no annotation
# Labels on both sides to make it messy
# Figure 3a
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

for i, row in d.iterrows():
    comp_val = competitor_jittered[i]
    our_val = our_company_jittered[i]
    # All lines same color (grey)
    ax.plot([x0, x1], [comp_val, our_val], linewidth=2, color='#666666', alpha=0.6, zorder=1)
    # Same color for both before and after points
    ax.scatter([x0], [comp_val], s=30, color='#666666', zorder=3)
    ax.scatter([x1], [our_val], s=30, color='#666666', zorder=3)

# Left side labels (attribute names) - makes it messy
for i, row in d.iterrows():
    comp_val = competitor_jittered[i]
    ax.text(x0 - 0.08, comp_val, row["attribute"],
            ha="right", va="center", fontsize=7, color='#666666')

# Right side labels (attribute names) - makes it messy
for i, row in d.iterrows():
    our_val = our_company_jittered[i]
    ax.text(x1 + 0.03, our_val, row["attribute"],
            ha="left", va="center", fontsize=7, color='#666666')

ax.set_xlim(-0.5, 1.2)
ax.set_xticks([x0, x1])
ax.set_xticklabels(["2024 Q4", "2025 Q4"])
ax.set_ylim(40, 75)
ax.set_ylabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_3a.png")
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Saved: {output_path}")

# -------------------------
# 2) BETTER: Change color of before/after
# Figure 3b
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

for i, row in d.iterrows():
    comp_val = competitor_jittered[i]
    our_val = our_company_jittered[i]
    # Lines still same color (grey)
    ax.plot([x0, x1], [comp_val, our_val], linewidth=2, color=COLOR_LINE, alpha=0.6, zorder=1)
    # Different colors for before and after
    ax.scatter([x0], [comp_val], s=30, color=COLOR_2024, zorder=3)
    ax.scatter([x1], [our_val], s=30, color=COLOR_2025, zorder=3)

# Right side labels (attribute names only)
for i, row in d.iterrows():
    our_val = our_company_jittered[i]
    ax.text(x1 + 0.03, our_val, row["attribute"],
            ha="left", va="center", fontsize=9, color=COLOR_2025)

ax.set_xlim(-0.2, 1.2)
ax.set_xticks([x0, x1])
ax.set_xticklabels(["2024 Q4", "2025 Q4"])
ax.set_ylim(40, 75)
ax.set_ylabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_3b.png")
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Saved: {output_path}")

# -------------------------
# 3) GREAT: Change color based on direction differently, add annotation
# Figure 3c
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

for i, row in d.iterrows():
    comp_val = competitor_jittered[i]
    our_val = our_company_jittered[i]
    diff = our_val - comp_val
    
    # Color based on direction: increase (positive) vs decrease (negative)
    # Using theme colors only
    if diff > 0:
        # Increase: grey style (like non-highlighted in 3d)
        line_color = COLOR_GREY
        point_color_before = COLOR_GREY
        point_color_after = COLOR_GREY
        line_width = 1.5
        point_size = 20
        edge_width = 0
        use_edge = False
        line_alpha = 0.4
        point_alpha = 0.5
        zorder_line = 1
        zorder_point = 3
    elif diff < 0:
        # Decrease: use COLOR_2025 (Reddish-Brown) - highlighted effect
        line_color = COLOR_2025
        point_color_before = COLOR_2024
        point_color_after = COLOR_2025
        line_width = 3
        point_size = 50
        edge_width = 1.5
        use_edge = True
        line_alpha = 0.9
        point_alpha = 1.0
        zorder_line = 2
        zorder_point = 4
    else:
        # No change: grey
        line_color = COLOR_GREY
        point_color_before = COLOR_GREY
        point_color_after = COLOR_GREY
        line_width = 1.5
        point_size = 20
        edge_width = 0
        use_edge = False
        line_alpha = 0.4
        point_alpha = 0.5
        zorder_line = 1
        zorder_point = 3
    
    ax.plot([x0, x1], [comp_val, our_val], linewidth=line_width, color=line_color, alpha=line_alpha, zorder=zorder_line)
    if use_edge:
        ax.scatter([x0], [comp_val], s=point_size, color=point_color_before, zorder=zorder_point, edgecolors='white', linewidths=edge_width, alpha=point_alpha)
        ax.scatter([x1], [our_val], s=point_size, color=point_color_after, zorder=zorder_point, edgecolors='white', linewidths=edge_width, alpha=point_alpha)
    else:
        ax.scatter([x0], [comp_val], s=point_size, color=point_color_before, zorder=zorder_point, alpha=point_alpha)
        ax.scatter([x1], [our_val], s=point_size, color=point_color_after, zorder=zorder_point, alpha=point_alpha)

# Right side labels (attribute names)
for i, row in d.iterrows():
    our_val = our_company_jittered[i]
    diff = row["our_company"] - row["competitor"]
    if diff > 0:
        label_color = '#999999'
        label_weight = "normal"
        label_size = 9
        label_alpha = 0.6
    elif diff < 0:
        label_color = COLOR_2025
        label_weight = "bold"
        label_size = 10
        label_alpha = 1.0
    else:
        label_color = '#999999'
        label_weight = "normal"
        label_size = 9
        label_alpha = 0.6
    
    ax.text(x1 + 0.03, our_val, row["attribute"],
            ha="left", va="center", fontsize=label_size, color=label_color, weight=label_weight, alpha=label_alpha,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='none', alpha=0.9))

# Target reference line with annotation
ax.axhline(TARGET, linewidth=2, linestyle="--", alpha=0.6, color=COLOR_TARGET, zorder=0)
ax.text(x0 - 0.02, TARGET, f"TARGET", ha="right", va="bottom", 
        fontsize=7, color=COLOR_TARGET)

ax.set_xlim(-0.2, 1.2)
ax.set_xticks([x0, x1])
ax.set_xticklabels(["2024 Q4", "2025 Q4"])
ax.set_ylim(40, 75)
ax.set_ylabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_3c.png")
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Saved: {output_path}")

# -------------------------
# 4) EVEN BETTER: Highlight one bar, make all other lines grey
# Figure 3d
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

# Highlight "Tools & systems" category
highlight_attr = "Tools & systems"
highlight_idx = d[d["attribute"] == highlight_attr].index[0] if len(d[d["attribute"] == highlight_attr]) > 0 else 0

for i, row in d.iterrows():
    comp_val = competitor_jittered[i]
    our_val = our_company_jittered[i]
    
    if i == highlight_idx:
        # Highlighted line - use theme colors
        ax.plot([x0, x1], [comp_val, our_val], linewidth=3, color=COLOR_2025, alpha=0.9, zorder=2)
        ax.scatter([x0], [comp_val], s=50, color=COLOR_2024, zorder=4, edgecolors='white', linewidths=1.5)
        ax.scatter([x1], [our_val], s=50, color=COLOR_2025, zorder=4, edgecolors='white', linewidths=1.5)
    else:
        # Grey lines for others
        ax.plot([x0, x1], [comp_val, our_val], linewidth=1.5, color=COLOR_GREY, alpha=0.4, zorder=1)
        ax.scatter([x0], [comp_val], s=20, color=COLOR_GREY, zorder=3, alpha=0.5)
        ax.scatter([x1], [our_val], s=20, color=COLOR_GREY, zorder=3, alpha=0.5)

# Right side labels
for i, row in d.iterrows():
    our_val = our_company_jittered[i]
    if i == highlight_idx:
        # Bold label for highlighted
        ax.text(x1 + 0.03, our_val, row["attribute"],
                ha="left", va="center", fontsize=10, color=COLOR_2025, weight="bold")
    else:
        ax.text(x1 + 0.03, our_val, row["attribute"],
                ha="left", va="center", fontsize=9, color='#999999', alpha=0.6)

ax.set_xlim(-0.2, 1.2)
ax.set_xticks([x0, x1])
ax.set_xticklabels(["2024 Q4", "2025 Q4"])
ax.set_ylim(40, 75)
ax.set_ylabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_3d.png")
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Saved: {output_path}")

print("\nAll slope progression plots generated successfully!")
print("  - figure_3a.png: All lines same color, same before/after, no annotation")
print("  - figure_3b.png: Different colors for before/after")
print("  - figure_3c.png: Color by direction, with annotations")
print("  - figure_3d.png: Highlight one category, grey others")
