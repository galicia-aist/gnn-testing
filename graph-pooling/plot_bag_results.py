import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Plot efficiency of a dataset with optional zoom.")
parser.add_argument('--dataset', type=str, required=True,
                    choices=['coraml', 'citeseer', 'photo', 'actor'],
                    help="Dataset to plot.")
parser.add_argument('--zoom_x', type=float, nargs=2, default=None,
                    help="Optional zoom range for X axis (time), e.g., 60 80")
parser.add_argument('--zoom_y', type=float, nargs=2, default=None,
                    help="Optional zoom range for Y axis (epoch), e.g., 480 520")
args = parser.parse_args()

dataset = args.dataset
zoom_x = args.zoom_x
zoom_y = args.zoom_y

# --- Paths ---
input_csv = './results/all_results.csv'
output_dir = './results_plotted'
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
df = pd.read_csv(input_csv)

# Ensure correct types
df['bag_ratio'] = df['bag_ratio'].astype(float).round(1)
df['epoch'] = df['epoch'].astype(int)
df['time'] = df['time'].astype(float)
df['model'] = df['model'].str.lower()

# Keep only allowed models
allowed_models = ['gcn', 'gat', 'sage']
df = df[df['model'].isin(allowed_models)]

# --- Define colors for models ---
model_colors = {
    'gcn': 'tab:blue',
    'gat': 'tab:green',
    'sage': 'tab:orange'
}

# Define markers for bag ratios
ratio_markers = {
    60.0: 'o',
    70.0: 's',
    80.0: '^',
    90.0: 'D'
}

# --- Filter dataset ---
data_subset = df[df['dataset'] == dataset]

# --- Create main figure ---
fig, ax = plt.subplots(figsize=(8, 6))

# Group by (model, bag_ratio) for connected lines
for (model, bag_ratio), group in data_subset.groupby(['model', 'bag_ratio']):
    group = group.sort_values('time')
    color = model_colors.get(model, 'gray')
    marker = ratio_markers.get(bag_ratio, 'x')
    label = f"{int(bag_ratio)}% ratio with {model.upper()}"

    ax.plot(
        group['time'],
        group['epoch'],
        marker=marker,
        color=color,
        linewidth=2,
        markersize=6,
        label=label
    )

# Main plot settings
ax.set_title(f"Efficiency Plot – {dataset.capitalize()} Dataset", fontsize=14)
ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_ylabel("Epoch", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)

# Remove duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# --- Optional zoomed inset ---
if zoom_x and zoom_y:
    inset = inset_axes(ax, width="35%", height="35%", loc='lower right', borderpad=2)
    for (model, bag_ratio), group in data_subset.groupby(['model', 'bag_ratio']):
        group = group.sort_values('time')
        color = model_colors.get(model, 'gray')
        marker = ratio_markers.get(bag_ratio, 'x')

        zoom_group = group[(group['epoch'] >= zoom_y[0]) & (group['epoch'] <= zoom_y[1]) &
                           (group['time'] >= zoom_x[0]) & (group['time'] <= zoom_x[1])]
        if not zoom_group.empty:
            inset.plot(
                zoom_group['time'],
                zoom_group['epoch'],
                marker=marker,
                color=color,
                linewidth=1.5,
                markersize=4
            )

    inset.set_xlim(zoom_x)
    inset.set_ylim(zoom_y)
    inset.grid(True, linestyle='--', alpha=0.4)
    inset.set_title("Zoomed region", fontsize=10)

plt.tight_layout()

# --- Save figure ---
output_path = os.path.join(output_dir, f"{dataset}_efficiency_other.png")
plt.savefig(output_path)
plt.close()
print(f"✅ Efficiency plot for '{dataset}' saved in '{output_path}'")
