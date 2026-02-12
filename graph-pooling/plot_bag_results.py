import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Plot efficiency or accuracy of a dataset.")
parser.add_argument('--dataset', type=str, required=True,
                    choices=['coraml', 'citeseer', 'photo', 'actor', 'arxiv'],
                    help="Dataset to plot.")
parser.add_argument('--plot_type', type=str, default='efficiency',
                    choices=['efficiency', 'accuracy'],
                    help="Type of plot: efficiency (time vs epoch) or accuracy (epoch vs accuracy)")
parser.add_argument('--zoom_x', type=float, nargs=2, default=None,
                    help="Optional zoom range for X axis")
parser.add_argument('--zoom_y', type=float, nargs=2, default=None,
                    help="Optional zoom range for Y axis")
args = parser.parse_args()

dataset = args.dataset
plot_type = args.plot_type
zoom_x = args.zoom_x
zoom_y = args.zoom_y

# --- Paths ---
input_csv = './results/all_results.csv'
output_dir = './results_plotted'
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
df = pd.read_csv(input_csv)

# --- Type cleanup ---
df['bag_ratio'] = df['bag_ratio'].astype(float).round(1)
df['epoch'] = df['epoch'].astype(int)
df['time'] = df['time'].astype(float)
df['accuracy'] = df['accuracy'].astype(float)
df['model'] = df['model'].str.lower()
df['pool'] = df['pool'].str.lower()
df['single_layer'] = df['single_layer'].astype(bool)

# --- Combine model + pooling for "ins" variants ---
df['model_id'] = df.apply(
    lambda r: f"{r['model']}-{r['pool']}" if r['model'] == 'ins' else r['model'],
    axis=1
)

# --- Allowed models ---
allowed_models = [
    'gcn', 'gat', 'sage',
    'ins-sum', 'ins-mean', 'ins-attention', 'ins-set2set'
]
df = df[df['model_id'].isin(allowed_models)]

# --- Define colors ---
model_colors = {
    'gcn': 'tab:blue',
    'gat': 'tab:green',
    'sage': 'tab:orange',
    'ins-sum': 'tab:pink',
    'ins-mean': 'tab:olive',
    'ins-attention': 'tab:cyan',
    'ins-set2set': 'tab:purple'
}

# --- Markers for bag ratios ---
ratio_markers = {
    60.0: 'o',
    70.0: 's',
    80.0: '^',
    90.0: 'D'
}

# --- Line styles for layer type ---
layer_linestyle = {
    True: '-',    # SL
    False: '--'   # DL
}

# --- Filter dataset ---
data_subset = df[df['dataset'] == dataset]

# --- Create figure ---
fig, ax = plt.subplots(figsize=(8, 10))

# --- Choose axes based on plot type ---
if plot_type == 'efficiency':
    x_col, y_col = 'time', 'epoch'
    x_label, y_label = "Time (seconds)", "Epoch"
else:
    x_col, y_col = 'epoch', 'accuracy'
    x_label, y_label = "Epoch", "Accuracy"

# --- Group & plot ---
for (model_id, bag_ratio, single_layer), group in data_subset.groupby(
        ['model_id', 'bag_ratio', 'single_layer']):

    group = group.sort_values(x_col)
    color = model_colors.get(model_id, 'gray')
    marker = ratio_markers.get(bag_ratio, 'x')
    layer_tag = "SL" if single_layer else "DL"

    label = f"{int(bag_ratio)}% ratio with {model_id.upper()} {layer_tag}"

    ax.plot(
        group[x_col],
        group[y_col],
        marker=marker,
        color=color,
        linestyle=layer_linestyle[single_layer],
        linewidth=2,
        markersize=6,
        label=label
    )

# --- Main plot formatting ---
title_type = "Efficiency" if plot_type == 'efficiency' else "Accuracy"
ax.set_title(f"{title_type} Plot – {dataset.capitalize()} Dataset", fontsize=14)
ax.set_xlabel(x_label, fontsize=12)
ax.set_ylabel(y_label, fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)

# --- Deduplicate legend ---
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# --- Optional zoom inset ---
if zoom_x and zoom_y:
    inset = inset_axes(ax, width="35%", height="35%", loc='lower right', borderpad=2)

    for (_, _, _), group in data_subset.groupby(['model_id', 'bag_ratio', 'single_layer']):
        group = group.sort_values(x_col)

        zoom_group = group[
            (group[x_col] >= zoom_x[0]) & (group[x_col] <= zoom_x[1]) &
            (group[y_col] >= zoom_y[0]) & (group[y_col] <= zoom_y[1])
        ]

        if not zoom_group.empty:
            inset.plot(
                zoom_group[x_col],
                zoom_group[y_col],
                linewidth=1.5
            )

    inset.set_xlim(zoom_x)
    inset.set_ylim(zoom_y)
    inset.grid(True, linestyle='--', alpha=0.4)
    inset.set_title("Zoomed region", fontsize=10)

plt.tight_layout()

# --- Save ---
output_path = os.path.join(
    output_dir, f"{dataset}_{plot_type}_plot.png"
)
plt.savefig(output_path)
plt.close()

print(f"✅ {title_type} plot for '{dataset}' saved in '{output_path}'")
