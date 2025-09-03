import sys

sys.path.append('.')  # For running from root directory
sys.path.append('..')  # For running from scripts/ directory

import numpy as np

from src.utils import compute_proportions, generate_scenario, analyze_mc_scenario

from src.samplers import generate_samples_MC_random

import matplotlib.pyplot as plt

from tqdm import tqdm

from params.parameters import CLASS_PARAMS, n_MC

import os
os.makedirs('figures', exist_ok=True)

colors = [CLASS_PARAMS[i]['color'] for i in range(len(CLASS_PARAMS))]
classes = [CLASS_PARAMS[i]['name'] for i in range(len(CLASS_PARAMS))]
classes[8] = "Hard bottoms w/ ph. algae >50cm"

img = np.load("data/map.npy")

true_proportions = compute_proportions(img, n_classes=len(classes) - 1)

n_MC = int(n_MC)  # number of simulations
n_quadrats = 50  # number of quadrats per simulations
quadrat_size = 0.5  # meters
n_points_per_quadrat = 10
MC_samples = generate_samples_MC_random(img, n_quadrats, n_MC, quadrat_size)

cats = [5, 8, 7]
precisions = np.arange(0.5, 1.0001, 0.05)
recalls = np.arange(0.5, 1.0001, 0.05)

# Store results for all categories
all_MAEs = {}
all_Biases = {}
all_true_covers = {}

# Process all categories
for cat_idx, cat in enumerate(cats):
    print(f"Processing category {cat} ({cat_idx + 1}/{len(cats)})")

    true_cover = true_proportions[cat] * 100
    all_true_covers[cat] = true_cover

    Biases = np.zeros((len(precisions), len(recalls)))
    MAEs = np.zeros((len(precisions), len(recalls)))

    for i, p in enumerate(precisions):
        print("  Precision = %.2f" % (p))
        for j, r in tqdm(enumerate(recalls), desc="  Recalls"):
            cover_estimates = generate_scenario(MC_samples, n_quadrats, n_points_per_quadrat, p, r, cat)
            metrics = analyze_mc_scenario(cover_estimates, true_cover)
            bias = metrics["bias"]
            mae = metrics["mae"]

            Biases[i, j] = bias
            MAEs[i, j] = mae

    all_Biases[cat] = Biases
    all_MAEs[cat] = MAEs

# Create the combined figure
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

# Prepare labels
recall_labels = [f'{100 * x:.0f}' for x in recalls]
precision_labels = [f'{100 * x:.0f}' for x in precisions[::-1]]

for cat_idx, cat in enumerate(cats):
    # MAE heatmap (left column)
    ax_mae = axes[cat_idx, 0]
    data_mae = np.flipud(all_MAEs[cat])
    im_mae = ax_mae.imshow(data_mae, cmap='YlOrRd', aspect='auto')

    # Add value annotations for MAE
    for i in range(len(precisions)):
        for j in range(len(recalls)):
            ax_mae.text(j, i, f'{data_mae[i, j]:.2f}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        color="black", fontsize=10)

    # Set ticks for MAE subplot
    ax_mae.set_xticks(np.arange(len(recalls)))
    ax_mae.set_yticks(np.arange(len(precisions)))

    # X-axis labels and title only for bottom row
    if cat_idx == len(cats) - 1:  # Bottom row
        ax_mae.set_xticklabels(recall_labels, fontsize=14)
        ax_mae.set_xlabel('Recall (%)', fontsize=16)
    else:
        ax_mae.set_xticklabels([])
        ax_mae.set_xticks([])

    # Y-axis labels and title only for left column
    ax_mae.set_yticklabels(precision_labels, fontsize=14)
    ax_mae.set_ylabel('Precision (%)', fontsize=16)

    # Add colorbar for MAE
    cbar_mae = plt.colorbar(im_mae, ax=ax_mae)
    # cbar_mae.set_label('Mean Absolute Error')

    # Row titles (category names with true cover) - only on left column
    if cat_idx == 0:  # First row - add column titles
        ax_mae.set_title('Mean Absolute Error', fontsize=14, fontweight='bold', pad=10)

    # Add row label on the left
    ax_mae.text(-0.25, 0.5, f'{classes[cat + 1]}',
                transform=ax_mae.transAxes, fontsize=14, fontweight='bold',
                verticalalignment='center', horizontalalignment='center', rotation=90)

    # Bias heatmap (right column)
    ax_bias = axes[cat_idx, 1]
    data_bias = np.flipud(all_Biases[cat])

    max_abs_bias = np.max(np.abs(data_bias))
    vmin_bias = -max_abs_bias
    vmax_bias = max_abs_bias

    im_bias = ax_bias.imshow(data_bias, cmap='RdBu_r', aspect='auto', vmin=vmin_bias, vmax=vmax_bias)

    # Add value annotations for Bias
    for i in range(len(precisions)):
        for j in range(len(recalls)):
            ax_bias.text(j, i, f'{data_bias[i, j]:.2f}',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color="black", fontsize=10)

    # Set ticks for Bias subplot
    ax_bias.set_xticks(np.arange(len(recalls)))
    ax_bias.set_yticks(np.arange(len(precisions)))

    # X-axis labels and title only for bottom row
    if cat_idx == len(cats) - 1:  # Bottom row
        ax_bias.set_xticklabels(recall_labels, fontsize=14)
        ax_bias.set_xlabel('Recall (%)', fontsize=16)
    else:
        ax_bias.set_xticklabels([])
        ax_bias.set_xticks([])

    # Y-axis labels only for left column (but no title since it's right column)
    ax_bias.set_yticklabels([])  # Right column gets no y-labels
    ax_bias.set_yticks([])

    # Column title for Bias - only on first row
    if cat_idx == 0:
        ax_bias.set_title('Bias', fontsize=14, fontweight='bold', pad=10)

    # Add colorbar for Bias
    cbar_bias = plt.colorbar(im_bias, ax=ax_bias)
    # cbar_bias.set_label('Bias')

# Adjust layout
plt.tight_layout()
plt.savefig("figures/annotation_error_effects.pdf", bbox_inches='tight')