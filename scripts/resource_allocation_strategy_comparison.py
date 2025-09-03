import sys

sys.path.append('.')  # For running from root directory
sys.path.append('..')  # For running from scripts/ directory

import numpy as np

from src.utils import compute_proportions, compute_proportions_MC, analyze_mc_scenario, generate_scenario

from src.samplers import generate_samples_MC_random, \
                    subsample_quadrats, sample_quadrats_MC, get_half_quadrat_samples, simulate_predictions_MC

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from params.parameters import CLASS_PARAMS

colors = [CLASS_PARAMS[i]['color'] for i in range(len(CLASS_PARAMS))]
classes = [CLASS_PARAMS[i]['name'] for i in range(len(CLASS_PARAMS))]
classes[8] = "Hard bottoms w/ ph. algae >50cm"

img = np.load("data/map.npy")

true_proportions = compute_proportions(img, n_classes=len(classes) - 1)

n_MC = int(1e4)  # number of simulations

print("Scenario 1\n")
n_quadrats = 200  # number of quadrats per simulations
quadrat_size = 0.5  # meters
n_points_per_quadrat = 10
MC_samples = generate_samples_MC_random(img, n_quadrats, n_MC, quadrat_size)

cats = [5, 8, 7]
precision = 0.75
recall = 0.75

cover_estimates_1 = []
for cat_idx, cat in enumerate(cats):
    print(f"Processing category {cat} ({cat_idx + 1}/{len(cats)})")

    true_cover = true_proportions[cat] * 100

    cover_estimates_1.append(generate_scenario(MC_samples, n_quadrats, n_points_per_quadrat, precision, recall, cat))

print("\nScenario 2\n")
n_quadrats = 50  # number of quadrats per simulations
quadrat_size = 0.5  # meters
n_points_per_quadrat = 10
MC_samples = generate_samples_MC_random(img, n_quadrats, n_MC, quadrat_size)

cats = [5, 8, 7]
precision = 0.99
recall = 0.99

cover_estimates_2 = []
for cat_idx, cat in enumerate(cats):
    print(f"Processing category {cat} ({cat_idx + 1}/{len(cats)})")

    true_cover = true_proportions[cat] * 100

    cover_estimates_2.append(generate_scenario(MC_samples, n_quadrats, n_points_per_quadrat, precision, recall, cat))

cover_estimates_1 = [np.array(cover_estimates_1[i]) for i in range(3)]
cover_estimates_2 = [np.array(cover_estimates_2[i]) for i in range(3)]

cats = [5, 8, 7]  # Your class indices
color1 = 'tab:blue'  # Many samples, poor model
color2 = 'tab:orange'  # Few samples, good model

# Create figure with 3 subplots side by side
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

for idx, (ax, cat) in enumerate(zip(axs, cats)):
    true_cover = true_proportions[cat] * 100

    # Get estimates for both scenarios
    estimates_s1 = [prop for prop in cover_estimates_1[idx]]  # Many samples, poor model
    estimates_s2 = [prop for prop in cover_estimates_2[idx]]  # Few samples, good model

    # Calculate MAE for both scenarios
    mae_s1 = np.mean(np.abs(np.array(estimates_s1) - true_cover))
    mae_s2 = np.mean(np.abs(np.array(estimates_s2) - true_cover))

    # Create split violin plots at position 1
    violin_parts1 = ax.violinplot([estimates_s1], positions=[1], showmeans=True,
                                  quantiles=[[0.025, 0.975]], side="low")
    violin_parts2 = ax.violinplot([estimates_s2], positions=[1], showmeans=True,
                                  quantiles=[[0.025, 0.975]], side="high")

    # Add true cover line
    ax.axhline(y=true_cover, color='black', linestyle='--', linewidth=2, alpha=0.6)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.15 * (y_max - y_min))  # Add some extra space at the top

    # Add MAE annotations
    max_y_s1 = max(estimates_s1)
    max_y_s2 = max(estimates_s2)
    max_y = max(max_y_s1, max_y_s2)

    # MAE for scenario 1 (left side)
    ax.text(1 - 0.15, max_y + 0.02 * (y_max - y_min),
            f'MAE:\n{mae_s1:.2f}%',
            horizontalalignment='center',
            color=color1,
            fontsize=16)

    # MAE for scenario 2 (right side)
    ax.text(1 + 0.15, max_y + 0.02 * (y_max - y_min),
            f'MAE:\n{mae_s2:.2f}%',
            horizontalalignment='center',
            color=color2,
            fontsize=16)

    # Formatting for each subplot
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1])
    ax.tick_params(bottom=False)

    # Add class labels below each subplot
    ax.set_xlabel(f'{classes[cat + 1]}', fontsize=16)
    ax.set_xticklabels([''], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Only add ylabel to leftmost subplot
    if idx == 0:
        ax.set_ylabel('Cover estimate (%)', fontsize=16)

    # Set x-axis limits to center the violins better
    ax.set_xlim(0.4, 1.6)

# Create legend below the graph
legend_elements = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='True cover'),
    Patch(facecolor=color1, alpha=0.6, label="Many quadrats (200), poor annotations (P=0.75; R=0.75)"),
    Patch(facecolor=color2, alpha=0.6, label="Few quadrats (50), good annotations (P=0.99; R=0.99)")
]

# Add legend below the plots
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1),
           ncol=3, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for legend below
plt.savefig("figures/resource_allocation_strategy_comparison.pdf", bbox_inches='tight')