import sys

sys.path.append('.')  # For running from root directory
sys.path.append('..')  # For running from scripts/ directory

import numpy as np

from src.utils import compute_proportions, compute_proportions_MC

from src.samplers import generate_samples_MC_free_transects, generate_samples_MC_ND_transects, \
                    generate_samples_MC_parallel_transects, generate_samples_MC_random, \
                    subsample_quadrats, get_half_quadrat_samples

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from params.parameters import CLASS_PARAMS

colors = [CLASS_PARAMS[i]['color'] for i in range(len(CLASS_PARAMS))]
classes = [CLASS_PARAMS[i]['name'] for i in range(len(CLASS_PARAMS))]

img = np.load("data/map.npy")

true_proportions = compute_proportions(img, n_classes=len(classes) - 1)

n_MC_samples = int(1e4)

n_samples = 100
quadrat_size = 1.0

MC_samples = generate_samples_MC_random(img, n_samples, n_MC_samples, quadrat_size)

n_samples = [10, 25, 50, 100]

proportions = []
for i, n in enumerate(n_samples):
    if n < MC_samples.shape[0]:
        samples = subsample_quadrats(MC_samples, n)
    else:
        samples = MC_samples[:]
    proportions.append(compute_proportions_MC(samples))

proportions05 = []
for i, n in enumerate(n_samples):
    if n < MC_samples.shape[0]:
        samples = subsample_quadrats(MC_samples, n)
    else:
        samples = MC_samples[:]
    samples = get_half_quadrat_samples(samples)
    proportions05.append(compute_proportions_MC(samples))

cats = [5, 8, 7]
color1 = 'tab:blue'
color2 = 'tab:orange'

# Create figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for idx, cat in enumerate(cats):
    true_cover = true_proportions[cat] * 100

    cover_estimates = []
    cover_estimates05 = []
    maes = []  # MAE for 1m quadrats
    maes05 = []  # MAE for 0.5m quadrats

    for i, n in enumerate(n_samples):
        # Get estimates for both quadrat sizes
        estimates = [prop[cat] * 100 for prop in proportions[i]]
        estimates05 = [prop[cat] * 100 for prop in proportions05[i]]

        cover_estimates.append(estimates)
        cover_estimates05.append(estimates05)

        mae = np.mean(np.abs(np.array(estimates) - true_cover))
        mae05 = np.mean(np.abs(np.array(estimates05) - true_cover))

        maes.append(mae)
        maes05.append(mae05)

    ax = axs[idx]
    ax.axhline(y=true_cover, color='black', linestyle='--', alpha=0.6)

    violin_parts1 = ax.violinplot(cover_estimates, showmeans=True,
                                  quantiles=len(cover_estimates) * [[0.025, 0.975]], side="low")
    violin_parts2 = ax.violinplot(cover_estimates05, showmeans=True,
                                  quantiles=len(cover_estimates05) * [[0.025, 0.975]], side="high")

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.3 * y_max)  # Add some extra space at the top

    # Add MAE annotations - for 1m quadrats (left side)
    for i, mae in enumerate(maes):
        ax.text(i + 1 - 0.1, max(cover_estimates05[i]) + 0.03 * y_max,
                f'MAE: \n{mae:.2f}%',
                horizontalalignment='right',
                color=color1,
                fontsize=16)

    # Add MAE annotations - for 0.5m quadrats (right side)
    for i, mae in enumerate(maes05):
        ax.text(i + 1 + 0.1, max(cover_estimates05[i]) + 0.03 * y_max,
                f'MAE: \n{mae:.2f}%',
                horizontalalignment='left',
                color=color2,
                fontsize=16)

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', label=f'True cover: {true_cover:.2f}%'),
        Patch(facecolor=color1, alpha=0.6, label="quadrat size = 1m"),
        Patch(facecolor=color2, alpha=0.6, label="quadrat size = 0.5m")
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(n_samples) + 1))

    # Only set x-tick labels for the bottom subplot
    if idx == 2:  # Bottom subplot
        ax.set_xticklabels(n_samples, fontsize=16)
        ax.set_xlabel('Number of quadrats', fontsize=16)
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis='y', labelsize=16)
    ax.set_title(f'{classes[cat + 1]}', fontsize=16, fontdict={'style': 'italic'})
    ax.set_ylabel('Cover estimate (%)', fontsize=16)

plt.tight_layout()
plt.savefig("figures/quadrat_size_and_number_effects.pdf", bbox_inches='tight')