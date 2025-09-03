import sys

sys.path.append('.')  # For running from root directory
sys.path.append('..')  # For running from scripts/ directory

import numpy as np

from src.utils import compute_proportions, compute_proportions_MC

from src.samplers import generate_samples_MC_random, sample_quadrats_MC

import matplotlib.pyplot as plt

from params.parameters import CLASS_PARAMS, n_MC

import os
os.makedirs('figures', exist_ok=True)

colors = [CLASS_PARAMS[i]['color'] for i in range(len(CLASS_PARAMS))]
classes = [CLASS_PARAMS[i]['name'] for i in range(len(CLASS_PARAMS))]

img = np.load("data/map.npy")

true_proportions = compute_proportions(img, n_classes=len(classes) - 1)

n_MC_samples = int(n_MC)

n_samples = 50
quadrat_size = 0.5

MC_samples = generate_samples_MC_random(img, n_samples, n_MC_samples, quadrat_size)

n_points = [10, 25, 100]

proportions = []
for i, n in enumerate(n_points):
    samples = MC_samples[:]
    samples_pts = sample_quadrats_MC(MC_samples[:], n)

    proportions.append(compute_proportions_MC(samples_pts))

proportions.append(compute_proportions_MC(MC_samples))
proportions = np.array(proportions)

cats = [5, 8, 7]

# Create figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for idx, cat in enumerate(cats):
    true_cover = true_proportions[cat] * 100

    cover_estimates = []
    maes = []
    for i in range(len(proportions)):
        estimates = [prop[cat] * 100 for prop in proportions[i]]
        cover_estimates.append(estimates)
        errors = np.abs(np.array(estimates) - true_cover)
        mae = np.mean(errors)
        maes.append(mae)

    ax = axs[idx]
    ax.axhline(y=true_cover, color='black', linestyle='--', alpha=0.6, label=f'True cover: {true_cover:.2f}%')

    violin_parts = ax.violinplot(cover_estimates, showmeans=True,
                                 quantiles=len(cover_estimates) * [[0.025, 0.975]])
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.3 * y_max)  # Add some extra space at the top

    # Add MAE values at the top of each violin plot
    for i, mae in enumerate(maes):
        ax.text(i + 1, max(cover_estimates[i]) + 0.02 * y_max,
                f'MAE: {mae:.2f}%',
                horizontalalignment='center',
                verticalalignment="bottom",
                fontsize=16, color='tab:blue')

    labels = ["10 points", "25 points", "100 points", "All points"]
    ax.set_xticks(range(1, 1 + len(labels)))

    # Only set x-tick labels for the bottom subplot
    if idx == 2:  # Bottom subplot
        ax.set_xticklabels(labels, fontsize=16)
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3)

    # Set title with italics
    ax.set_title(f'{classes[cat + 1]}', fontsize=16, fontdict={'style': 'italic'})

    ax.set_ylabel('Cover estimate (%)', fontsize=16)
    ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig("figures/point_sampling_density_effects.pdf", bbox_inches='tight')
