import sys

sys.path.append('.')  # For running from root directory
sys.path.append('..')  # For running from scripts/ directory

import numpy as np

from src.utils import display_map, compute_proportions, display_proportions
from src.samplers import random_sampling_map, free_transect_sampling_map, parallel_transect_sampling_map, ND_transect_sampling_map

from params.parameters import CLASS_PARAMS

colors = [CLASS_PARAMS[i]['color'] for i in range(len(CLASS_PARAMS))]
classes = [CLASS_PARAMS[i]['name'] for i in range(len(CLASS_PARAMS))]

img = np.load("data/map.npy")

# Save map figure
display_map(img, CLASS_PARAMS, save_path="figures/map.pdf", display=False)

# Random sampling
n_quadrats = 25
quadrat_size = 1 # in meters

sample_points_rdm, samples_rdm = random_sampling_map(img, n_quadrats, quadrat_size)

display_map(img, CLASS_PARAMS,
            sample_points=sample_points_rdm,
            window_size_meters=quadrat_size, pixel_size=0.01,
            figsize=(15, 10), fast_display=False, legend=False,
            save_path="figures/map_w_random_samples.pdf",
            display=False)


# Unconstrained transect sampling
quadrat_size = 1 # in meters
pixel_size = 0.01
n_quadrat_per_transect = 6
distance_between_quadrats = 2.0
n_transects = 10

sample_points_tsct, samples_tsct = free_transect_sampling_map(img, n_transects, n_quadrat_per_transect,
                                                              distance_between_quadrats)

display_map(img, CLASS_PARAMS,
            sample_points=sample_points_tsct,
            window_size_meters=quadrat_size, pixel_size=0.01,
            figsize=(15, 10), fast_display=False, legend=False,
            save_path="figures/map_w_unconstrained_transect_samples.pdf",
            display=False)


# Parallel transect sampling

quadrat_size = 1 # in meters
pixel_size = 0.01
n_quadrat_per_transect = 6
distance_between_quadrats = 2.0
distance_between_transects = 5.0
n_transects = 10

sample_points_prll, samples_prll = parallel_transect_sampling_map(img, n_transects, n_quadrat_per_transect,
                                                                  distance_between_quadrats, distance_between_transects)

display_map(img, CLASS_PARAMS,
            sample_points=sample_points_prll,
            window_size_meters=quadrat_size, pixel_size=0.01,
            figsize=(15, 10), fast_display=False, legend=False,
            save_path="figures/map_w_parallel_transect_samples.pdf",
            display=False)


# Non-directional transect sampling

quadrat_size = 1 # in meters
pixel_size = 0.01
n_quadrat_per_transect = 6
distance_between_quadrats = 2.0
n_transects = 10

sample_points_ndt, samples_ndt = ND_transect_sampling_map(img, n_transects, n_quadrat_per_transect,
                                                            distance_between_quadrats)

display_map(img, CLASS_PARAMS,
            sample_points=sample_points_ndt,
            window_size_meters=quadrat_size, pixel_size=0.01,
            figsize=(15, 10), fast_display=False, legend=False,
            save_path="figures/map_w_ND_transect_samples.pdf",
            display=False)


# Print proportions
true_proportions = compute_proportions(img, n_classes=len(classes) - 1)
proportions_sampling_rdm = compute_proportions(samples_rdm, n_classes=len(classes) - 1)
proportions_sampling_tsct = compute_proportions(samples_tsct, n_classes=len(classes) - 1)
proportions_sampling_prll = compute_proportions(samples_prll, n_classes=len(classes) - 1)
proportions_sampling_ndt = compute_proportions(samples_ndt, n_classes=len(classes) - 1)

display_proportions([true_proportions, proportions_sampling_rdm, proportions_sampling_tsct,
                     proportions_sampling_prll, proportions_sampling_ndt],
                    classes, ["True", "Rand", "Free Tsct", "Prll Tsct", "ND Tsct"])