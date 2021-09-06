import copy
import math
from numba import cuda
import numpy as np

from baselines import bilateral_depth_filter
from data import SLSDataset

def visualize(dataset, args):
    import matplotlib.pyplot as plt
    
    print(f"Visualizing sample {args.sample_index}")
    print(f"-- sigma spatial {args.sigma_spatial}")
    print(f"-- sigma depth {args.sigma_depth}")
    print(f"-- error threshold {args.error_threshold}")

    data = dataset[args.sample_index]

    depth_groundtruth = copy.deepcopy(data['target'])
    depth_reconstructed = copy.deepcopy(data['depth'])
    mask_groundtruth = depth_groundtruth > 0
    mask_reconstructed = depth_reconstructed > 0

    depth_denoised = bilateral_depth_filter(depth_reconstructed, sigma_spatial=args.sigma_spatial, sigma_depth=args.sigma_depth)

    min_depth = depth_groundtruth[mask_groundtruth].min()
    max_depth = depth_groundtruth[mask_groundtruth].max()

    error_reconstructed = depth_reconstructed - depth_groundtruth
    error_reconstructed[~mask_reconstructed] = 0

    error_denoised = depth_denoised - depth_groundtruth
    error_denoised[~mask_reconstructed] = 0

    error_threshold = args.error_threshold
    error_min = -error_threshold
    error_max = error_threshold

    fig, axs = plt.subplots(1, 3, figsize=(25, 10), constrained_layout=True)
    axs[0].imshow(error_reconstructed[:, :, 0], vmin=error_min, vmax=error_max, cmap='bwr')
    axs[1].imshow(error_denoised[:, :, 0], vmin=error_min, vmax=error_max, cmap='bwr')

    bins = np.linspace(-error_threshold, error_threshold, 100)
    axs[2].hist(error_reconstructed[mask_reconstructed & (np.abs(error_reconstructed) < error_threshold)].ravel(), bins=bins, alpha=0.5, label='Reconstructed')
    axs[2].hist(error_denoised[mask_reconstructed & (np.abs(error_denoised) < error_threshold)].ravel(), bins=bins, alpha=0.5, label='Denoised')
    axs[2].set_xlabel('Meters')
    axs[2].axvline(x=0, color='green', linewidth=0.25)
    axs[2].legend()

    plt.show()

def get_parameter_values(values, num_steps):
    if num_steps > 0:
        # With valid number of steps we construct a range from the first two values
        values = np.linspace(values[0], values[1], num=num_steps)
    return values

def grid_search(dataset, args):
    sigma_spatial_values = get_parameter_values(args.sigma_spatial_values, args.sigma_spatial_steps)
    sigma_depth_values = get_parameter_values(args.sigma_depth_values, args.sigma_depth_steps)

    min_loss = None
    min_parameters = None
    for sigma_spatial in sigma_spatial_values:
        for sigma_depth in sigma_depth_values:
            loss = 0.0
            for data in dataset:
                depth_groundtruth = data['target']
                depth_reconstructed = data['depth']
                mask_groundtruth = depth_groundtruth > 0
                mask_reconstructed = depth_reconstructed > 0

                depth_denoised = bilateral_depth_filter(depth_reconstructed, sigma_spatial=sigma_spatial, sigma_depth=sigma_depth)

                error_denoised = depth_denoised - depth_groundtruth
                loss += np.abs(error_denoised[mask_reconstructed & mask_groundtruth]).mean()

            loss /= len(dataset)

            if min_loss is None or loss < min_loss:
                min_loss = loss
                min_parameters = (sigma_spatial, sigma_depth)
                print(f"Found new parameters {min_parameters} with loss {min_loss}")

    print(f"Grid search finished. Loss: {min_loss}. Parameters: {min_parameters}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    import time

    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--mode", type=str, default='visualize', choices=['visualize', 'grid_search'])

    # Arguments for visualization
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sigma_spatial", type=float, default=5)
    parser.add_argument("--sigma_depth", type=float, default=0.002)
    parser.add_argument("--error_threshold", type=float, default=0.001)

    # Arguments for grid search
    parser.add_argument("--sigma_spatial_values", type=float, nargs="+", default=[0.5, 5])
    parser.add_argument("--sigma_spatial_steps", type=int, default=10)
    parser.add_argument("--sigma_depth_values", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--sigma_depth_steps", type=int, default=10)

    args = parser.parse_args()

    dataset = SLSDataset(args.input_dir, 
                         depth_dir_name='.', depth_file_pattern='*_depth_rec*',
                         target_dir_name='.', target_file_pattern='*_depth_gt*',
                         transform=None)

    if args.mode == 'visualize':
        visualize(dataset, args)
    elif args.mode == 'grid_search':
        grid_search(dataset, args)
    else:
        raise RuntimeError("Invalid mode")