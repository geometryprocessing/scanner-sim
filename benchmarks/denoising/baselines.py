import copy
import math
from numba import cuda
import numpy as np

from common import unproject

@cuda.jit
def bilateral_depth_filter_kernel(depth_in, depth_out, sigma_spatial, sigma_depth, window_radius, width, height):
    position_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    position_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    grid_stride_y = cuda.gridDim.y * cuda.blockDim.y
    grid_stride_x = cuda.gridDim.x * cuda.blockDim.x

    for i in range(position_y, height, grid_stride_y):
        for j in range(position_x, width, grid_stride_x):
            depth = depth_in[i, j, 0]

            # TODO: Use mask?

            depth_sum = 0.0
            weight_sum = 0.0
            for ii_unclamped in range(i - window_radius, i + window_radius + 1):
                for jj_unclamped in range(j - window_radius, j + window_radius + 1):
                    ii = max(0, min(ii_unclamped, height - 1))
                    jj = max(0, min(jj_unclamped, width - 1))

                    ddepth = depth_in[ii, jj, 0]

                    spatial_distance = math.sqrt(float(i - ii)**2 + float(j - jj)**2)
                    spatial_weight = math.exp(-spatial_distance**2 / (2 * sigma_spatial**2))

                    depth_distance = abs(depth - ddepth)
                    depth_weight = math.exp(-depth_distance**2 / (2 * sigma_depth**2))

                    weight = spatial_weight * depth_weight

                    depth_sum += weight * ddepth
                    weight_sum += weight

            depth_out[i, j, 0] = depth_sum / weight_sum

def bilateral_depth_filter(depth, sigma_spatial: float, sigma_depth: float):
    """ Bilateral depth filter.
    """

    depth_in = cuda.to_device(copy.deepcopy(depth))
    depth_out = np.zeros_like(depth)

    width = depth.shape[1]
    height = depth.shape[0]
    window_radius = 2 * sigma_spatial

    threads_per_block = (16, 16)
    blocks_per_grid = (40, 40)#(math.ceil(height / threads_per_block[0]), math.ceil(width / threads_per_block[1]))
    bilateral_depth_filter_kernel[blocks_per_grid, threads_per_block](depth_in, depth_out, sigma_spatial, sigma_depth, window_radius, width, height)

    return depth_out

def laplace_depth_filter(depth, K, R, t):
    """ Meshlab's triangle mesh-based depth filter.
    """
    import pymeshlab

    X = unproject(depth, K, R, t, depth_is_distance=True)
    mask = depth > 0

    # Mesh the depth map, connecting neighboring pixels with valid depths
    indices_1 = np.arange(X.shape[0]*X.shape[1]).reshape(X.shape[0], X.shape[1])
    indices_1 = indices_1[:-1, :-1]
    indices_2 = indices_1 + X.shape[1]
    indices_3 = indices_1 + 1
    indices_4 = indices_2 + 1

    i1 = np.stack([indices_1, indices_2, indices_3], axis=-1)
    i1_mask = (mask & np.roll(mask, -1, axis=0) & np.roll(mask, -1, axis=1))[:-1, :-1, 0]

    i2 = np.stack([indices_2, indices_4, indices_3], axis=-1)
    i2_mask = (np.roll(mask, -1, axis=0) & np.roll(mask, (-1, -1), axis=(0, 1)) & np.roll(mask, -1, axis=1))[:-1, :-1, 0]

    faces = np.concatenate([i1[i1_mask].reshape(-1, 3), i2[i2_mask].reshape(-1, 3)], axis=0)

    mesh = pymeshlab.Mesh(X.reshape(-1, 3), faces)

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(mesh, "mesh")

    mesh_set.apply_filter("depth_smooth")

    mesh_smooth = mesh_set.current_mesh()
    X_smooth = mesh_smooth.vertex_matrix()
    
    return np.linalg.norm(X_smooth, axis=-1).reshape(depth.shape)
