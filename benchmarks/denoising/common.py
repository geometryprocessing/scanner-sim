import copy
import json
import math
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Any, Optional, Union

def hwc_to_chw(tensor):
    if len(tensor.shape) == 3:
        return tensor.permute(2, 0, 1)
    elif len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported dimension {len(tensor.shape)}")

def chw_to_hwc(tensor):
    if len(tensor.shape) == 3:
        return tensor.permute(1, 2, 0)
    elif len(tensor.shape) == 4:
        return tensor.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unsupported dimension {len(tensor.shape)}")

def crop(image, roi):
    return image[roi[0]:roi[1], roi[2]:roi[3]]

def load_config(path: Union[str, Path]):
    with open(path) as file:
        return json.load(file)

def load_network(path, device):
    from persistence import Checkpoint
    from networks import create_network

    checkpoint = Checkpoint.load(path, map_location=device)

    config = {}
    checkpoint.restore_config(config)

    num_channels = 1

    if 'color_dir_name' in config['input']['train']:
        num_channels += 3

    if 'ambient_dir_name' in config['input']['train']:
        num_channels += 3

    network = create_network(num_channels, 1, config=config['network'])
    network = network.to(device)
    checkpoint.restore_model(network)

    return network, checkpoint, config

def predict(network, sample, device, tiler=None):
    input = torch.tensor(sample['depth'])

    if 'color' in sample:
        input = torch.cat([input, torch.tensor(sample['color'])], dim=-1)

    if 'ambient' in sample:
        input = torch.cat([input, torch.tensor(sample['ambient'])], dim=-1)

    if not tiler:
        tiler = Tiler(256, overlap=0.94, margin=100) 

    tiles = tiler.split(input)

    with torch.no_grad():
        for chunk in tqdm(torch.split(tiles, 16), leave=False):
            chunk[...] = chw_to_hwc(network(hwc_to_chw(chunk).to(device)).cpu())

    depth_predicted = tiler.join(sample['depth'].shape, tiles)
    depth_predicted[sample['depth'] == 0] = 0

    return depth_predicted

def get_new_version(experiment_directory: Path):
    if not experiment_directory.exists():
        return 0

    version_directories = [e for e in experiment_directory.iterdir() if e.is_dir()]
    if len(version_directories) == 0:
        return 0

    # Get an increment to the latest version number
    versions = [int(str(d.name).split('_')[1]) for d in version_directories]
    versions = sorted(versions)

    return versions[-1] + 1

def get_output_directory(version: Optional[int] = None):
    output_directory = Path('./out')

    if version is None:
        version = get_new_version(output_directory)

    return output_directory / f'version_{version}'

def nanmean(tensor, *args, inplace=False, **kwargs):
    # Adapted from https://github.com/yulkang/pylabyk
    
    if not inplace:
        tensor = tensor.clone()

    nan_mask = torch.isnan(tensor)
    tensor[nan_mask] = 0

    return tensor.sum(*args, **kwargs) / (~nan_mask).float().sum(*args, **kwargs)

class Tiler:
    def __init__(self, tile_size:int = 0, overlap: float = 0.0, margin: int = 0):
        self.tile_size = tile_size
        self.overlap = overlap
        self.margin = margin

    def __get_spacing(self):
        spacing = int(self.tile_size * (1.0 - self.overlap))
        return spacing, spacing

    def get_max_sample_count(self):
        active_tile_size = self.tile_size - 2 * self.margin
        delta_tile_size = (self.tile_size * (1 - self.overlap))
        return int(math.ceil(active_tile_size / delta_tile_size))**2

    def split(self, image):
        height, width, num_channels = image.shape
        tile_spacing_x, tile_spacing_y = self.__get_spacing()
        num_tiles_x = (width + tile_spacing_x - 1) // tile_spacing_x
        num_tiles_y = (height + tile_spacing_y - 1) // tile_spacing_y

        tiles = []
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                ii = i * tile_spacing_y
                jj = j * tile_spacing_x

                roi = (ii, min(ii + self.tile_size, height), jj, min(jj + self.tile_size, width))

                tile = torch.zeros((self.tile_size, self.tile_size, num_channels))
                tile[:roi[1] - roi[0], :roi[3] - roi[2]] = image[roi[0]:roi[1], roi[2]:roi[3]]

                tiles.append(tile)
        
        return torch.stack(tiles, dim=0)

    def join(self, shape, tiles, method: str='mean'):
        max_sample_count = self.get_max_sample_count()
        # print(f"Tiler: Max sample count in join operation is {max_sample_count}")

        height, width, num_channels = shape
        
        tile_spacing_x, tile_spacing_y = self.__get_spacing()
        num_tiles_x = (width + tile_spacing_x - 1) // tile_spacing_x
        num_tiles_y = (height + tile_spacing_y - 1) // tile_spacing_y

        sample_buffer = torch.full((max_sample_count, height, width, num_channels), float('nan'), dtype=tiles[0].dtype, device=tiles[0].device)
        sample_counts = torch.zeros(height, width, num_channels).long()

        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                ii = i * tile_spacing_y
                jj = j * tile_spacing_x

                tile_index = i * num_tiles_x + j
                
                roi = torch.tensor([
                    ii + self.margin, ii + self.margin + self.tile_size - (2 * self.margin), 
                    jj + self.margin, jj + self.margin + self.tile_size - (2 * self.margin)
                ], dtype=torch.int32)

                roi[:2] = torch.clamp(roi[:2], min=0, max=height)
                roi[2:] = torch.clamp(roi[2:], min=0, max=width)

                roi_height = roi[1] - roi[0]
                roi_width = roi[3] - roi[2]
                if roi_height == 0 or roi_width == 0:
                    continue

                roi_tile = torch.tensor([
                    self.margin, self.margin + roi_height, 
                    self.margin, self.margin + roi_width
                ], dtype=torch.int32)

                sample_indices = sample_counts[roi[0]:roi[1], roi[2]:roi[3]]
                tile_values = tiles[tile_index][roi_tile[0]:roi_tile[1], roi_tile[2]:roi_tile[3]]
                sample_buffer[:, roi[0]:roi[1], roi[2]:roi[3]].scatter_(0, sample_indices.unsqueeze(0), tile_values.unsqueeze(0))
                sample_counts[roi[0]:roi[1], roi[2]:roi[3]] += 1

        if method == 'mean':
            result = nanmean(sample_buffer, dim=0)
        elif method== 'median':
            result = torch.from_numpy(np.nanmedian(sample_buffer.numpy(), axis=0))

        result[torch.isnan(result)] = 0.0

        return result

def unproject(depth, K, R=np.eye(3), t=np.zeros(3), depth_is_distance=True):
    """ Unproject a depth map into 3d space

   Args:
        depth (tensor): depth map of size H x W (x 1)
        K (tensor): intrinsic matrix of size 3 x 3
        R (tensor): rotation matrix of size 3 x 3
        t (tensor): translation of size 3 (x 1)
        depth_is_distance (bool): indicator whether the depth is the distance from the camera origin 
                                  or the Z coordinate in camera coordinates
    """

    is_torch = torch.is_tensor(depth)

    depth = np.array(depth)
    K = np.array(K)
    K_inv = np.linalg.inv(K)
    R = np.array(R)
    t = np.array(t)

    if len(t.shape) == 1:
        t = t[:, np.newaxis]

    height, width = depth.shape[:2]

    xs = np.arange(width)
    ys = np.arange(height)
    pixels = np.stack(np.meshgrid(xs, ys) + [np.ones((height, width))], axis=-1).astype(np.float32)

    # Unproject the pixel x in homogeneous coordinates
    # according to the equation X = R^t (K^(-1) * Z * x - t)
    pixels = pixels.reshape(-1, 3).transpose() # From H x W x 3 to 3 x N
    depth = depth.reshape(-1, 1).transpose()   # From H x W (x 1) to 1 x N

    points = K_inv @ pixels

    if depth_is_distance:
        # "Depth" is not the Z coordinate of the points but the euclidean distance from the camera origin
        points /= np.linalg.norm(points, axis=0)

    points = (R.transpose() @ (depth * points - t))
    
    points = points.transpose().reshape(height, width, 3)   # From 3 x N to H x W x 3

    return torch.from_numpy(points) if is_torch else points

def ensure_prefix(prefix: str, string: str):
    if not string.startswith(prefix):
        string = prefix + string
    return string

def instantiate_from_string(class_string: str):
    """ Instantiate a class from a string.

    The class string must be the full quantified class name with module path,
    i.e., example.module.Class or example.module.Class(). Parameters are
    also correctly handled and can be passed like example.module.Class(1, 2, param="hello").
    
    Args:
        class_string: Class name with module path and optionally parameters.

    Returns:
        Instantiated class object.
    """

    from importlib import import_module

    try:
        if class_string.find('(') == -1:
            # Assume default constructed objects if no parameters are given
            class_string += "()"

        class_string_parts = class_string.rsplit('.', 1)

        context = {}
        if len(class_string_parts) == 2:
            # We have to import a module to instantiate the class
            module_path = class_string_parts[0]
            context[module_path] = import_module(module_path)

        return eval(class_string, context)
    except (ImportError, AttributeError) as e:
        raise ImportError(class_str)

def export_point_cloud(path, depth, K, R, t, colors=None):
    from pyntcloud import PyntCloud
    from pandas import DataFrame
    
    X = unproject(depth, K, R, t, depth_is_distance=True)

    # The z axis points away from the camera, so we have to invert the normals
    # to make them point towards the camera
    normals = -estimate_normals(X)

    mask = (depth > 0)[:, :, 0]

    df = DataFrame({
        'x': X[:, :, 0][mask].ravel(),
        'y': X[:, :, 1][mask].ravel(),
        'z': X[:, :, 2][mask].ravel(),
        'nx': normals[:, :, 0][mask].ravel(),
        'ny': normals[:, :, 1][mask].ravel(),
        'nz': normals[:, :, 2][mask].ravel(),
    })

    if colors is not None:
        df['red'] = colors[:, :, 0][mask].ravel()
        df['green'] = colors[:, :, 1][mask].ravel()
        df['blue'] = colors[:, :, 2][mask].ravel()

    pc = PyntCloud(df)
    pc.to_file(path)

def estimate_normals(points, normalize=True):
    dx = np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)
    dy = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normals = np.cross(dx, dy)

    if normalize:
        length = np.linalg.norm(normals, axis=-1, keepdims=True)
        mask = (length > 0)[..., 0]
        normals[mask] = normals[mask] / length[mask]

    return normals

def draw_roi(image, roi, thinkness=10, color = [1.0, 0, 0]):
    image_out = copy.deepcopy(image)
    image_roi = crop(image_out, roi)
    image_roi[:, :] = color if image.shape[2] == 3 else (color + [1.0])

    roi_inner = [roi[0] + thinkness, roi[1] - thinkness, roi[2] + thinkness, roi[3] - thinkness]
    image_roi_inner = crop(image_out, roi_inner)
    image_roi_inner[:, :] = crop(image[:, :], roi_inner)

    return image_out