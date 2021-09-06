import imageio
import numpy as np
from pathlib import Path
from typing import Any

def get_color_from_numpy(data: Any):
    # Unpack color from a dict or multi-channel RGBD array
    if isinstance(data, dict):
        return data['color']
    elif len(data.shape) == 3 and data.shape[2] == 4:
        return data[:, :, :3]
    else:
        return data

def get_depth_from_numpy(data: Any):
    # Unpack depth from a dict or multi-channel RGBD array
    if isinstance(data, dict):
        return data['depth']
    elif len(data.shape) == 3 and data.shape[2] == 4:
        # Multi-channel array
        return data[:, :, 3:4]
    elif len(data.shape) == 3 and data.shape[2] == 1:
        return data[:, :, :1]
    elif len(data.shape) == 2:
        return data[:, :, np.newaxis]
    else:
        return data

def sanitize_color(color: Any):
    # Make sure that grayscale images have at least three color channels
    if len(color.shape) == 2:
        color = color[:, :, None]

    if color.shape[2] == 1:
        color = np.repeat(color, 3, axis=-1)

    # Retain RGB channels
    color = color[:, :, :3]

    # Normalize the image depending on the data type
    norm_factor = 1.0
    if color.dtype == np.uint8:
        norm_factor = 1.0 / 255.0

    # TODO: What about HDR input?

    return color.astype(np.float32) * norm_factor

def sanitize_depth(depth: Any):
    if len(depth.shape) == 2:
        depth = depth[:, :, None]

    return depth

def read_color(file_path: Path):
    if file_path.suffix == '.npy':
        color = get_color_from_numpy(np.load(file_path))
    else:
        color = np.array(imageio.imread(file_path))

    return sanitize_color(color)

def read_depth(file_path: Path):
    if file_path.suffix == '.npy':
        depth = get_depth_from_numpy(np.load(file_path))
    elif file_path.suffix == '.exr':
        raise NotImplementedError("EXR loading not implemented")
    else:
        depth = np.array(imageio.imread(file_path))

        if depth.dtype != np.uint16:
            raise ValueError("Expected 16 bit depth image")

        # Convert depth to meters
        depth = depth.astype(np.float32) / 1000.0

    return sanitize_depth(depth)

def read_rgbd(file_path: Path):
    result = {}
    
    if file_path.suffix == '.npy':
        data = np.load(file_path)

        result['depth'] = sanitize_depth(get_depth_from_numpy(data))
        result['color'] = sanitize_color(get_color_from_numpy(data))
    elif file_path.suffix == '.exr':
        raise NotImplementedError("EXR loading not implemented")
    
    return result