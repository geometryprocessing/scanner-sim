import copy
import imageio
import numpy as np
from pathlib import Path
import torch
from typing import Any, Optional, Union

from common import ensure_prefix, instantiate_from_string
from data_io import read_depth, read_color, read_rgbd

class DataField:
    def __init__(self, dir: Union[Path, str], read_fn: Any, pattern: Optional[str] = None, num_files_per_sample: int = 1, reduce_fn: Optional[Any] = None, num_expected: Optional[int] = None, preload: bool = False):
        self.__file_paths = self.__collect_file_paths(Path(dir), pattern=pattern)
        self.__num_files_per_sample = num_files_per_sample
        self.__reduce_fn = reduce_fn
        self.num_samples = len(self.__file_paths) // self.__num_files_per_sample

        if num_expected is not None and self.num_samples != num_expected:
            raise ValueError(f"Expected {num_expected} samples, but actually discovered {self.num_samples}")

        self.__read_fn = read_fn

        self.__data = None

        if preload:
            self.__preload()

    def __collect_file_paths(self, dir: Path, pattern: Optional[str]=None):
        file_paths = [f for f in dir.iterdir() if f.is_file()]

        if pattern is not None:
            file_paths = [f for f in file_paths if f.match(pattern)]

        return file_paths

    def __preload(self):
        self.__data = [self.__read_fn(p) for p in self.__file_paths]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        items = []

        for i in range(self.__num_files_per_sample):
            item_idx = idx * self.__num_files_per_sample + i

            if self.__data is not None:
                items.append(self.__data[item_idx])
            else:
                items.append(self.__read_fn(self.__file_paths[item_idx]))

        if len(items) == 1:
            return items[0]
        else:
            if self.__reduce_fn is not None:
                items = self.__reduce_fn(items)

            return items

class SLSDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir: Union[Path, str], 
                 depth_dir_name: str = '.',
                 depth_file_pattern: Optional[str] = '*depth_rec*',
                 color_dir_name: Optional[str] = None,
                 color_file_pattern: Optional[str] = None,
                 ambient_dir_name: Optional[str] = None,
                 ambient_file_pattern: Optional[str] = None,
                 target_dir_name: Optional[str] = '.',
                 target_file_pattern: Optional[str] = '*depth_gt*',
                 preload: bool = True,
                 transform: Optional[Union[Any, str]] = "CropImage(256)",
                 repetitions: int = 1):
        super().__init__()

        self.__base_dir = Path(base_dir)

        self.__depths = DataField(self.__base_dir / depth_dir_name, read_depth, pattern=depth_file_pattern, preload=preload)

        # Optional: Color input
        self.__colors = None
        if color_dir_name is not None:
            self.__colors = DataField(self.__base_dir / color_dir_name, read_color, pattern=color_file_pattern, preload=preload, num_expected=len(self.__depths))

        self.__ambient = None
        if ambient_dir_name is not None:
            self.__ambient = DataField(self.__base_dir / ambient_dir_name, read_color, pattern=ambient_file_pattern, preload=preload, num_expected=len(self.__depths))

        # Optional: Handle targets for training
        self.__targets = None
        if target_dir_name is not None:
            self.__targets = DataField(self.__base_dir / target_dir_name, read_depth, pattern=target_file_pattern, preload=preload, num_expected=len(self.__depths))

        self.transform = transform
        if isinstance(self.transform, str):
            # Instantiate transform from a class string
            if self.transform.lower() == "none":
                self.transform = None
            else:
                print(f"Notice: Dataset transformation is {self.transform}. If you meant to not apply a transform, pass None to override the default.")
                self.transform = instantiate_from_string(ensure_prefix('data.', self.transform))

        self.repetitions = repetitions

    def has_color(self):
        return self.__colors is not None

    def has_ambient(self):
        return self.__ambient is not None

    def __len__(self):
        return len(self.__depths) * self.repetitions

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx // self.repetitions

        sample = {
            'depth': self.__depths[idx]
        }

        if self.__colors is not None:
            sample['color'] = self.__colors[idx]

        if self.__ambient is not None:
            sample['ambient'] = self.__ambient[idx]

        if self.__targets is not None:
            sample['target'] = self.__targets[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class CropImage(object):
    """ Transformation that crops a square with (random) position from the images.

    Args:
        image_size: Width and height of the cropped square.
        anchor_x (optional): X coordinate of the top left corner of the square in the source image. 
                             Default is None, so it is randomly sampled.
        anchor_y (optional): Y coordinate of the top left corner of the square in the source image. 
                             Default is None, so it is randomly sampled.
    """

    def __init__(self, image_size: int, anchor_x: Optional[int] = None, anchor_y: Optional[int] = None):
        self.image_size = image_size
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y

    def __call__(self, sample):
        height, width, _ = sample['depth'].shape

        crop_anchor_y = np.random.randint(0, height - self.image_size + 1) if self.anchor_y is None else self.anchor_y
        crop_anchor_x = np.random.randint(0, width - self.image_size + 1) if self.anchor_x is None else self.anchor_x

        for (k, v) in sample.items():
            sample[k] = v[crop_anchor_y:crop_anchor_y+self.image_size, crop_anchor_x:crop_anchor_x+self.image_size]

        return sample

if __name__ == '__main__':
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    from torchvision.transforms import Compose

    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    dataset = SLSDataset(**{ "base_dir": args.input_dir, "transform": "CropImage(256)" })

    depth = dataset[0]['depth']
    mask = depth > 0
    min_depth = depth[mask].min() if np.any(mask) else 0
    max_depth = depth[mask].max() if np.any(mask) else 0

    fig, ax = plt.subplots()
    ax.imshow(depth[:, :, 0], vmin=min_depth, vmax=max_depth)
    plt.show()