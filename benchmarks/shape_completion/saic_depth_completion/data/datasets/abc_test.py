import os
import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
import glob
import cv2

# ROOT = '/Vol1/dbstore/datasets/depth_completion/Matterport3D/'
ROOT = "dataset/abc"

class AbcTest:
    def __init__(
            self, root=ROOT, split="train", transforms=None
    ):
        self.transforms = transforms
        self.data_root = root
        self.split = split
        self.color_name, self.depth_name, self.gt_name = [], [], []

        self._load_data()

    def _load_data(self):
        imgs = sorted(glob.glob("%s/%s/*_i_*.png"%(self.data_root, self.split)))
        #print(len(imgs))
        for img in imgs:               
            depth_img = img.replace("_i_", "_d_").replace(".png", ".npy")
            gt_img = img.replace("_i_", "_g_").replace(".png", ".npy")


            self.depth_name.append(depth_img)
            self.gt_name.append(gt_img)
            self.color_name.append(img)
            
        #print(len(self.depth_name), len(self.gt_name), len(self.color_name))

    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        color = np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255.
        color = color[:3]
        #print(color.shape, np.min(color[0]), np.max(color[0]), np.min(color[1]), np.max(color[1]), np.min(color[2]), np.max(color[2]))
        with open(self.gt_name[index], "rb") as fi:
            gt_depth = np.load(fi)
        with open(self.depth_name[index], "rb") as fi:
            depth = np.load(fi)

        mask = np.zeros_like(depth)
        mask[np.where(depth > 0)] = 1
        
        #print(color.shape)

        return  {
            'color':        torch.tensor(color, dtype=torch.float32),
            'raw_depth':    torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
            'mask':         torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'gt_depth':     torch.tensor(gt_depth, dtype=torch.float32).unsqueeze(0),
        }
