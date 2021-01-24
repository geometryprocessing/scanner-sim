import itertools
import os
import cv2
import json
import Imath
import OpenEXR
import joblib
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
from utils import *
import meshio
import open3d as o3d
from process import *
from decode import *
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.morphology as morph
from camera import load_camera_calibration
from projector import load_projector_calibration

from scipy.spatial.transform import Rotation as R


def merge(data_path, files, stage_calib, save_as=None, plot=False):
    p0, dir = stage_calib["p"], stage_calib["dir"]

    points = [load_ply(data_path + file) for file in files]
    points = [p - p0 for p in points]

    merged = [points[0]]

    if plot:
        ax = plot_3d(points[0][::1000, :], "Merged", label=str(0))
        line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")

    for i in range(len(points) - 1):
        rot = R.from_rotvec((-(i + 1) * 30 * np.pi / 180) * dir)
        p_rot = rot.apply(points[i+1])
        merged.append(p_rot)

        if plot:
            scatter(ax, p_rot[::10000, :], s=5, label=str(i + 1))

    merged = np.concatenate(merged, axis=0)

    proj = dir[None, :] * np.matmul(merged, dir)[:, None]
    dist = np.linalg.norm(merged - proj, axis=1)
    merged = merged[dist < 120, :]

    merged += p0

    if save_as is not None:
        print("Saving...")
        save_ply(data_path + save_as, merged)

    if plot:
        plt.legend()

    return merged


if __name__ == "__main__":
    data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/"

    stage_calib = numpinize(json.load(open("../calibration/stage/stage_calibration.json", "r")))

    files = ["position_%d/gray/decoded/reconstructed/all_points.ply" % (a*30) for a in range(12)]
    merge(data_path, files, stage_calib, save_as="/pawn_all.ply", plot=True)

    files = ["position_%d/gray/decoded/reconstructed/group_points.ply" % (a*30) for a in range(12)]
    merge(data_path, files, stage_calib, save_as="/pawn_group.ply", plot=True)

    plt.show()
