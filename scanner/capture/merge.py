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


def merge_single_30_deg(data_path, filename_template, stage_calib, max_dist=100, title="Merged", skip=1000, plot=False, **kw):
    files = [data_path + filename_template % (a * 30) for a in range(12)]
    points = [load_ply(file) for file in files]

    p0, dir = stage_calib["p"], stage_calib["dir"]
    points = [p - p0 for p in points]
    merged = [points[0]]

    if plot:
        ax = plot_3d(points[0][::skip, :], title, label=str(0) + " deg")
        # line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")

    for i in range(len(points) - 1):
        rot = R.from_rotvec((-(i + 1) * 30 * np.pi / 180) * dir)
        p_rot = rot.apply(points[i+1])
        merged.append(p_rot)

        if plot:
            scatter(ax, p_rot[::skip, :], s=5, label=str(30*(i + 1)) + " deg")

    merged = np.concatenate(merged, axis=0)
    proj = dir[None, :] * np.matmul(merged, dir)[:, None]
    dist = np.linalg.norm(merged - proj, axis=1)
    merged = merged[dist < max_dist, :] + p0

    if plot:
        plt.legend()

    return merged


def merge_both_30_deg(data_path, object_name, stage_calib, save=True, plot=False, save_figures=True, **kw):
    if save:
        save_path = data_path + "/reconstructed/"
        ensure_exists(save_path)

    for suffix, skip in zip(["all", "group"], [10000, 1000]):
        print("Merging object: %s (%s)" % (object_name, suffix))
        merged = merge_single_30_deg(data_path, "/position_%s/gray/reconstructed/%s_points.ply" % ("%d", suffix),
                                     stage_calib, title=object_name + "_" + suffix, plot=plot, **kw)
        if save:
            filename = object_name + "_%s.ply" % suffix
            print("Saving", filename)
            save_ply(save_path + filename, merged)

            if plot and save_figures:
                plt.savefig(save_path + object_name + "_%s.png" % suffix, dpi=160)


if __name__ == "__main__":
    stage_calib = numpinize(json.load(open("../calibration/stage/stage_calibration.json", "r")))

    # Debug
    # merge_both_30_deg("D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/", "pawn", stage_calib, plot=True)

    # data_path_template = "D:/scanner_sim/captures/stage_batch_2/no_ambient/%s_30_deg/"
    # for object_name in ["pawn", "rook", "shapes"]:
    #     merge_both_30_deg(data_path_template % object_name, object_name, stage_calib, plot=True)

    data_path_template = "D:/scanner_sim/captures/stage_batch_2/%s_30_deg/"
    for object_name in ["dodo", "avocado", "house", "chair", "vase", "bird", "radio"]:
        merge_both_30_deg(data_path_template % object_name, object_name, stage_calib, max_dist=70, plot=True)

    plt.show()
