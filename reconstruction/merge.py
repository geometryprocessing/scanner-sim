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
from scipy.spatial.transform import Rotation as R


def merge_single_30_deg(data_path, filename_template, stage_calib, max_dist=100, title="Merged", skip=1000, plot=False, sim=False, max_range=12, **kw):
    if sim:
        files = [data_path + filename_template % a for a in range(max_range)]
    else:
        files = [data_path + filename_template % (a * 30) for a in range(max_range)]
      
    points = []
    normals = []
    colors = []
    
    for fi in files:
        poi, nor, col = load_ply(fi)
        points.append(poi)
        normals.append(nor)
        colors.append(col)
   
    p0, dir = stage_calib["p"], stage_calib["dir"]
    points = [p - p0 for p in points]
    merged = [points[0]]
    merged_normals = [normals[0]]

    if plot:
        ax = plot_3d(points[0][::skip, :], title, label=str(0) + " deg")
        # line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")

    angle = int(360/max_range)
    for i in range(len(points) - 1):
        rot = R.from_rotvec((-(i + 1) * angle * np.pi / 180) * dir)
        p_rot = rot.apply(points[i+1])
        n_rot = rot.apply(normals[i+1])
        merged.append(p_rot)
        merged_normals.append(n_rot)

        if plot:
            scatter(ax, p_rot[::skip, :], s=5, label=str(30*(i + 1)) + " deg")

    merged = np.concatenate(merged, axis=0)
    m_normals = np.concatenate(merged_normals, axis=0)
    colors = np.concatenate(colors, axis=0)
    proj = dir[None, :] * np.matmul(merged, dir)[:, None]
    dist = np.linalg.norm(merged - proj, axis=1)
    merged = merged[dist < max_dist, :] + p0
    #print(merged.shape, colors.shape, m_normals.shape, len(merged_normals))
    m_normals = m_normals[dist < max_dist, :]
    colors = colors[dist < max_dist, :]

    if plot:
        plt.legend()

    return merged, m_normals, colors


def merge_both_30_deg(data_path, object_name, stage_calib, save=True, plot=False, save_figures=True, sim=False, **kw):
    if save:
        save_path = data_path + "/reconstructed/"
        ensure_exists(save_path)

    for suffix, skip in zip(["all", "group"], [10000, 1000]):
        print("Merging object: %s (%s)" % (object_name, suffix))

        if sim:
            merged, normals, colors = merge_single(data_path, "/rot_%s/reconstructed/%s_points.ply" % ("%03i", suffix),
                                     stage_calib, title=object_name + "_" + suffix, plot=plot, sim=sim, **kw)
        else:
            merged, normals, colors = merge_single_30_deg(data_path, "/position_%s/gray/reconstructed/%s_points.ply" % ("%d", suffix),
                                     stage_calib, title=object_name + "_" + suffix, plot=plot, sim=sim, **kw)
        if save:
            filename = object_name + "_%s.ply" % suffix
            print("Saving", filename)
            save_ply(save_path + filename, merged, normals, colors)

            if plot and save_figures:
                plt.savefig(save_path + object_name + "_%s.png" % suffix, dpi=160)


if __name__ == "__main__":
    # stage_calib = load_calibration("../calibration/stage/stage_geometry.json")
    stage_calib = load_calibration("D:/scanner_sim/captures/stage_batch_3/stage_calib_2_deg_before/merged/stage/stage_geometry.json")

    # Debug
    # merge_both_30_deg("D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/", "pawn", stage_calib, plot=True)

    data_path_template = "D:/scanner_sim/captures/stage_batch_3/pawn_30_deg_%s/"
    for object_name in ["matte", "gloss"]:
        merge_both_30_deg(data_path_template % object_name, object_name, stage_calib, plot=True)

    # data_path_template = "D:/scanner_sim/captures/stage_batch_2/%s_30_deg/"
    # for object_name in ["pawn", "rook", "shapes"]:
    #     merge_both_30_deg(data_path_template % object_name, object_name, stage_calib, plot=True)

    # data_path_template = "D:/scanner_sim/captures/stage_batch_2/no_ambient/%s_30_deg/"
    # for object_name in ["pawn", "rook", "shapes"]:
    #     merge_both_30_deg(data_path_template % object_name, object_name, stage_calib, plot=True)

    # data_path_template = "D:/scanner_sim/captures/stage_batch_2/%s_30_deg/"
    # for object_name in ["dodo", "avocado", "house", "chair", "vase", "bird", "radio"]:
    #     merge_both_30_deg(data_path_template % object_name, object_name, stage_calib, max_dist=70, plot=True)

    plt.show()
