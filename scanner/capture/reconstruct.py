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
from process import *
from decode import *
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.morphology as morph
from camera import load_camera_calibration
from projector import load_projector_calibration
from skimage import measure
from skimage import filters


def img_to_ray(p_img, mtx):
    p_img = (p_img - mtx[:2, 2]) / mtx[[0, 1], [0, 1]]
    return np.concatenate([p_img, np.ones((p_img.shape[0], 1))], axis=1)


def triangulate(cam_rays, proj_xy, proj_calib):
    u_proj_xy = cv2.undistortPoints(proj_xy.astype(np.float), proj_calib["mtx"], proj_calib["dist"]).reshape((-1, 2))
    proj_rays = np.concatenate([u_proj_xy, np.ones((u_proj_xy.shape[0], 1))], axis=1)
    proj_rays = np.matmul(proj_calib["basis"].T, proj_rays.T).T
    proj_origin = proj_calib["origin"]

    v12 = np.sum(np.multiply(cam_rays, proj_rays), axis=1)
    v1, v2 = np.linalg.norm(cam_rays, axis=1)**2, np.linalg.norm(proj_rays, axis=1)**2
    L = (np.matmul(cam_rays, proj_origin) * v2 + np.matmul(proj_rays, -proj_origin) * v12) / (v1 * v2 - v12**2)

    return cam_rays * L[:, None]


def reconstruct_single(data_path, cam_calib, proj_calib, out_dir="reconstructed", max_group=25, gen_depth_map=True,
                       save=True, plot=False, save_figures=True, verbose=False, **kw):
    if save:
        save_path = data_path + out_dir + "/"
        ensure_exists(save_path)
        data_path += "decoded/"
    else:
        save_path = None

    all, groups = load_decoded(data_path)
    cam_xy, proj_xy, mask = all
    if groups:
        group_cam_xy, group_proj_xy, group_counts, group_rcs = groups
    undistorted = bool(open(data_path + "undistorted.txt", "r").read())
    print("Loaded:", data_path)

    if undistorted:
        u_cam_xy = cv2.undistortPoints(cam_xy.astype(np.float), cam_calib["new_mtx"], None).reshape((-1, 2))
        if groups:
            u_group_cam_xy = cv2.undistortPoints(group_cam_xy.astype(np.float), cam_calib["new_mtx"], None).reshape((-1, 2))
    else:
        u_cam_xy = cv2.undistortPoints(cam_xy.astype(np.float), cam_calib["mtx"], cam_calib["dist"]).reshape((-1, 2))
        if groups:
            u_group_cam_xy = cv2.undistortPoints(group_cam_xy.astype(np.float), cam_calib["mtx"], cam_calib["dist"]).reshape((-1, 2))

    cam_rays = np.concatenate([u_cam_xy, np.ones((u_cam_xy.shape[0], 1))], axis=1)
    if groups:
        group_cam_rays = np.concatenate([u_group_cam_xy, np.ones((u_group_cam_xy.shape[0], 1))], axis=1)

    print("Triangulating...")

    all_points = triangulate(cam_rays, proj_xy, proj_calib)
    if groups:
        group_points = triangulate(group_cam_rays, group_proj_xy, proj_calib)
        idx = np.nonzero(group_counts < max_group)[0]
        print("%d groups larger than %d excluded" % (group_counts.shape[0] - idx.shape[0], max_group))
        group_points = group_points[idx, :]

    if gen_depth_map:
        print("Generating depth map(s)")
        full_depth_map = np.zeros_like(mask, dtype=np.float32)
        full_depth_map[cam_xy[:, 1], cam_xy[:, 0]] = np.linalg.norm(all_points, axis=1)

        if groups:
            group_depth_map = np.zeros_like(mask, dtype=np.float32)
            for i, id in enumerate(idx):
                rcs = group_rcs[id]
                if i % 100000 == 0 and verbose:
                    print("Group", i)
                group_depth_map[rcs[:, 0], rcs[:, 1]] = np.linalg.norm(group_points[i, :])

    if save:
        save_ply(save_path + "all_points.ply", all_points)
        if groups:
            save_ply(save_path + "group_points.ply", group_points)
            with open(save_path + "max_group_size.txt", "w") as f:
                f.write(str(max_group))

        if gen_depth_map:
            np.save(save_path + "full_depth_map.npy", full_depth_map.astype(np.float32))
            if groups:
                np.save(save_path + "group_depth_map.npy", group_depth_map.astype(np.float32))

    if plot:
        if not save_figures:
            save_path = None

        plot_3d(all_points[::1000, :], "All Points", save_as=save_path + "all_points" if save_path else None)
        if groups:
            plot_3d(group_points[::100, :], "Group Points", save_as=save_path + "group_points" if save_path else None)

        if gen_depth_map:
            vmin = np.min(full_depth_map[full_depth_map > 1])
            plot_image(full_depth_map, "Full Depth Map", data_path + " - Full Depth Map", vmin=vmin, save_as=save_path + "full_depth_map" if save_path else None)
            if groups:
                vmin = np.min(full_depth_map[group_depth_map > 1])
                plot_image(group_depth_map, "Group Depth Map", data_path + " - Group Depth Map", vmin=vmin, save_as=save_path + "group_depth_map" if save_path else None)

    return all_points, group_points if groups else None


def reconstruct_many(path_template, cam_calib, proj_calib, suffix="gray/", **kw):
    paths = glob.glob(path_template)
    print("Found %d directories:" % len(paths), paths)

    jobs = [joblib.delayed(reconstruct_single, check_pickle=False)
            (path + "/" + suffix, cam_calib, proj_calib, **kw) for path in paths]

    results = joblib.Parallel(verbose=15, n_jobs=8, batch_size=1, pre_dispatch="all")(jobs)

    return {path: result for path, result in zip(paths, results)}


if __name__ == "__main__":
    cam_calib = load_camera_calibration("../calibration/camera/camera_calibration.json")
    proj_calib = load_projector_calibration("../calibration/projector/projector_calibration.json")[2]

    # Debug / Development
    # data_path = "D:/scanner_sim/captures/plane/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_330/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/rook_30_deg/position_330/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/shapes_30_deg/position_330/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/"

    # all, group = reconstruct_single(data_path + "default_scan/", cam_calib, proj_calib, max_group=25, save=True, plot=True, verbose=True)
    # print(all.shape, group.shape if group is not None else "")

    # Planes
    # data_path = "D:/scanner_sim/captures/plane/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/"
    # reconstruct_single(data_path + "gray/", cam_calib, proj_calib, max_group=25, plot=True, verbose=True)

    # data_path_template = "D:/scanner_sim/captures/stage_batch_2/no_ambient/%s_30_deg/position_*"
    # for object in ["pawn", "rook", "shapes"]:
    #     reconstruct_many(data_path_template % object, cam_calib, proj_calib, max_group=25, plot=True, verbose=True)

    # data_path_template = "D:/scanner_sim/captures/stage_batch_2/%s_30_deg/position_*"
    # for object in ["dodo", "avocado", "house", "chair", "vase", "bird", "radio"]:
    #     reconstruct_many(data_path_template % object, cam_calib, proj_calib, max_group=25, plot=True, verbose=True)

    plt.show()
