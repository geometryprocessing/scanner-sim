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
from skimage import measure
from skimage import filters


def apply_connected_components_correction(cam, proj, res=(4852, 6464), plot=True):
    img = np.zeros(res, dtype=np.int)
    pat_r = np.zeros(res, dtype=np.int)
    pat_c = np.zeros(res, dtype=np.int)
    r, c = cam[:, 1], cam[:, 0]

    img[r, c] = proj[:, 0] + proj[:, 1].astype(np.int) * 1920
    pat_r[r, c] = proj[:, 1]
    pat_c[r, c] = proj[:, 0]

    labels = measure.label(img, connectivity=2)

    cam2, proj2 = [], []
    for i, region in enumerate(measure.regionprops(labels)):
        if i % 10000 == 0:
            print(i)
        if region.label > 0:
            cam2.append(region.centroid[::-1])
            idx = region.coords
            # if i % 10000 == 0:
            #     print(idx)
            r, c = idx[0, 0], idx[0, 1]
            proj2.append([pat_c[r, c], pat_r[r, c]])

        # if i > 100000:
        #     break
    cam2, proj2 = np.array(cam2), np.array(proj2)
    print(cam2, cam2.shape)
    print(proj2, proj2.shape)

    if plot:
        plot_image(img, "Decoded")
        plot_image(labels, "Labels", vmax=10)

        # img = np.zeros(res, dtype=np.int)
        # r, c = cam2[:, 1].astype(np.int), cam2[:, 0].astype(np.int)
        # img[r, c] = proj2[:, 0] + proj2[:, 1].astype(np.int) * 1920
        # plot_image(img, "Grouped")

        plt.show()

    return cam2, proj2


def img_to_ray(p_img, mtx):
    p_img = (p_img - mtx[:2, 2]) / mtx[[0, 1], [0, 1]]
    return  np.concatenate([p_img, np.ones((p_img.shape[0], 1))], axis=1)


def reconstruct_single(data_path, cam_calib, proj_calib, save=False, plot=False, save_figures=True, **kw):
    p_cam, p_proj = load_decoded(data_path)
    p_cam, p_proj = apply_connected_components_correction(p_cam, p_proj)
    # exit()

    # cam2 = cv2.undistortPoints(p_cam.astype(np.float), cam_calib["mtx"], cam_calib["dist"], None, cam_calib["new_mtx"]).reshape((-1, 2))
    # proj2 = cv2.undistortPoints(p_proj.astype(np.float), proj_calib["mtx"], proj_calib["dist"], None, proj_calib["new_mtx"]).reshape((-1, 2))
    # cam_3d = img_to_ray(cam2, cam_calib["new_mtx"])
    # proj_3d = img_to_ray(proj2, proj_calib["new_mtx"])

    cam2 = cv2.undistortPoints(p_cam.astype(np.float), cam_calib["mtx"], cam_calib["dist"]).reshape((-1, 2))
    proj2 = cv2.undistortPoints(p_proj.astype(np.float), proj_calib["mtx"], proj_calib["dist"]).reshape((-1, 2))
    cam_3d = np.concatenate([cam2, np.ones((cam2.shape[0], 1))], axis=1)
    proj_3d = np.concatenate([proj2, np.ones((proj2.shape[0], 1))], axis=1)

    proj_3d = np.matmul(proj_calib["basis"].T, proj_3d.T).T

    v12 = np.sum(np.multiply(cam_3d, proj_3d), axis=1)
    v1 = np.linalg.norm(cam_3d, axis=1)**2
    v2 = np.linalg.norm(proj_3d, axis=1)**2
    L = (np.matmul(cam_3d, proj_calib["origin"]) * v2 + np.matmul(proj_3d, -proj_calib["origin"]) * v12) / (v1 * v2 - v12**2)

    p_3d = cam_3d * L[:, None]

    if save:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p_3d.astype(np.float32))
        o3d.io.write_point_cloud(data_path + "/points.ply", pcd)

    if plot:
        plt.figure("Reconstruction", (12, 12))
        plt.clf()
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Reconstruction")

        scatter(ax, p_3d[::500, :], s=5, label="p")

        ax.set_title("Reconstruction")
        ax.set_xlabel("x, mm")
        ax.set_ylabel("z, mm")
        ax.set_zlabel("-y, mm")
        plt.tight_layout()
        axis_equal_3d(ax)

        if save and save_figures:
            plt.savefig(data_path + "/points.png", dpi=160)

    return p_3d


if __name__ == "__main__":
    # data_path = "D:/scanner_sim/captures/plane/default_scan/decoded"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_330/default_scan/decoded/"
    data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/default_scan/decoded/"

    cam_calib = load_camera_calibration("../calibration/camera/camera_calibration.json")
    proj_calib = load_projector_calibration("../calibration/projector/projector_calibration.json")[2]

    p_3d = reconstruct_single(data_path, cam_calib, proj_calib, save=True, plot=True)
    print(p_3d.shape)

    plt.show()
