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


def load_points(filename):
    return np.asarray(o3d.io.read_point_cloud(filename).points)


def build_local(stage_calib):
    p0, dir = stage_calib["p"], stage_calib["dir"]
    ex = np.array([1, 0, 0])
    proj = dir * np.dot(dir, ex)
    ex = ex - proj
    ex = ex / np.linalg.norm(ex)
    # print(ex, np.dot(ex, dir))
    ey = np.cross(dir, ex)
    R = np.array([ex, ey, dir])
    # print(R)
    # print(np.matmul(R, R.T))
    return R


def compute_pca_variation(points, plot=True):
    pca = PCA(n_components=3)
    p2 = pca.fit_transform(points)
    fit_plane(points)

    if plot:
        plt.figure("PCA", (16, 9))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            mean, std = np.mean(p2[:, i]), np.std(p2[:, i])
            idx = np.nonzero(np.abs(p2[:, i] - mean) < 3 * std)[0]
            plt.hist(p2[idx, i], bins=1000)
            mean, std = np.mean(p2[idx, i]), np.std(p2[idx, i])
            print(i, mean, std)
            plt.title("Component %d (mean = %.5f, std = %.5f)" % (i, mean, std))
            plt.xlabel("Variance, mm")
            plt.tight_layout()
            # if i == 2:
            #     plt.savefig(path + "plane_reconstruction_errors.png", dpi=160)


def fit_ring(points, stage_calib, plot=True):
    p0, dir = stage_calib["p"], stage_calib["dir"]

    R = build_local(stage_calib)
    # points -= p0

    # proj = dir[None, :] * np.matmul(points, dir)[:, None]

    local = np.matmul(R, (points-p0).T).T

    def circle_loss(p, xy):
        cx, cy, R = p
        x, y = xy[:, 0] - cx, xy[:, 1] - cy
        r = np.sqrt(x ** 2 + y ** 2)
        return r - R

    cx, cy, radius = least_squares(circle_loss, [0, 0, 1], args=(local,))['x']
    print(cx, cy, radius)
    c = np.array([cx, cy, 0])
    # c = p0 + np.matmul(R.T, c)

    if plot:
        ax = plot_3d(points[::10000, :], "Ring")
        line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")
        basis(ax, p0, R.T, length=20)
        # basis(ax, c, R.T, length=20)
        axis_equal_3d(ax)

        # ax = plot_3d(local[::10000, :], "Local")
    return c


def fit_sphere(points, stage_calib, plot=True):
    p0, dir = stage_calib["p"], stage_calib["dir"]

    R = build_local(stage_calib)

    local = np.matmul(R, (points - p0).T).T

    def sphere_loss(p, xyz):
        cx, cy, cz, R = p
        x, y, z = xyz[:, 0] - cx, xyz[:, 1] - cy, xyz[:, 2] - cz
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return r - R

    cx, cy, cz, radius = least_squares(sphere_loss, [0, 0, 0, 1], args=(local,))['x']
    print(cx, cy, cz, radius)
    c = np.array([cx, cy, cz])
    # print(c)

    if plot:
        ax = plot_3d(points[::100, :], "Sphere")
        line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")
        basis(ax, p0, R.T, length=20)
        basis(ax, p0 + np.matmul(R.T, c), R.T, length=20)
        axis_equal_3d(ax)

    return c


if __name__ == "__main__":
    data_path = "D:/scanner_sim/captures/plane/gray/decoded/reconstructed/"
    # data_path = "D:/scanner_sim/captures/plane/default_scan/decoded/reconstructed/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/decoded"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/gray/decoded/reconstructed/"

    stage_calib = numpinize(json.load(open("../calibration/stage/stage_calibration.json", "r")))

    # all, groups = load_decoded(data_path)
    p = load_points(data_path + "/group_points.ply")
    # print(p.shape, cam.shape)
    compute_pca_variation(p, plot=True)
    plt.show()
    exit()

    data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/"

    ring = load_ply(data_path + "pawn_ring.ply")
    c_ring = fit_ring(ring, stage_calib, plot=True)

    sphere = load_ply(data_path + "pawn_sphere.ply")
    sphere = sphere[::100, :]
    c_sphere = fit_sphere(sphere, stage_calib, plot=True)

    R = build_local(stage_calib)

    p0, dir = stage_calib["p"], stage_calib["dir"]

    print("\nc_ring:", c_ring)
    print("c_sphere:", c_sphere)
    print("R:\n", R)

    c = np.zeros(3)
    c[:2] = (c_ring[:2] + c_sphere[:2]) / 2
    c[2] = c_sphere[2]
    print(c)
    c = p0 + np.matmul(R.T, c)
    print("Ball origin:", c)
    c -= dir * 5 * 25.4
    print("Pawn origin:", c)

    ax = plot_3d(ring[::1000, :], "Global")
    line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")
    scatter(ax, sphere[::100, :])
    basis(ax, c, R.T, length=20)
    axis_equal_3d(ax)

    plt.show()
