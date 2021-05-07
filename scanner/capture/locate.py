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


def compute_pca_variation(points, plot=False):
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


def fit_ring(points, stage_calib, title="Ring", plot=False):
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
        ax = plot_3d(points[::100, :], title, axis_equal=False)
        line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")
        basis(ax, p0, R.T, length=20)
        # basis(ax, c, R.T, length=20)
        axis_equal_3d(ax)

        # ax = plot_3d(local[::10000, :], "Local")
    return c, ax


def fit_sphere(points, stage_calib, plot=False):
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
        ax = plot_3d(points[::100, :], "Sphere", axis_equal=False)
        line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")
        basis(ax, p0, R.T, length=20)
        basis(ax, p0 + np.matmul(R.T, c), R.T, length=20)
        axis_equal_3d(ax)

    return c, ax


def locate_pawn(data_path, stage_calib, plot=False):
    print("\n\tLocating Pawn\n")

    ring = load_ply(data_path + "reconstructed/pawn_ring.ply")
    c_ring, _ = fit_ring(ring, stage_calib, title="Pawn Ring", plot=plot)

    sphere = load_ply(data_path + "reconstructed/pawn_sphere.ply")
    sphere = sphere[::100, :]
    c_sphere, _ = fit_sphere(sphere, stage_calib, plot=plot)

    R = build_local(stage_calib)

    p0, dir = stage_calib["p"], stage_calib["dir"]

    print("\nc_ring:", c_ring)
    print("c_sphere:", c_sphere)
    print("R:\n", R)

    c = np.zeros(3)
    c[:2] = (c_ring[:2] + c_sphere[:2]) / 2
    c[2] = c_sphere[2]
    print(c)
    c_loc = c
    c = p0 + np.matmul(R.T, c)
    print("Ball origin:", c)
    c -= dir * 5 * 25.4
    print("Pawn origin:", c)

    if plot:
        ax = plot_3d(ring[::100, :], "Global", axis_equal=False)
        line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")
        scatter(ax, sphere[::10, :], s=5)
        basis(ax, c, R.T, length=20)
        axis_equal_3d(ax)

    c_loc[:2] = 0
    stage_base = p0 + np.matmul(R.T, c_loc) - dir * 5 * 25.4
    print("Stage Base:", stage_base)

    return (c, R), stage_base


def locate_rook(data_path, stage_calib, stage_base, plot=False):
    print("\n\tLocating Rook\n")

    ring = load_ply(data_path + "reconstructed/rook_ring.ply")
    c_ring, ax = fit_ring(ring, stage_calib, title="Rook Ring", plot=plot)

    p0, dir = stage_calib["p"], stage_calib["dir"]
    R = build_local(stage_calib)

    c = stage_base + np.matmul(R.T, c_ring)
    print("Rook origin:\n", c)
    print("Rook basis:", R)

    if plot:
        basis(ax, c, R.T, length=20)

    return c, R


def locate_shapes(data_path, plot=False):
    print("\n\tLocating Shapes\n")

    def fit(filename, ax=None, **kwargs):
        p = np.asarray(o3d.io.read_point_cloud(filename).points)
        print("\n" + filename, p.shape)

        pca = PCA(n_components=3)
        p2 = pca.fit_transform(p)
        print(pca.mean_, pca.singular_values_, "\n", pca.components_)

        if ax:
            scatter(ax, p[::100, :], s=5, label="p", **kwargs)
            basis(ax, pca.mean_, pca.components_.T, length=20, **kwargs)

        return p, p2, pca.mean_, pca.singular_values_, pca.components_

    if plot:
        plt.figure("Shapes Target Origin", (12, 12))
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
    else:
        ax = None

    base = fit(data_path + "/reconstructed/shapes_base.ply", ax)
    hexagon = fit(data_path + "/reconstructed/shapes_hexagon.ply", ax)
    cylinder = fit(data_path + "/reconstructed/shapes_cylinder.ply", ax)

    ez = base[4][2, :]
    hexagon_c = hexagon[2]
    cylinder_c = cylinder[2]
    ey = cylinder_c - hexagon_c
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    R = np.stack([ex, ey, ez], axis=0)
    T = hexagon_c - 82*ex - 24.2*ey - 36*ez

    print("Shapes location:")
    print(T, "\n", R)

    if plot:
        basis(ax, T, R.T, length=50)

        ax.set_title("Shapes Target Origin")
        ax.set_xlabel("x, mm")
        ax.set_ylabel("z, mm")
        ax.set_zlabel("-y, mm")
        plt.tight_layout()
        axis_equal_3d(ax)

    return T, R


def locate_plane(data_path, plot=False):
    print("\n\tLocating Plane\n")

    def fit(filename, ax=None, **kwargs):
        p = np.asarray(o3d.io.read_point_cloud(filename).points)
        print("\n" + filename, p.shape)

        pca = PCA(n_components=3)
        p2 = pca.fit_transform(p)
        print(pca.mean_, pca.singular_values_, "\n", pca.components_)

        if ax:
            scatter(ax, p[::200, :], s=5, label="p", **kwargs)
            basis(ax, pca.mean_, pca.components_.T, length=20, **kwargs)

        return p, p2, pca.mean_, pca.singular_values_, pca.components_

    if plot:
        plt.figure("Plane Target Origin", (12, 12))
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
    else:
        ax = None

    plane = fit(data_path + "/reconstructed/plane.ply", ax)
    T, R = plane[2], plane[4]

    print("Plane location:")
    print("T:", T, "\nR:\n", R)

    if plot:
        basis(ax, T, R.T, length=50)

        ax.set_title("Plane Target Origin")
        ax.set_xlabel("x, mm")
        ax.set_ylabel("z, mm")
        ax.set_zlabel("-y, mm")
        plt.tight_layout()
        axis_equal_3d(ax)

        z = plane[1][:, 2]
        std = np.std(z)
        plt.figure("Plane Variance", (12, 8))
        plt.hist(z, bins=200)
        plt.title("Plane Variance (std=%.3f mm)" % std)
        plt.xlabel("mm")
        plt.ylabel("Counts")
        plt.tight_layout()

        plt.savefig(data_path + "/reconstructed/plane_variance.png", dpi=150)

    with open(data_path + "/reconstructed/plane_location.json", "w") as f:
        json.dump({"T": T,
                   "R": R,
                   "T_Help": "[x, y, z] in mm in camera's frame of reference",
                   "R_Help": "[ex, ey, ez]",
                   "Camera": "https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html"},
                  f, indent=4, cls=NumpyEncoder)

    return T, R

if __name__ == "__main__":
    stage_calib = numpinize(json.load(open("../calibration/stage/stage_calibration.json", "r")))

    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/"
    # _, stage_base = locate_pawn(data_path, stage_calib, plot=True)
    #
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/rook_30_deg/"
    # locate_rook(data_path, stage_calib, stage_base, plot=True)
    #
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/shapes_30_deg/position_0/gray/"
    # locate_shapes(data_path, plot=True)

    # data_path = "D:/scanner_sim/captures/stage_batch_2/shapes_30_deg/position_0/gray/"
    # locate_shapes(data_path, plot=True)

    data_path = "D:/scanner_sim/calibration/accuracy_test/clear_plane/gray/"
    # data_path = "D:/scanner_sim/calibration/accuracy_test/charuco_plane/gray/"
    locate_plane(data_path, plot=True)

    plt.show()
