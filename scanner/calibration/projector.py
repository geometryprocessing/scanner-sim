import json
import cv2
import numpy as np
from camera import load_camera_calibration
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import least_squares
from utils import *
from calibrate import *


def scatter(ax, p, *args, **kwargs):
    if len(p.shape) > 1:
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1], *args, **kwargs)
    else:
        ax.scatter(p[0], p[2], -p[1], **kwargs)


def line(ax, p1, p2, *args, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], *args, **kwargs)


def basis(ax, T, R, *args, length=1, **kwargs):
    line(ax, T, T + length * R[:, 0], "r")
    line(ax, T, T + length * R[:, 1], "g")
    line(ax, T, T + length * R[:, 2], "b")


def board(ax, T, R, *args, label="", **kwargs):
    line(ax, T, T + 375 * R[:, 0], "orange", linestyle="--", label=label)
    line(ax, T, T + 270 * R[:, 1], "orange", linestyle="--")
    line(ax, T + 375 * R[:, 0], T + 375 * R[:, 0] + 270 * R[:, 1], "orange", linestyle="--")
    line(ax, T + 270 * R[:, 1], T + 375 * R[:, 0] + 270 * R[:, 1], "orange", linestyle="--")

    basis(ax, T, R, length=10)


def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def calibrate_geometry(data_path, camera_calib, max_planes=70, intrinsic=None, no_tangent=False, save=False, plot=False, save_figures=True, **kw):
    charuco, checker, plane_errors = reconstruct_planes(data_path, camera_calib, **kw)
    checker_3d, checker_2d, checker_local = checker
    avg_plane_errors, all_plane_errors = plane_errors
    w, h = 1920, 1080
    print("\nReconstructed:", len(checker_3d))
    print("Mean plane error", np.mean(avg_plane_errors))

    stride = len(checker_3d) // max_planes + 1
    checker_3d, checker_2d, checker_local = checker_3d[::stride], checker_2d[::stride], checker_local[::stride]

    initial_calib, initial_errors = calibrate(checker_local, checker_2d, (h, w), no_tangent=no_tangent,
                                              out_dir=data_path, plot=plot, save_figures=save_figures, **kw)
    mtx_guess, dist_guess, _, _ = initial_calib
    selected = initial_errors[2]

    all_obj = np.vstack([checker_3d[i] for i in selected]).astype(np.float32)
    all_img = np.vstack([checker_2d[i] for i in selected]).astype(np.float32)

    if intrinsic is None:
        flags = (cv2.CALIB_FIX_TANGENT_DIST if no_tangent else 0) | cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K2
        # flags |= cv2.CALIB_FIX_K1

        # mtx_guess[1, 2] = min(h - 1, mtx_guess[1, 2])  # initial guess of center point cannot be outside of the image
        mtx_guess = np.array([[3000, 0, 1000], [0, 3000, 1000], [0, 0, 1]]).astype(np.float32)

        full_calib = cv2.calibrateCamera([all_obj], [all_img], (w, h), mtx_guess, None, flags=flags)
        # full_errors = projection_errors([all_obj], [all_img], full_calib)
        # full_errors = full_errors[0][0], full_errors[1][0]

        ret, mtx, dist, rvecs, tvecs = full_calib
        print("\nCalibration matrix:\n", mtx)
        print("\nDistortions:", dist)
        # print("\nMean full error:", full_errors[0])

        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        print("\nOptimal calibration matrix:\n", new_mtx)
        print("\nRegion of interest:", roi)

        intrinsic = mtx, dist.ravel(), new_mtx, np.array(roi)
        # T, (R, _) = tvecs[0].ravel(), cv2.Rodrigues(rvecs[0])
        # origin = np.matmul(R.T, -T)
        # print("\nProjector Origin:", origin)
        # print("\nProjector Basis [ex, ey, ez]:\n", R)
    else:
        mtx, dist, new_mtx, roi = intrinsic

    ret, rvec, tvec = cv2.solvePnP(all_obj, all_img, mtx, dist)
    full_errors = projection_errors([all_obj], [all_img], (ret, mtx, dist, [rvec], [tvec]))
    full_errors = full_errors[0][0], full_errors[1][0]
    print("\nMean full error:", full_errors[0])

    T, (R, _) = tvec.ravel(), cv2.Rodrigues(rvec)
    origin = np.matmul(R.T, -T)
    print("\nProjector Origin:", origin)
    print("\nProjector Basis [ex, ey, ez]:\n", R)
    extrinsic = origin, R

    if plot:
        plt.figure("Projector Calibration", (12, 12))
        plt.clf()
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Projector Calibration")
        skip = 3

        charuco_3d, charuco_id, charuco_frame = charuco
        to_plot = charuco_frame[::stride]
        for i in range(len(to_plot)):
            if i % skip == 0:
                Ti, Ri = to_plot[i]
                board(ax, Ti, Ri, label="Charuco Boards" if i == 0 else "")

        scatter(ax, np.concatenate(charuco_3d[::stride][::skip], axis=0), c="g", s=5, label="Charuco Corners")
        scatter(ax, np.concatenate(checker_3d[::skip], axis=0), c="b", s=8, label="Checker Corners")
        scatter(ax, origin, c="k", s=15, label="Projector Origin")
        basis(ax, origin, R.T, length=20)

        ax.set_xlabel("x, mm")
        ax.set_ylabel("z, mm")
        ax.set_zlabel("-y, mm")
        plt.legend()
        plt.tight_layout()
        axis_equal_3d(ax)

        if save_figures:
            ax.view_init(elev=10, azim=-20)
            plt.savefig(data_path + "/calibration_view1.png", dpi=320)
            ax.view_init(elev=12, azim=26)
            plt.savefig(data_path + "/calibration_view2.png", dpi=320)

        plt.figure("Distortions", (16, 9))
        plt.clf()
        points = np.mgrid[0:17, 0:8].T.reshape(-1, 2) * 100 + np.array([160, 190])
        u_points = cv2.undistortPoints(points.astype(np.float32), mtx, dist, None, new_mtx).reshape(-1, 2)
        plt.plot(points[:, 0], points[:, 1], ".r")
        plt.plot(u_points[:, 0], u_points[:, 1], ".b")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/distortions.png", dpi=160)


        plt.figure("Errors", (12, 7))
        plt.clf()
        plt.subplot(2, 1, 1, title="Camera projection")
        plt.hist(np.concatenate(all_plane_errors), bins=50)
        plt.xlabel("Error, pixels")
        plt.tight_layout()

        plt.subplot(2, 1, 2, title="Projector projection")
        plt.hist(full_errors[1], bins=50)
        plt.xlabel("Error, pixels")
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/calibration_errors.png", dpi=160)

    if save:
        save_projector_calibration(intrinsic, extrinsic, data_path + "/calibration.json", mean_error=full_errors[0])

    return intrinsic, extrinsic, full_errors


def save_projector_calibration(intrinsic, extrinsic, filename, mean_error=0.0):
    with open(filename, "w") as f:
        json.dump({"mtx": intrinsic[0],
                   "dist": intrinsic[1],
                   "new_mtx": intrinsic[2],
                   "roi": intrinsic[3],
                   "origin": extrinsic[0],
                   "basis": extrinsic[1],
                   "mean_projection_error, pixels": mean_error,
                   "projector": "Texas Instrument DPL4710LC",
                   "image_width, pixels": 1920,
                   "image_height, pixels": 1080,
                   "focus_distance, cm": 50,
                   "aperture, mm": 7.5}, f, indent=4, cls=NumpyEncoder)


def load_projector_calibration(filename):
    with open(filename, "r") as f:
        calib = numpinize(json.load(f))

    intrinsic = calib["mtx"], calib["dist"], calib["new_mtx"], calib["roi"]
    extrinsic = calib["origin"], calib["basis"]

    return intrinsic, extrinsic, calib


if __name__ == "__main__":
    camera_calib = load_camera_calibration("D:/Scanner/Calibration/camera_intrinsics/data/charuco/calibration.json")

    data_path = "D:/Scanner/Calibration/projector_intrinsics/data/charuco_checker_5mm/"
    intrinsic, _, _ = calibrate_geometry(data_path, camera_calib, max_planes=500, no_tangent=True, save=True, plot=True, save_figures=True)

    data_path = "D:/Scanner/Calibration/projector_extrinsic/data/charuco_checker_5mm/"
    _, extrinsic, errors = calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, max_planes=500, no_tangent=True, save=True, plot=True, save_figures=True)

    save_projector_calibration(intrinsic, extrinsic, "projector/calibration.json", mean_error=errors[0])
    intrinsic, extrinsic, all = load_projector_calibration("projector/calibration.json")

    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_5_deg_before/merged/"
    # calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, center=True, no_tangent=True, save=True, plot=True, save_figures=True)

    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_2_deg_after/merged/"
    # calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, max_planes=50, center=True, no_tangent=True, save=True, plot=True, save_figures=True)

    plt.show()
