import json
import cv2
import numpy as np
from camera import *
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import least_squares
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


# def projector(ax, pos, rvec, mtx, p):
#     T, R = pos.ravel(), cv2.Rodrigues(rvec)
#     p = p.reshape((-1, 3))
#
#     basis(ax, T, R)
#     scatter(ax, p, c="m")


def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def calibrate_geometry(data_path, camera_calib, center=False, plot=False, save_figures=True, **kw):
    full, half = [[160, 190]] * 65, [[800 + 160, 190]] * 35
    half.extend([[160, 190]] * 30)
    if center:
        half = [400 + 160, 190] * 23

    charuco, checker, cam_errors = reconstruct_planes(data_path, camera_calib, full_offsets=full, half_offsets=half)
    charuco_3d, charuco_id, charuco_frame = charuco
    checker_3d, checker_2d, checker_local = checker

    mtx_guess = np.array([[3000, 0, 1000], [0, 3000, 1000], [0, 0, 1]]).astype(np.float32)
    initial_calib, initial_errors = calibrate(checker_local[::2], checker_2d[::2], (1080, 1920), mtx_guess=mtx_guess,
                                         out_dir=data_path, plot=plot, save_figures=save_figures, **kw)
    mtx, dist, new_mtx, roi = initial_calib

    flags = cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS

    all_obj, all_img = np.vstack(checker_3d).astype(np.float32), np.vstack(checker_2d).astype(np.float32)
    full_calib = cv2.calibrateCamera([all_obj], [all_img], (1920, 1080), mtx_guess, None, flags=flags)
    ret, mtx, dist, rvecs, tvecs = full_calib

    full_error = projection_errors([all_obj], [all_img], full_calib)[0]
    T, (R, _) = tvecs[0].ravel(), cv2.Rodrigues(rvecs[0])
    origin = np.matmul(R.T, -T)

    w, h = 1920, 1080
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print("Calibration matrix:\n", mtx)
    print("Distortions:\n", dist)
    print("Optimal calibration matrix:\n", new_mtx)
    print("Region of interest:\n", roi)
    print("Mean projector error:", full_error)
    print("Projector Origin:", origin)
    print("Projector Basis:", R)

    if plot:
        plt.figure("Projector Calibration", (12, 12))
        plt.clf()
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Projector Calibration")
        skip = 5

        for i in range(len(charuco[0])):
            if i % skip == 0:
                Ti, Ri = charuco_frame[i]
                board(ax, Ti, Ri, label="Charuco Boards" if i == 0 else "")

        # scatter(ax, np.concatenate(charuco_3d, axis=0)[::skip, :], c="g", s=5, label="Charuco Corners")
        scatter(ax, np.concatenate(checker_3d, axis=0)[::skip, :], c="b", s=8, label="Checker Corners")
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

        plt.figure("Errors", (12, 7))
        # plt.clf()
        plt.subplot(2, 2, 1, title="Camera projection")
        plt.hist(cam_errors, bins=50)
        plt.xlabel("Error, pixels")
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/calibration_errors.png", dpi=160)


def old_load_corners(filename):
    corners = json.load(open(filename, "r"))

    names = [name for name, points in corners.items()]
    img_points = [np.array(points["img_points"]).reshape(-1, 2).astype(np.float32) for name, points in corners.items()]
    obj_points = [np.array(points["obj_points"]).reshape(-1, 3).astype(np.float32) for name, points in corners.items()]
    ids = [np.array(points["ids"]).ravel().astype(np.int) for name, points in corners.items()]

    return names, img_points, obj_points, ids


def calibrate_intrinsic(data_path, camera_calib, no_tangent=True, plot=False, save_figures=True, **kw):
    cam_mtx, cam_dist, cam_new_mtx = camera_calib["mtx"], camera_calib["dist"], camera_calib["new_mtx"]

    c_names, c_img_points, c_obj_points, c_ids = old_load_corners(data_path + "/charuco/corners.json")
    f_names, f_img_points, f_obj_points, f_ids = old_load_corners(data_path + "/checker/detected_full/corners.json")
    h_names, h_img_points, h_obj_points, h_ids = old_load_corners(data_path + "/checker/detected_half/corners.json")

    c_dict = dict(zip(c_names, zip(c_img_points, c_obj_points, c_ids)))
    f_dict = dict(zip(f_names, zip(f_img_points, f_obj_points, f_ids)))
    h_dict = dict(zip(h_names, zip(h_img_points, h_obj_points, h_ids)))

    checker_board = np.stack(np.meshgrid(np.arange(8), np.arange(17), indexing="ij"), axis=2) * 100
    checker_board[:, :, 0] += 190
    checker_board[:, :, 1] += 160
    checker_board = checker_board[:, :, ::-1]

    charuco_3d, checker_3d = [], []
    objs, prjs = [], []

    if plot:
        plt.figure("Projector Calibration", (12, 12))
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Projector Calibration")

    c_proj_errors, skip = [], 5

    for i, name in enumerate(c_names):
        id = int(name[name.rfind("_") + 1:-4])
        pair = "checker_%d.png" % id

        if not pair in f_names:  # and not pair in h_names:
            continue
        else:
            print("Including", pair)

        c_img = cv2.undistortPoints(c_img_points[i], cam_mtx, cam_dist, None, cam_new_mtx).reshape(-1, 2)
        c_obj = c_obj_points[i]

        p_img = cv2.undistortPoints(f_dict[pair][0], cam_mtx, cam_dist, None, cam_new_mtx).reshape(-1, 2)
        p_obj = f_dict[pair][1]
        p_obj = p_obj.reshape(-1, 3)[:, :2]
        p_obj += [160, 190]

        ret, rvec, tvec = cv2.solvePnP(c_obj, c_img, cam_new_mtx, None)
        T, (R, _) = tvec.ravel(), cv2.Rodrigues(rvec)

        c_proj, _ = cv2.projectPoints(c_obj, rvec, tvec, cam_new_mtx, None)
        c_proj_errors.extend(np.linalg.norm(c_img - c_proj.reshape(-1, 2), axis=1).tolist())

        c_3d = np.matmul(R, c_obj.T) + tvec
        charuco_3d.append(c_3d.T)

        p_3d = lift_to_3d(p_img, cam_new_mtx, T, R, offset=0)
        checker_3d.append(p_3d)

        obj = np.zeros((p_3d.shape[0], 3))
        obj[:, 0] = np.dot(p_3d - T, R[:, 0])
        obj[:, 1] = np.dot(p_3d - T, R[:, 1])
        obj[:, 2] = 0
        objs.append(obj.astype(np.float32))
        prjs.append(p_obj.astype(np.float32))

        if plot and i % skip == 0:
            board(ax, T, R, label="Charuco Boards" if i == 0 else "")

    print("Mean charuco re-projection error:", np.average(c_proj_errors))

    # flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST
    flags = cv2.CALIB_FIX_TANGENT_DIST if no_tangent else 0

    w, h = 1920, 1080
    calibration_estimate = cv2.calibrateCamera(objs, prjs, (w, h), None, None, flags=flags)
    errors = projection_errors(objs, prjs, calibration_estimate)
    ret, mtx_guess, dist, rvecs, tvecs = calibration_estimate
    print("Reprojection errors:", errors, "\nMean error:", np.mean(errors))
    print("Calibration matrix guess:\n", mtx_guess)

    mtx_guess = np.array([[3000, 0, 1000], [0, 3000, 1000], [0, 0, 1]]).astype(np.float32)

    checker_all = np.vstack(checker_3d).astype(np.float32)
    prj_all = np.vstack(prjs).astype(np.float32)
    full_calibration = cv2.calibrateCamera([checker_all], [prj_all], (1920, 1080), mtx_guess, None, flags=(flags | cv2.CALIB_USE_INTRINSIC_GUESS))
    error = projection_errors([checker_all], [prj_all], full_calibration)[0]
    ret, mtx, dist, rvecs, tvecs = full_calibration
    T, (R, _) = tvecs[0].ravel(), cv2.Rodrigues(rvecs[0])
    origin = np.matmul(R.T, -T)

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print("Calibration matrix:\n", mtx)
    print("Distortions:\n", dist)
    print("Optimal calibration matrix:\n", new_mtx)
    print("Region of interest:\n", roi)
    print("Mean projector error:", error)
    print("Projector Origin:", origin)
    print("Projector Basis:", R)
    # print("Distance from lines origin:", np.linalg.norm(origin - lines_origin))

    if plot:
        scatter(ax, np.concatenate(charuco_3d[::skip], axis=0), c="g", s=5, label="Charuco Corners")
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

        plt.figure("Errors", (12, 7))
        plt.subplot(2, 2, 1, title="Charuco re-projection")
        plt.hist(c_proj_errors, bins=50)
        plt.xlabel("Error, pixels")

        if save_figures:
            plt.savefig(data_path + "/calibration_errors.png", dpi=160)


if __name__ == "__main__":
    data_path = "D:/Scanner/Calibration/projector_intrinsics/data/charuco_checker_5mm/"
    # data_path = "D:/Scanner/Calibration/projector_extrinsic/data/charuco_checker_5mm/"
    # data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_5_deg_before/"

    camera_calib = load_camera_calibration("D:/Scanner/Calibration/camera_intrinsics/data/charuco/calibration.json")
    calibrate_geometry(data_path, camera_calib, error_thr=1.0, no_tangent=True, plot=True, save_figures=True)
    # calibrate_intrinsic(data_path, camera_calib, no_tangent=True, plot=True, save_figures=True)

    plt.show()
