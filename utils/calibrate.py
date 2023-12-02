import json
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Duplicate from detect.py in scanner
def load_corners(filename):
    corners = {}

    for name, points in json.load(open(filename, "r")).items():
        img_points = np.array(points["img_points"]).reshape(-1, 2).astype(np.float32)
        obj_points = np.array(points["obj_points"]).reshape(-1, 3).astype(np.float32)
        ids = np.array(points["ids"]).ravel().astype(np.int32)
        corners[name] = {"img": img_points, "obj": obj_points, "idx": ids}

    return corners


def fit_line(points):
    center = np.mean(points, axis=0)
    uu, dd, vv = np.linalg.svd(points - center)
    return center, vv[0]


def point_line_dist(p, l0, l1):
    return np.linalg.norm(np.cross(l1 - l0, p - l0)) / np.linalg.norm(l1 - l0)


def projection_errors(obj_points, img_points, calibration):
    ret, mtx, dist, rvecs, tvecs = calibration

    avg_errors, all_errors = [], []
    for i, (obj_p, img_p) in enumerate(zip(obj_points, img_points)):
        img_points_2, _ = cv2.projectPoints(obj_p, rvecs[i], tvecs[i], mtx, dist)
        img_points_2 = img_points_2.reshape((obj_p.shape[0], 2))
        all_errors.append(np.linalg.norm(img_p - img_points_2, axis=1))
        avg_errors.append(np.average(all_errors[-1]))

    return np.array(avg_errors), all_errors


def calibrate(obj_points, img_points, dim, error_thr=1.0, mtx_guess=None, no_tangent=True,
                                centerPrincipalPoint=None, out_dir="", plot=False, save_figures=True, **kw):
    h, w, n = dim[0], dim[1], len(img_points)
    print("Initial:", n)

    flags = cv2.CALIB_FIX_TANGENT_DIST if no_tangent else 0
    if mtx_guess is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    initial_calibration = cv2.calibrateCamera(obj_points, img_points, (w, h), mtx_guess, None, flags=flags)

    initial_errors = projection_errors(obj_points, img_points, initial_calibration)
    print("Mean initial error:", np.mean(initial_errors[0]))

    selected = np.nonzero(initial_errors[0] < error_thr)[0]
    print("Selected:", len(selected))

    obj_selected, img_selected = [obj_points[i] for i in selected], [img_points[i] for i in selected]
    refined_calibration = cv2.calibrateCamera(obj_selected, img_selected, (w, h), mtx_guess, None, flags=flags)

    refined_errors = projection_errors(obj_selected, img_selected, refined_calibration)
    print("Mean selected error:", np.mean(refined_errors[0]))

    ret, mtx, dist, rvecs, tvecs = refined_calibration
    print("\nmtx:\n", mtx)
    print("\ndist:", dist)

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h), centerPrincipalPoint=centerPrincipalPoint)
    print("\nnew_mtx:\n", new_mtx)
    print("\nroi:", roi)

    if plot:
        plt.figure("Calibration", (4.4, 3))
        plt.clf()
        plt.plot(np.arange(n), initial_errors[0], ".r", markersize=3.5, label="Initial Errors")
        plt.plot(selected, refined_errors[0], ".b", markersize=3.5, label="Selected Stops")
        plt.plot([-1, n], [error_thr, error_thr], '--k', linewidth=1.25, label="Threshold")
        # plt.title("Projection Errors")
        plt.xlabel("Stop #")
        plt.ylabel("Error, pixels")
        plt.xlim([-2, n+1])
        plt.ylim([0, 1.1 * np.max(initial_errors[0])])
        plt.legend()
        plt.tight_layout()
        if save_figures:
            plt.savefig(out_dir + "/initial_projection_errors.png", dpi=300)

    calibration = mtx, dist.ravel(), new_mtx, np.array(roi)
    errors = initial_errors, refined_errors, selected

    return calibration, errors


def trace_ray(T, R, p, d):
    A = np.stack((R[:, 0], R[:, 1], -d), axis=1)
    b = p - T
    uvt = np.matmul(np.linalg.inv(A), b)
    return p + uvt[2]*d


def lift_to_3d(p_img, mtx, T, R, offset=0.0):
    p_world = np.zeros((p_img.shape[0], 3))
    for i in range(p_img.shape[0]):
        p_world[i, :] = trace_ray(T + offset * R[:, 2], R, np.zeros((3)), np.array([(p_img[i, 0] - mtx[0, 2]) / mtx[0, 0],
                                                                                    (p_img[i, 1] - mtx[1, 2]) / mtx[1, 1], 1]))
    return p_world


def reconstruct_planes(data_path, camera_calib, min_points=80, thr=35, center=False, undistorted=False, charuco_only=False, extra_output=False, **kw):
    cam_mtx, cam_dist, cam_new_mtx = camera_calib["mtx"], camera_calib["dist"], camera_calib["new_mtx"]
    charuco = load_corners(data_path + "/charuco/corners.json")

    if not charuco_only:
        full = load_corners(data_path + "/checker/detected_full/corners.json")
        half = load_corners(data_path + "/checker/detected_half/corners.json")

        n = len(charuco.keys())
        full_offsets, half_offsets = [[160, 190]] * n, [[800 + 160, 190]] * thr
        half_offsets.extend([[160, 190]] * (n - thr))
        if center:
            half_offsets = [[400 + 160, 190]] * n

    charuco_template, checker_template = "blank_%d.png", "checker_%d.png"
    charuco_img, charuco_3d, charuco_id, charuco_frame = [], [], [], []
    checker_img, checker_3d, checker_2d, checker_local = [], [], [], []

    avg_errors, all_errors = [], []
    for i, id in enumerate(sorted([int(name[name.rfind("_") + 1:-4]) for name in charuco.keys()])):
        name, pair = charuco_template % id, checker_template % id

        if not charuco_only:
            if pair not in full and pair not in half:
                continue

        c_obj = charuco[name]["obj"]
        c_idx = charuco[name]["idx"]
        c_img = charuco[name]["img"].astype(np.float32).reshape((-1, 2))
        if not undistorted:
            c_img = cv2.undistortPoints(c_img.reshape((-1, 1, 2)), cam_mtx, cam_dist, P=cam_new_mtx).reshape((-1, 2))

        if len(c_idx) < min_points:
            continue

        ret, rvec, tvec = cv2.solvePnP(c_obj, c_img, cam_new_mtx, None)
        T, (R, _) = tvec.ravel(), cv2.Rodrigues(rvec)

        projected = cv2.projectPoints(c_obj, rvec, tvec, cam_new_mtx, None)[0].reshape(-1, 2)
        all_errors.append(np.linalg.norm(c_img - projected, axis=1))
        avg_errors.append(np.average(all_errors[-1]))

        c_3d = np.matmul(R, c_obj.T) + tvec
        charuco_img.append(c_img)
        charuco_3d.append(c_3d.T)
        charuco_id.append(c_idx)
        charuco_frame.append((T, R))

        if not charuco_only:
            src = full if pair in full else half
            p_img = src[pair]["img"].astype(np.float32).reshape((-1, 2))
            if not undistorted:
                p_img = cv2.undistortPoints(p_img.reshape((-1, 1, 2)), cam_mtx, cam_dist, P=cam_new_mtx).reshape((-1, 2))
            p_prj = src[pair]["obj"][:, :2] + (full_offsets[i] if pair in full else half_offsets[i])

            p_3d = lift_to_3d(p_img, cam_new_mtx, T, R, offset=0)
            checker_img.append(p_img)
            checker_3d.append(p_3d)
            checker_2d.append(p_prj.astype(np.float32))

            local = np.zeros((p_3d.shape[0], 3))
            local[:, 0] = np.dot(p_3d - T, R[:, 0])
            local[:, 1] = np.dot(p_3d - T, R[:, 1])
            checker_local.append(local.astype(np.float32))

    chrk = (charuco_img, charuco_3d, charuco_id, charuco_frame) if extra_output else (charuco_3d, charuco_id, charuco_frame)
    chck = (checker_img, checker_3d, checker_2d, checker_local) if extra_output else (checker_3d, checker_2d, checker_local)

    if charuco_only:
        return chrk, (avg_errors, all_errors)
    else:
        return chrk, chck, (avg_errors, all_errors)
