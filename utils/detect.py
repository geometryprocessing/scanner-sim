import os
import glob
import json
import joblib
import cv2
from cv2 import aruco
# print(cv2.__version__)
from utils import *
import numpy as np


# Detect checker board on gray scale image. Use pre_scale to do the initial pass on a lower resolution image.
# Use draw_scale to do reduce the resolution of an overlay image.
# Board origin: top-left. First axis: X to the right. Second axis: Y down
def detect_checker(gray, n=11, m=8, size=20, pre_scale=1, draw_on=None, draw_scale=1):
    assert(len(gray.shape) == 2)
    assert(gray.dtype == np.uint8)
    pre_scale = pre_scale or 1

    flags = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH

    if pre_scale > 1.5:
        gray2 = cv2.resize(gray, (gray.shape[1] // pre_scale, gray.shape[0] // pre_scale))
        ret, corners2 = cv2.findChessboardCorners(gray2, (n, m), flags=flags)
        if ret:
            corners = corners2 * pre_scale
    else:
        ret, corners = cv2.findChessboardCorners(gray, (n, m), flags=flags)

    if ret:
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (n, m), (-1, -1), criteria)
    else:
        corners = None

    if corners is not None:
        img_points = corners.reshape((m, n, 2))
        obj_points = np.zeros((n * m, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2) * size
        ids = np.arange(n*m)
    else:
        img_points, obj_points, ids = None, None, None

    if draw_on is not None:
        if draw_scale > 1.5:
            draw_on = cv2.resize(draw_on, (draw_on.shape[1] // draw_scale, draw_on.shape[0] // draw_scale))
        img = cv2.drawChessboardCorners(draw_on, (n, m), corners, ret)
    else:
        img = None

    return (img_points, obj_points, ids), img


# Detect charuco board on a gray scale image. Use pre_scale to do the initial pass on a lower resolution image.
# Use draw_scale to do reduce the resolution of an overlay image.
# Board origin: bottom-left. First axis: X to the right. Second axis: Y up
def detect_charuco(gray, n=25, m=18, size=15, pre_scale=1, draw_on=None, draw_scale=1):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(n, m, size, size * 12 / 15, aruco_dict)

    if pre_scale > 1.5:
        gray2 = cv2.resize(gray, (gray.shape[1] // pre_scale, gray.shape[0] // pre_scale))
        m_pos2, m_ids, _ = cv2.aruco.detectMarkers(gray2, aruco_dict)
        m_pos = [m * pre_scale for m in m_pos2]
    else:
        m_pos, m_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if len(m_pos) > 0:
        count, c_pos, c_ids = cv2.aruco.interpolateCornersCharuco(m_pos, m_ids, gray, board)
    else:
        count, c_pos, c_ids = 0, None, None

    if count:
        img_points, obj_points = np.array(c_pos).reshape((-1, 2)), board.chessboardCorners[c_ids].reshape((-1, 3))
        ids = c_ids.ravel()
    else:
        img_points, obj_points, ids = None, None, None

    if draw_on is not None:
        if draw_scale > 1.5:
            draw_on = cv2.resize(draw_on, (draw_on.shape[1] // draw_scale, draw_on.shape[0] // draw_scale))
        if len(draw_on.shape) == 2:
            draw_on = np.repeat(draw_on[:, :, None], 3, axis=2)
        img = aruco.drawDetectedMarkers(draw_on, np.array(m_pos) / draw_scale, m_ids)
        img = aruco.drawDetectedCornersCharuco(img, np.array(c_pos) / draw_scale, c_ids, cornerColor=(0, 255, 0))
    else:
        img = None

    return (img_points, obj_points, ids), img


# Load image from file and detect calibration board using detect_func. Save or plot an overlay image if need be
def detect_single(filename, detect_func, resize=1, out_dir="detected", draw=False, save=False, plot=False, return_image=True, **kw):
    assert(type(filename) is str)

    img = cv2.imread(filename)
    if resize > 1.5:
        img = cv2.resize(img, (img.shape[1] // resize, img.shape[0] // resize))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points, img = detect_func(gray, draw_on=img if draw else None, **kw)

    success = points[2] is not None
    print(filename, " - Success" if success else " - Failed")

    new_filename = None
    if save:
        if draw:
            path = os.path.dirname(filename) + "/" + out_dir + "/"
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            new_filename = path + os.path.basename(filename)[:-4] + ".png"
            cv2.imwrite(new_filename, img)
        else:
            print("Nothing new to save because draw=False")

    if plot:
        cv2.imshow(filename, img)

        while cv2.getWindowProperty(filename, 0) >= 0:
            cv2.waitKey(50)

    return points, new_filename, img if return_image else None


# Process all images that match a filename_template. Save valid results as json file if need be
def detect_all(filename_template, detect_func, out_dir="detected", save_json=True, **kw):
    filenames = glob.glob(filename_template)

    jobs = [joblib.delayed(detect_single)
            (name, detect_func, return_image=False, out_dir=out_dir, **kw) for name in filenames]

    results = joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)

    ret = {}
    for filename, result in zip(filenames, results):
        name, points = os.path.basename(filename), result[0]

        if points[2] is not None:
            if save_json:
                points = points[0].tolist(), points[1].tolist(), points[2].tolist()
            ret[name] = {"img_points": points[0], "obj_points": points[1], "ids": points[2]}

    if save_json:
        path = os.path.dirname(filename_template) + "/" + out_dir
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        with open(path + "/corners.json", "w") as f:
            json.dump(ret, f, indent=4)

    return ret


def load_corners(filename):
    corners = {}

    for name, points in json.load(open(filename, "r")).items():
        img_points = np.array(points["img_points"]).reshape(-1, 2).astype(np.float32)
        obj_points = np.array(points["obj_points"]).reshape(-1, 3).astype(np.float32)
        ids = np.array(points["ids"]).ravel().astype(np.int32)
        corners[name] = {"img": img_points, "obj": obj_points, "idx": ids}

    return corners


def test(data_path):
    filename = data_path + "checker/checker (1).bmp"

    detect_single(filename, detect_checker, draw=True, save=True, pre_scale=4, draw_scale=1)
    points, new_filename, img = detect_single(filename, detect_checker, resize=5, draw=True, plot=True)
    print([p.shape for p in points], new_filename, img.shape)

    filename = data_path + "charuco/charuco (1).bmp"

    detect_single(filename, detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)
    points, new_filename, img = detect_single(filename, detect_charuco, resize=4, draw=True, plot=True)
    print([p.shape for p in points], new_filename, img.shape)


if __name__ == "__main__":
    # data_path = "D:/Scanner/Calibration/camera_intrinsics/data/"

    data_path = "/home/vida/data/scanner-sim/accuracy_test/charuco_plane/combined/"
    filename = data_path + "blank_0.png"
    detect_single(filename, detect_charuco, draw=True, save=False, plot=True, pre_scale=2, draw_scale=2)


    # test(data_path)

    # detect_all(data_path + "/checker/*.bmp", detect_checker, draw=True, save=True, pre_scale=5, draw_scale=1)
    # detect_all(data_path + "/charuco/*.bmp", detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)

    # data_path = "D:/scanner_sim/calibration/accuracy_test/projector_calib/"
    # detect_all(data_path + "/charuco/*.bmp", detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)
