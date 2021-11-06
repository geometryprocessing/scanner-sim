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


def predistord_patterns(patterns_path, proj_calib, x4=False):
    patterns = glob.glob(patterns_path + "*.png")
    print("Found %d patterns:" % len(patterns), patterns)
    ensure_exists(patterns_path + "predistorted/")

    mtx, dist, new_mtx = proj_calib["mtx"], proj_calib["dist"], proj_calib["new_mtx"]
    print("new_mtx:\n", new_mtx)

    if x4:
        mtx[:2, :] *= 4
        new_mtx[:2, :] *= 4
        # test_mtx = new_mtx.copy()
        # test_mtx[:2, :] *= 4
        # print("test_mtx:\n", test_mtx)
        # w, h = 1920, 1080
        # opt_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, proj_calib["dist"], (w*4, h*4), 1, (w*4, h*4))
        # print("opt_mtx:\n", opt_mtx)
        # # new_mtx = opt_mtx
        # new_mtx = test_mtx  # Simply scaling works better
        # print("diff_mtx:\n", opt_mtx - test_mtx, mtx[1, 1] - mtx[0, 0])
        reference = load_ldr(patterns_path + "predistorted/reference.png")
        reference = np.repeat(np.repeat(reference, 4, axis=0), 4, axis=1)

    print("new_mtx" + ("_x4:\n" if x4 else ":\n"), new_mtx)

    for pattern in patterns:
        original = load_ldr(pattern)

        if x4:
            original = np.repeat(np.repeat(original, 4, axis=0), 4, axis=1)

        undistorted = cv2.undistort(original, mtx, dist, newCameraMatrix=new_mtx)
        new_filename = patterns_path + "predistorted/" + os.path.basename(pattern)
        save_ldr(new_filename, undistorted)
        print(new_filename)

        if x4:
            save_ldr(patterns_path + "predistorted/reference_diff.png", np.abs(reference - undistorted))


def undistort_images(images_path, cam_calib):
    images = glob.glob(images_path + "*.exr")
    ensure_exists(images_path + "undistorted/")
    print("Found %d images:" % len(images), images)

    for image in images:
        original = load_openexr(image)
        undistorted = cv2.undistort(original, cam_calib["mtx"], cam_calib["dist"], newCameraMatrix=cam_calib["new_mtx"])
        new_filename = images_path + "undistorted/" + os.path.basename(image)
        save_openexr(new_filename, undistorted)
        print(new_filename)


def remove_parasitic_light(images_path, patterns=("checker.exr"), parasitic="blank.exr", plot=False):
    par, clean = load_openexr(images_path + parasitic), None

    for pattern in patterns:
        pat = load_openexr(images_path + pattern)
        clean = np.maximum(pat - par, 0)
        new_filename = images_path + pattern[:-4] + "_clean.exr"
        save_openexr(new_filename, clean)
        print(new_filename)

    if plot:
        plt.figure("Parasitic Light", (12, 9))
        plt.imshow(par)
        plt.colorbar()
        plt.tight_layout()

        if clean is not None:
            plt.figure("Last Pattern (Clean)", (12, 9))
            plt.imshow(clean)
            plt.colorbar()
            plt.tight_layout()


if __name__ == "__main__":
    cam_calib = load_camera_calibration("../calibration/camera/camera_calibration.json")
    proj_calib = load_projector_calibration("../calibration/projector/projector_calibration.json")[2]

    # patterns_path = "../capture/patterns/checker/"
    # predistord_patterns(patterns_path, proj_calib, x4=True)
    # exit(0)

    # images_path = "D:/scanner_sim/captures/plane/gray/"
    # images_path = "D:/scanner_sim/captures/stage_batch_3/pawn_30_deg_matte/position_0/gray/"
    # images_path = "D:/scanner_sim/captures/stage_batch_3/pawn_30_deg_gloss/position_0/gray/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_0/color/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_0/gray/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/rook_30_deg/position_0/gray/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/shapes_30_deg/position_0/gray/"
    # images_path = "D:/scanner_sim/calibration/accuracy_test/charuco_plane/gray/"
    # images_path = "D:/scanner_sim/calibration/accuracy_test/charuco_plane/color/"
    # images_path = "D:/scanner_sim/calibration/accuracy_test/clear_plane/gray/"
    # images_path = "D:/scanner_sim/calibration/accuracy_test/clear_plane/color/"
    images_path = "/media/yurii/EXTRA/scanner-sim-data/material_calib_2_deg/position_44/"

    undistort_images(images_path, cam_calib)

    # remove_parasitic_light(images_path + "undistorted/", patterns=("checker.exr", "white.exr"), plot=True)
    # plt.show()
