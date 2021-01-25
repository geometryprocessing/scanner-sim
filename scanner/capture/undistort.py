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


if __name__ == "__main__":
    data_path = "D:/scanner_sim/captures/plane/gray/decoded/"
    data_path = "D:/scanner_sim/captures/plane/default_scan/decoded/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_330/default_scan/decoded/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/rook_30_deg/position_330/default_scan/decoded/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/shapes_30_deg/position_330/default_scan/decoded/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/default_scan/decoded/"
    data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/gray/decoded/"

    cam_calib = load_camera_calibration("../calibration/camera/camera_calibration.json")
    proj_calib = load_projector_calibration("../calibration/projector/projector_calibration.json")[2]

    # patterns_path =  "D:/scanner_sim/captures/patterns/gray/"
    # patterns = glob.glob(patterns_path + "*.png")
    # print("Found %d patterns:" % len(patterns), patterns)
    # ensure_exists(patterns_path + "predistorted/")

    # for pattern in patterns:
    #     original = load_ldr(pattern)
    #     undistorted = cv2.undistort(original, proj_calib["mtx"], proj_calib["dist"], newCameraMatrix=proj_calib["new_mtx"])
    #     new_filename = patterns_path + "predistorted/" + os.path.basename(pattern)
    #     print(new_filename)
    #     save_ldr(new_filename, undistorted)

    # images_path = "D:/scanner_sim/captures/plane/gray/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_0/color/"
    images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_0/gray/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/rook_30_deg/position_0/gray/"
    # images_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/shapes_30_deg/position_0/gray/"
    images = glob.glob(images_path + "*.exr")
    print("Found %d images:" % len(images), images)
    ensure_exists(images_path + "undistorted/")

    for image in images:
        original = load_openexr(image)
        undistorted = cv2.undistort(original, cam_calib["mtx"], cam_calib["dist"], newCameraMatrix=cam_calib["new_mtx"])
        new_filename = images_path + "undistorted/" + os.path.basename(image)
        print(new_filename)
        save_openexr(new_filename, undistorted)
