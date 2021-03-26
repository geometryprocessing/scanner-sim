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
    wb = numpinize(json.load(open("../calibration/projector/white_balance.json")))
    print("White Balance:", wb)

    data_path_template = "D:/scanner_sim/captures/stage_batch_2/%s_30_deg/position_0/color/"
    for object in ["dodo", "avocado", "house", "chair", "vase", "bird", "radio"]:
    # for object in ["dodo"]:
        data_path = data_path_template % object
        d = load_openexr(data_path + "../gray/img_01.exr")
        r = load_openexr(data_path + "red.exr") - d
        g = load_openexr(data_path + "green.exr") - d
        b = load_openexr(data_path + "blue.exr") - d
        # g_r = np.average(g/r)
        # g_b = np.average(g/b)
        # print(g_r, g_b)
        rgb = np.stack([r * wb["g/r"], g, b * wb["g/b"]], axis=2)
        # rgb = np.stack([r * g_r, g, b * g_b], axis=2)
        # rgb = np.stack([r, g, b], axis=2)
        print(rgb.shape)
        filename = data_path + "%s_color.exr" % object
        save_openexr(filename, rgb, keep_rgb=True)
        print("Svaed", filename)
