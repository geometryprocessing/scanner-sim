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


def compute_pca_variation(points, plot=True):
    pca = PCA(n_components=3)
    p2 = pca.fit_transform(points)

    if plot:
        plt.figure("PCA", (16, 9))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.hist(p2[:, i], bins=1000)
            mean, std = np.mean(p2[:, i]), np.std(p2[:, i])
            print(i, mean, std)
            plt.title("Component %d (mean = %.5f, std = %.5f)" % (i, mean, std))
            plt.xlabel("Variance, mm")
            plt.tight_layout()
            # if i == 2:
            #     plt.savefig(path + "plane_reconstruction_errors.png", dpi=160)


if __name__ == "__main__":
    data_path = "D:/scanner_sim/captures/plane/default_scan/decoded/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/decoded"
    data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/default_scan/decoded/"

    cam, proj = load_decoded(data_path)
    p = load_points(data_path + "/points.ply")
    print(p.shape, cam.shape)

    compute_pca_variation(p, plot=True)

    plt.show()
