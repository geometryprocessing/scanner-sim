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
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.morphology as morph


def gray_to_bin(num):
    num = np.bitwise_xor(num, np.right_shift(num, 16))
    num = np.bitwise_xor(num, np.right_shift(num, 8))
    num = np.bitwise_xor(num, np.right_shift(num, 4))
    num = np.bitwise_xor(num, np.right_shift(num, 2))
    num = np.bitwise_xor(num, np.right_shift(num, 1))
    return num


def get_single_bit_mask(filename, inverted_filename, undistort=None, plot=False, **kw):
    image = load_openexr(filename, make_gray=True)
    inverted = load_openexr(inverted_filename, make_gray=True)
    print("Loaded", filename)

    if undistort is not None:
        image = cv2.undistort(image, undistort["mtx"], undistort["dist"], None, undistort["new_mtx"])
        inverted = cv2.undistort(inverted, undistort["mtx"], undistort["dist"], None, undistort["new_mtx"])

    bit_mask = image > inverted

    if plot:
        plot_image(1 * bit_mask, filename + " - Bit Mask")

    return bit_mask


def get_all_bit_masks(template, inverted_template, ids, undistort=None, **kw):
    filenames = [template % id for id in ids]
    inverted_filenames = [inverted_template % id for id in ids]

    jobs = [joblib.delayed(get_single_bit_mask, check_pickle=False)
            (filename, inverted_filename, undistort=undistort, plot=False, **kw)
            for filename, inverted_filename in zip(filenames, inverted_filenames)]

    return np.array(joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs))


def decode_single(data_path, out_dir="decoded", mask_sigma=3, undistort=None, save=True, plot=False, save_figures=True, **kw):
    bit_masks = [get_all_bit_masks(data_path + dir + "_%d.exr", data_path + dir + "_%d_inv.exr",
                                   ids=range(11), undistort=undistort, **kw)
                 for dir in ["horizontal", "vertical"]]

    print(bit_masks[0].shape, bit_masks[0].size)

    blank = load_openexr(data_path + "blank.exr", make_gray=True)
    white = load_openexr(data_path + "white.exr", make_gray=True)

    ldr, thr_ldr = linear_map(gaussian_filter(white - blank, sigma=mask_sigma))
    thr_otsu, mask = cv2.threshold(ldr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Thresholds:", thr_ldr, thr_otsu)

    struct = scipy.ndimage.generate_binary_structure(2, 1)
    mask = morph.binary_erosion(mask, struct, 2)

    # Old: filters away parts of the image where projected pattern got blurred too much
    # diff = np.abs(hor[0, ...] - hor_i[0, ...])
    # diff = gaussian_filter(diff, sigma=3)
    # d_mask = diff > 0.07 * np.max(diff)
    # mask &= morph.binary_erosion(d_mask, struct, 2)

    idx = np.nonzero(~mask)
    bit_masks[0][:, idx[0], idx[1]] = 0
    bit_masks[1][:, idx[0], idx[1]] = 0

    h, v = np.zeros_like(mask, dtype=np.int), np.zeros_like(mask, dtype=np.int)

    for i in range(11):
        h = np.bitwise_or(h, np.left_shift(bit_masks[0][i, ...].astype(np.int), i))
        v = np.bitwise_or(v, np.left_shift(bit_masks[1][i, ...].astype(np.int), i))

    h, v = gray_to_bin(h), gray_to_bin(v)
    r, c = np.nonzero((h > 0) & (v > 0))
    p_r, p_c = h[r, c].ravel(), v[r, c].ravel()
    cam_xy = np.stack([c, r], axis=1)
    proj_xy = np.stack([p_c, p_r], axis=1)

    if save:
        save_path = data_path + "/" + out_dir + "/"
        ensure_exists(save_path)

        np.save(save_path + "camera.npy", cam_xy.astype(np.uint16))
        np.save(save_path + "projector.npy", proj_xy.astype(np.uint16))
    else:
        save_path = None

    if plot:
        if not save_figures:
            save_path = None

        plot_image(ldr, data_path + " - LDR Image", save_as=save_path + "ldr" if save_path else None)
        plot_hist(ldr.ravel(), data_path + " - LDR Hist", bins=256, save_as=save_path + "ldr_hist" if save_path else None)
        plot_image(1 * mask, data_path + " - Mask", save_as=save_path + "mask" if save_path else None)

    return cam_xy, proj_xy


def load_decoded(data_path):
    return np.load(data_path + "/camera.npy"), np.load(data_path + "/projector.npy")


if __name__ == "__main__":
    # data_path = "D:/scanner_sim/captures/plane/default_scan/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_330/default_scan/"
    data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/default_scan/"

    # get_single_bit_mask(data_path + "horizontal_0.exr", data_path + "horizontal_0_inv.exr", plot=True)

    cam, proj = decode_single(data_path, plot=True)
    cam, proj = load_decoded(data_path + "decoded/")
    print(cam.shape)

    plt.show()
