import itertools
import os
import pickle

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
from skimage import measure
from skimage import filters
from camera import load_camera_calibration


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
        image = cv2.undistort(image, undistort["mtx"], undistort["dist"], newCameraMatrix=undistort["new_mtx"])
        inverted = cv2.undistort(inverted, undistort["mtx"], undistort["dist"], newCameraMatrix=undistort["new_mtx"])

    bit_mask = image > inverted

    if plot:
        plot_image(1 * bit_mask, filename + " - Bit Mask")

    return bit_mask


def get_all_bit_masks(template, inverted_template, ids=None, undistort=None, **kw):
    if ids is not None:
        filenames = [template % id for id in ids]
        inverted_filenames = [inverted_template % id for id in ids]
    else:
        filenames, inverted_filenames = template, inverted_template

    jobs = [joblib.delayed(get_single_bit_mask, check_pickle=False)
            (filename, inverted_filename, undistort=undistort, plot=False, **kw)
            for filename, inverted_filename in zip(filenames, inverted_filenames)]

    return np.array(joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs))


def decode_single(data_path, symmetric=True, out_dir="decoded", mask_sigma=3, mask_iter=6, crop=None, offset=-150,
                  undistort=None, group=False, save=True, plot=False, save_figures=True, verbose=False, **kw):

    if symmetric:
        all_names = ["img_%02d.exr" % i for i in range(46)]
        v_names, v_inv_names = reversed(all_names[2:24:2]), reversed(all_names[3:24:2])
        h_names, h_inv_names = reversed(all_names[24::2]), reversed(all_names[25::2])

        h_masks = get_all_bit_masks([data_path + n for n in h_names], [data_path + n for n in h_inv_names], undistort=undistort, **kw)
        v_masks = get_all_bit_masks([data_path + n for n in v_names], [data_path + n for n in v_inv_names], undistort=undistort, **kw)
        bit_masks = h_masks, v_masks
    else:
        bit_masks = [get_all_bit_masks(data_path + dir + "_%d.exr", data_path + dir + "_%d_inv.exr",
                                       ids=range(11), undistort=undistort, **kw)
                     for dir in ["horizontal", "vertical"]]

    if verbose:
        print("Bit masks:", bit_masks[0].shape, bit_masks[0].size / 1024**2, "MB")

    if symmetric:
        white = load_openexr(data_path + "/img_00.exr", make_gray=True)
        blank = load_openexr(data_path + "/img_01.exr", make_gray=True)
    else:
        blank = load_openexr(data_path + "/blank.exr", make_gray=True)
        white = load_openexr(data_path + "/white.exr", make_gray=True)

    clean = white - blank
    if crop:
        clean[:, :crop] = 0  # crop to the left of the rotating stage
        clean[:, clean.shape[1] - crop + 2*offset:] = 0  # and to the right
    ldr, thr_ldr = linear_map(gaussian_filter(clean, sigma=mask_sigma))
    thr_otsu, mask = cv2.threshold(ldr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Thresholds:", thr_ldr, thr_otsu)

    struct = scipy.ndimage.generate_binary_structure(2, 1)
    mask = morph.binary_erosion(mask, struct, mask_iter)

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

    if verbose:
        print("Horizontal Range:", [np.min(h), np.max(h)])
        print("Vertical Range:", [np.min(v), np.max(v)])

    r, c = np.nonzero((h > 0) & (v > 0))
    p_r, p_c = h[r, c].ravel(), v[r, c].ravel()

    if symmetric:
        p_r -= 1024 - 1080 // 2
        p_c -= 1024 - 1920 // 2

    cam_xy, proj_xy = np.stack([c, r], axis=1), np.stack([p_c, p_r], axis=1)

    if group or plot:
        img = np.zeros_like(mask, dtype=np.int)
        img[r, c] = 1920 * p_r + p_c

    if group:
        labels = measure.label(img, background=0, connectivity=2)

        group_cam_xy, group_proj_xy, group_counts, group_rcs = [], [], [], []
        for i, region in enumerate(measure.regionprops(labels)):
            if i % 10000 == 0 and verbose:
                print("Group", i)
            # if i > 20000:
            #     break

            if region.label > 0:
                group_counts.append(region.coords.shape[0])
                group_rcs.append(region.coords)
                group_cam_xy.append(region.centroid[::-1])
                r0, c0 = region.coords[0, :]
                xy = np.array([v[r0, c0], h[r0, c0]])
                if symmetric:
                    xy -= [1024 - 1920 // 2, 1024 - 1080 // 2]
                group_proj_xy.append(xy)

        group_cam_xy, group_proj_xy, group_counts = np.array(group_cam_xy), np.array(group_proj_xy), np.array(group_counts)
        print("Groups:", group_cam_xy.shape)

    if save:
        save_path = data_path + "/" + out_dir + "/"
        ensure_exists(save_path)

        np.save(save_path + "camera_xy.npy", cam_xy.astype(np.uint16))
        np.save(save_path + "projector_xy.npy", proj_xy.astype(np.uint16))
        np.save(save_path + "mask.npy", mask)
        with open(save_path + "undistorted.txt", "w") as f:
            f.write(str(undistort is not None))

        if group:
            np.save(save_path + "group_cam_xy.npy", group_cam_xy.astype(np.float32))
            np.save(save_path + "group_proj_xy.npy", group_proj_xy.astype(np.uint16))
            np.save(save_path + "group_counts.npy", group_counts.astype(np.uint32))
            with open(save_path + "group_rcs.pkl", "wb") as f:
                pickle.dump(group_rcs, f)
    else:
        save_path = None

    if plot:
        if not save_figures:
            save_path = None

        plot_image(ldr, "Image", data_path + " - LDR Image", save_as=save_path + "ldr" if save_path else None)
        plot_hist(ldr.ravel(), "Hist", data_path + " - LDR Hist", bins=256, save_as=save_path + "ldr_hist" if save_path else None)
        plot_image(1 * mask, "Mask", data_path + " - Mask", save_as=save_path + "mask" if save_path else None)
        img[r, c] = p_r
        plot_image(img, "Horizontal", data_path + " - Decoded Horizontal", save_as=save_path + "horizontal" if save_path else None)
        img[r, c] = p_c
        plot_image(img, "Vertical", data_path + " - Decoded Vertical", save_as=save_path + "vertical" if save_path else None)

        if group:
            m = np.max(labels)
            lut, idx = np.random.randint(m, size=(m,)), np.nonzero(labels > 0)
            labels[idx] = lut[labels[idx] - 1] + 1
            plot_image(labels, "Groups", data_path + " - Groups", save_as=save_path + "groups" if save_path else None)
            plot_hist(group_counts, "Counts", data_path + " - Group Counts", bins=np.max(group_counts), save_as=save_path + "group_counts" if save_path else None)

    return (cam_xy, proj_xy, mask), (group_cam_xy, group_proj_xy, group_counts, group_rcs) if group else None


def load_decoded(path):
    all = np.load(path + "/camera_xy.npy"), np.load(path + "/projector_xy.npy"), np.load(path + "/mask.npy")

    if os.path.exists(path + "/group_cam_xy.npy"):
        groups = [np.load(path + "/group_cam_xy.npy"), np.load(path + "/group_proj_xy.npy"),
                  np.load(path + "/group_counts.npy"), None]
        with open(path + "/group_rcs.pkl", "rb") as f:
            groups[3] = pickle.load(f)
    else:
        groups = None

    return all, groups


def decode_many(path_template, suffix="gray/", **kw):
    paths = glob.glob(path_template)
    print("Found %d directories:" % len(paths), paths)

    for i, path in enumerate(paths):
        print("Decoding %d:" % i, path + "/" + suffix)
        plt.close("all")
        decode_single(path + "/" + suffix, **kw)


if __name__ == "__main__":
    camera_calib = load_camera_calibration("../calibration/camera/camera_calibration.json")

    # Debug / Development
    # data_path = "D:/scanner_sim/captures/plane/default_scan/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/pawn_30_deg/position_330/default_scan/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/rook_30_deg/position_330/default_scan/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/shapes_30_deg/position_330/default_scan/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/default_scan/"
    # get_single_bit_mask(data_path + "horizontal_0.exr", data_path + "horizontal_0_inv.exr", undistort=camera_calib, plot=True)

    # all, groups = decode_single(data_path, undistort=camera_calib, symmetric=False, group=True, plot=True, verbose=True)

    # all, groups = load_decoded(data_path + "decoded")
    # print(all[0].shape, groups[0].shape if groups is not None else "")

    # Planes
    # data_path = "D:/scanner_sim/captures/plane/gray/"
    # data_path = "D:/scanner_sim/captures/stage_batch_2/no_ambient/material_calib_2_deg/position_84/gray/"
    # decode_single(data_path, undistort=camera_calib, symmetric=True, group=True, plot=True, verbose=True)

    data_path_template = "D:/scanner_sim/captures/stage_batch_2/%s_30_deg/position_*"
    for object in ["pawn", "rook", "shapes"]:
        decode_many(data_path_template % object, undistort=camera_calib, symmetric=True, crop=1500, group=True, plot=True, verbose=True)

    # data_path_template = "D:/scanner_sim/captures/stage_batch_2/no_ambient/%s_30_deg/position_*"
    # for object in ["pawn", "rook", "shapes"]:
    #     decode_many(data_path_template % object, undistort=camera_calib, symmetric=True, crop=1500, group=True, plot=True, verbose=True)

    # data_path_template = "D:/scanner_sim/captures/stage_batch_2/%s_30_deg/position_*"
    # for object in ["dodo", "avocado", "house", "chair", "vase", "bird", "radio"]:
    #     decode_many(data_path_template % object, undistort=camera_calib, symmetric=True, crop=1600, group=True, plot=True, verbose=True)

    plt.show()
