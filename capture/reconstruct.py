import os
import cv2
import time
import json
import queue
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph
from sklearn.decomposition import PCA

from hdr import *
from projector import *


def gray_to_bin(num):
    num = np.bitwise_xor(num, np.right_shift(num, 16))
    num = np.bitwise_xor(num, np.right_shift(num, 8))
    num = np.bitwise_xor(num, np.right_shift(num, 4))
    num = np.bitwise_xor(num, np.right_shift(num, 2))
    num = np.bitwise_xor(num, np.right_shift(num, 1))
    return num


def load_cached(path):
    if not os.path.isfile(path + "dark.npy"):
        dark = load_openexr(path + "0.exr")
        np.save(path + "dark", np.array(dark))

    if not os.path.isfile(path + "white.npy"):
        white = load_openexr(path + "1.exr")
        np.save(path + "white", np.array(white))

    if not os.path.isfile(path + "hor.npy") or not os.path.isfile(path + "hor_i.npy"):
        hor, hor_i = [], []
        for i in range(11):
            hor.append(load_openexr(path + "%d.exr" % (100 + i)))
            hor_i.append(load_openexr(path + "%d.exr" % (200 + i)))

        np.save(path + "hor", np.array(hor))
        np.save(path + "hor_i", np.array(hor_i))

    if not os.path.isfile(path + "ver.npy") or not os.path.isfile(path + "ver_i.npy"):
        ver, ver_i = [], []
        for i in range(11):
            ver.append(load_openexr(path + "%d.exr" % (300 + i)))
            ver_i.append(load_openexr(path + "%d.exr" % (400 + i)))

        np.save(path + "ver", np.array(ver))
        np.save(path + "ver_i", np.array(ver_i))

    return np.load(path + "dark.npy"), np.load(path + "white.npy"), \
           np.load(path + "hor.npy"), np.load(path + "hor_i.npy"), \
           np.load(path + "ver.npy"), np.load(path + "ver_i.npy")


if __name__ == "__main__":
    path = "plane/"

    # dark, white, hor, hor_i, ver, ver_i = load_cached(path)
    #
    # hor -= dark
    # i = 0
    # save_openexr(path + "subtracted%d.exr" % i, hor[i, ...])
    #
    # white -= dark
    # hor -= dark
    # hor_i -= dark
    # ver -= dark
    # ver_i -= dark
    #
    # mask = white > 0.05*np.max(white)
    # struct = scipy.ndimage.generate_binary_structure(2, 2)
    # mask = morph.binary_erosion(mask, struct, 2)
    # idx = np.nonzero(mask)
    #
    # hor_m = hor > hor_i
    # ver_m = ver > ver_i
    # hor2 = np.zeros_like(hor_m)
    # ver2 = np.zeros_like(ver_m)
    # hor2[:, idx[0], idx[1]] = hor_m[:, idx[0], idx[1]]
    # ver2[:, idx[0], idx[1]] = ver_m[:, idx[0], idx[1]]
    #
    # h = np.zeros_like(mask, dtype=np.int)
    # v = np.zeros_like(mask, dtype=np.int)
    #
    # for i in range(11):
    #     h = np.bitwise_or(h, np.left_shift(hor2[i, ...].astype(np.int), i))
    #     v = np.bitwise_or(v, np.left_shift(ver2[i, ...].astype(np.int), i))
    #
    # h = gray_to_bin(h)
    # v = gray_to_bin(v)
    # r, c = np.nonzero((h > 0) & (v > 0))
    # p_r, p_c = h[r, c].ravel(), v[r, c].ravel()
    # cam = np.stack([c, r], axis=1)
    # proj = np.stack([p_c, p_r], axis=1)
    #
    # np.save(path + "cam", cam)
    # np.save(path + "proj", proj)

    cam, proj = np.load(path + "cam.npy"), np.load(path + "proj.npy")

    with open("../calibration/camera/refined_calibration.pkl", "rb") as f:
        ret, cam_mtx, cam_dist, rvecs, tvecs = pickle.load(f)

    w, h = 6464, 4852
    new_cam_mtx, cam_roi = cv2.getOptimalNewCameraMatrix(cam_mtx, cam_dist, (w, h), 1, (w, h))

    with open("../calibration/projector/calibration.pkl", "rb") as f:
        origin, R, proj_mtx, proj_dist = pickle.load(f)

    w, h = 1920, 1080
    new_proj_mtx, proj_roi = cv2.getOptimalNewCameraMatrix(proj_mtx, proj_dist, (w, h), 1, (w, h))

    cam2 = cv2.undistortPoints(cam.astype(np.float), cam_mtx, cam_dist).reshape((-1, 2))
    proj2 = cv2.undistortPoints(proj.astype(np.float), proj_mtx, proj_dist).reshape((-1, 2))

    cam_3d = np.concatenate([cam2, np.ones((cam2.shape[0], 1))], axis=1)
    proj_3d = np.concatenate([proj2, np.ones((proj2.shape[0], 1))], axis=1)

    # cam_3d = np.stack([(cam2[:, 0] - new_cam_mtx[0, 2]) / new_cam_mtx[0, 0],
    #                    (cam2[:, 1] - new_cam_mtx[1, 2]) / new_cam_mtx[1, 1], np.ones((cam2.shape[0]))], axis=1)
    #
    # proj_3d = np.stack([(proj2[:, 0] - new_proj_mtx[0, 2]) / new_proj_mtx[0, 0],
    #                     (proj2[:, 1] - new_proj_mtx[1, 2]) / new_proj_mtx[1, 1], np.ones((proj2.shape[0]))], axis=1)

    proj_3d = np.matmul(R, proj_3d.T).T

    v12 = np.sum(np.multiply(cam_3d, proj_3d), axis=1)
    v1 = np.linalg.norm(cam_3d, axis=1)**2
    v2 = np.linalg.norm(proj_3d, axis=1)**2
    L = (np.matmul(cam_3d, origin) * v2 + np.matmul(proj_3d, -origin) * v12) / (v1 * v2 - v12**2)

    p = cam_3d * L[:, None]

    # plt.figure()
    # plt.plot(proj2[::1000, 0], proj2[::1000, 1], '.')

    plt.figure("Reconstruction", (12, 12))
    ax = plt.subplot(111, projection='3d', proj_type='ortho')
    ax.set_title("Reconstruction")
    scatter(ax, p[::3000, :].T, label="p")
    # scatter(ax, 400*cam_3d[::5000, :].T, label="cam")
    # scatter(ax, np.array([0, 0, 0]), label="cam0")
    # scatter(ax, origin[:, None] + 150*proj_3d[::5000, :].T, label="proj")
    # scatter(ax, origin, label="proj0")

    ax.set_xlabel("x, mm")
    ax.set_ylabel("z, mm")
    ax.set_zlabel("-y, mm")
    plt.legend()
    plt.tight_layout()
    axis_equal_3d(ax)

    pca = PCA(n_components=3)
    p2 = pca.fit_transform(p)

    # p -= np.average(p, axis=0)
    # uu, dd, vv = np.linalg.svd(p)
    # print(vv)

    plt.figure("Plane Reconstruction", (12, 9))
    plt.hist(p2[:, 2], bins=1000)
    mean, std = np.mean(p2[:, 2]), np.std(p2[:, 2])
    print(mean, std)
    plt.title("Plane Reconstruction (std = %.5f)" % std)
    plt.xlabel("Error, mm")
    plt.tight_layout()
    plt.savefig(path + "plane_reconstruction_errors.png", dpi=160)


    # plt.figure()
    # plt.imshow(hor[5, ...], vmax=s)
    # plt.imshow(hor_m[0, ...])
    # plt.imshow(h)
    # plt.figure()
    # plt.imshow(v)
    # plt.colorbar()
    # plt.tight_layout()

    plt.show()
    print("Done")
