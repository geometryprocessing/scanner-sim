import json
import cv2
import imageio
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.morphology as morph
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from utils import *
from calibrate import *
from detect import *
from projector import *


def img_to_ray(p_img, mtx):
    p_img = (p_img - mtx[:2, 2]) / mtx[[0, 1], [0, 1]]
    return np.concatenate([p_img, np.ones((p_img.shape[0], 1))], axis=1)


def triangulate(cam_rays, proj_xy, proj_calib):
    u_proj_xy = cv2.undistortPoints(proj_xy.astype(np.float), proj_calib["mtx"], proj_calib["dist"]).reshape((-1, 2))
    proj_rays = np.concatenate([u_proj_xy, np.ones((u_proj_xy.shape[0], 1))], axis=1)
    proj_rays = np.matmul(proj_calib["basis"].T, proj_rays.T).T
    proj_origin = proj_calib["origin"]

    v12 = np.sum(np.multiply(cam_rays, proj_rays), axis=1)
    v1, v2 = np.linalg.norm(cam_rays, axis=1)**2, np.linalg.norm(proj_rays, axis=1)**2
    L = (np.matmul(cam_rays, proj_origin) * v2 + np.matmul(proj_rays, -proj_origin) * v12) / (v1 * v2 - v12**2)

    return cam_rays * L[:, None]


def test_accuracy(data_path, camera_calib, proj_calib, captured=None, rendered=None, save=False, plot=False, save_figures=True, **kw):
    cam_mtx, cam_dist, cam_new_mtx = camera_calib["mtx"], camera_calib["dist"], camera_calib["new_mtx"]
    charuco, checker, plane_errors = reconstruct_planes(data_path, camera_calib, extra_output=True, **kw)
    charuco_img, charuco_3d, charuco_id, charuco_frame = charuco
    checker_img, checker_3d, checker_2d, checker_local = checker
    avg_plane_errors, all_plane_errors = plane_errors
    w, h = 1920, 1080

    print("\nReconstructed:", len(checker_3d))
    print("Mean plane error", np.mean(avg_plane_errors))
    assert(len(checker_3d) == 1)

    T, R = charuco_frame[0]

    print("Plane location:")
    print("T:", T, "\nR:\n", R, "\n")

    cam_offsets = np.zeros(2)
    if rendered is not None:
        cam_offsets = np.array([(6464 - 1)/2, (4852 - 1)/2]) - cam_new_mtx[:2, 2]
    print("Camera offsets:", cam_offsets)

    u_cam_xy = cv2.undistortPoints(checker_img[0].astype(np.float), cam_new_mtx, None).reshape((-1, 2))
    cam_rays = np.concatenate([u_cam_xy, np.ones((u_cam_xy.shape[0], 1))], axis=1)

    # proj_offsets = np.array([0., 0.])
    proj_offsets = np.array([0.5, 0.5])
    print("Projector offsets:", proj_offsets)

    triangulated_3d = triangulate(cam_rays, checker_2d[0] + proj_offsets[None, :], proj_calib)
    prj_triangulated = cv2.projectPoints(triangulated_3d, np.eye(3), np.zeros(3), cam_new_mtx, None)[0].reshape(-1, 2)

    d0, d = np.linalg.norm(checker_3d[0], axis=1), np.linalg.norm(triangulated_3d, axis=1)
    err = np.linalg.norm(triangulated_3d - checker_3d[0], axis=1)
    print(err, "\nMean triangulation error:", np.mean(err), "mm")
    print("Mean triangulation bias:", np.mean(d - d0), "mm")

    if plot:
        plt.figure("Plane Reconstruction", (12, 12))
        plt.clf()
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Plane Reconstruction")

        scatter(ax, charuco_3d[0], c="g", s=5, label="Charuco Corners")
        scatter(ax, checker_3d[0], c="b", s=8, label="Checker Corners")
        board(ax, T, R, label="Charuco Board")

        ax.set_xlabel("x, mm")
        ax.set_ylabel("z, mm")
        ax.set_zlabel("-y, mm")
        plt.legend()
        plt.tight_layout()
        axis_equal_3d(ax)

        if save_figures:
            ax.view_init(elev=10, azim=-20)
            plt.savefig(data_path + "/reconstruction_view1.png", dpi=120)
            ax.view_init(elev=12, azim=26)
            plt.savefig(data_path + "/reconstruction_view2.png", dpi=120)

        if captured is not None and rendered is not None:
            img_1 = np.minimum(1.2 * captured/np.max(captured), 1.)
            img_2 = rendered/np.max(rendered)
            img_1[1:, 3:] = img_1[:-1, :-3]  # Correct for optical axis offset

            plt.figure("Image Overlay", (16, 10))
            plt.imshow(img_1 - img_2)
            # plt.imshow((img_1 + img_2) * 0.5)
            # plt.imshow(np.stack([img_1, img_2, np.zeros_like(img_1)], axis=2))
            plt.colorbar()
            plt.tight_layout()

            if save_figures:
                plt.savefig(data_path + "/image_overlay.png", dpi=120)

            if rendered is not None:
                detected, img = detect_checker(rendered, draw_on=None, n=17, m=8, size=100)
                prj_detected = detected[0].reshape((-1, 2)) if detected[0] is not None else None
            else:
                prj_detected = None

            plt.figure("Marker Overlay", (16, 10))
            plt.imshow(rendered if rendered is not None else captured)
            plt.colorbar()

            plt.plot(charuco_img[0][:, 0] + cam_offsets[0], charuco_img[0][:, 1] + cam_offsets[1], "b+", markersize=12, label="Detected Charuco")
            plt.plot(checker_img[0][:, 0] + cam_offsets[0], checker_img[0][:, 1] + cam_offsets[1], "bx", markersize=12, label="Detected Checker")

            prj_charuco = cv2.projectPoints(charuco_3d[0], np.eye(3), np.zeros(3), cam_new_mtx, None)[0].reshape(-1, 2)
            prj_checker = cv2.projectPoints(checker_3d[0], np.eye(3), np.zeros(3), cam_new_mtx, None)[0].reshape(-1, 2)

            plt.plot(prj_charuco[:, 0] + cam_offsets[0], prj_charuco[:, 1] + cam_offsets[1], "r+", markersize=9, label="Projected Charuco")
            if rendered is not None:
                if prj_detected is not None:
                    plt.plot(prj_detected[:, 0], prj_detected[:, 1], "rx", markersize=9, label="Rendered Checker")
            else:
                plt.plot(prj_triangulated[:, 0] + cam_offsets[0], prj_triangulated[:, 1] + cam_offsets[1], "rx", markersize=9, label="Triangulated Checker")

            plt.tight_layout()
            plt.legend()

            if save_figures:
                plt.savefig(data_path + "/marker_overlay.png", dpi=120)

            if prj_detected is not None:
                ch_d = np.linalg.norm(checker_img[0] + cam_offsets[None, :] - prj_detected, axis=1)
                print("Mean rendering error:", np.mean(ch_d), "camera pixels")

                plt.figure("Rendering errors", (16, 9))
                plt.subplot(211)
                plt.imshow((ch_d).reshape(8, 17))
                plt.title("Rendering errors, camera pixels")
                plt.colorbar()
                plt.subplot(212)
                plt.hist(ch_d, bins=50)
                plt.title("Mean = %.3f camera pixels" % np.mean(ch_d))
                plt.tight_layout()

                if save_figures:
                    plt.savefig(data_path + "/rendering_errors.png", dpi=120)

                plt.figure("Rendering Errors - Paper Edition", (4, 3.5))
                plt.hist(ch_d, bins=50, range=[0, 5])
                plt.xlabel("Rendering error, camera pixels")
                plt.ylabel("Checker corner counts")
                plt.tight_layout()

                if save_figures:
                    plt.savefig(data_path + "/rendering_errors_paper.png", dpi=300)


        plt.figure("Triangulation errors", (16, 9))
        plt.subplot(211)
        plt.imshow((d-d0).reshape(8, 17))
        plt.title("Triangulation errors, mm depth")
        plt.colorbar()
        plt.subplot(212)
        plt.hist(d-d0, bins=50)
        plt.title("Mean = %.3f mm" % np.mean(d-d0))
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/triangulation_errors.png", dpi=120)

    if save:
        with open(data_path + "plane_location.json", "w") as f:
            json.dump({"T": T,
                       "R": R,
                       "T_Help": "[x, y, z] in mm in camera's frame of reference",
                       "R_Help": "[ex, ey, ez]",
                       "Camera": "https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html"},
                      f, indent=4, cls=NumpyEncoder)

    return T, R


if __name__ == "__main__":
    camera_calib = load_calibration("calibration/camera/camera_geometry.json")
    proj_calib = load_calibration("calibration/projector/projector_geometry_test.json")

    data_path = "E:/scanner_sim/calibration/accuracy_test/charuco_plane/combined/"

    # captured = imageio.imread("accuracy/checker_undistorted_gamma.png")[:, :, 0]
    captured = imageio.imread("accuracy/checker_undistorted_tone.png")[:, :, 0]
    # rendered = imageio.imread("accuracy/textured_render.png")
    rendered = imageio.imread("accuracy/clear_render_offset.png")
    # rendered = np.repeat(np.repeat(imageio.imread("accuracy/rendered_half_res.png"), 2, axis=0), 2, axis=1)

    T, R = test_accuracy(data_path, camera_calib, proj_calib, captured=captured, rendered=rendered,
                         save=True, plot=True, save_figures=True)

    # Plane normals
    n_c = R[2, :]
    with open(data_path + "../gray/reconstructed/plane_location.json", "r") as f:
        n_r = np.array(json.load(f)["R"])[2, :]
    # n_c = np.array([0.54043317, -0.02033796, -0.8411411])  # From charuco markers
    # n_r = np.array([0.53988591,  0.01586706, -0.8415886])  # From point cloud PCA

    print("Plane reconstruction tilt:", np.arccos(np.dot(n_c, n_r)) * 180 / np.pi, "deg")

    plt.show()
