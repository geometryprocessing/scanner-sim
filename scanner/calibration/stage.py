import json
import cv2
import numpy as np
from camera import load_camera_calibration
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from utils import *
from calibrate import *


def calibrate_axis(data_path, camera_calib, min_plane_points=80, min_circle_points=50, save=None, plot=False, save_figures=None, **kw):
    save = save or False
    save_figures = save_figures or save

    charuco, charuco_errors = reconstruct_planes(data_path,camera_calib, min_points=min_plane_points, charuco_only=True, **kw)
    charuco_3d, charuco_id, charuco_frame = charuco
    print("\nReconstructed planes:", len(charuco_3d))
    print("Mean plane error", np.mean(charuco_errors[0]))

    all_points, all_ids = np.concatenate(charuco_3d), np.concatenate(charuco_id)
    max_id = np.max(all_ids)
    c_centers, c_errors, p_errors = [], [], []
    print("Max corner id:", max_id)

    if plot:
        plt.figure("Circles", (12, 12))
        plt.clf()

    for i in range(max_id):
        cp = all_points[all_ids == i, :]

        if cp.shape[0] < min_circle_points:
            continue

        pca = PCA(n_components=3)
        cp2 = pca.fit_transform(cp)
        mean, sv, comp = pca.mean_, pca.singular_values_, pca.components_

        def circle_loss(p, xy):
            cx, cy, R = p
            x, y = xy[:, 0] - cx, xy[:, 1] - cy
            r = np.sqrt(x ** 2 + y ** 2)
            return r - R

        cx, cy, R = least_squares(circle_loss, [0, 0, 1], args=(cp2,))['x']
        c = np.array([cx, cy, 0])
        c_centers.append(mean + np.matmul(comp.T, c))
        c_errors.extend(circle_loss((cx, cy, R), cp2).tolist())

        if plot:
            plt.plot(cx, cy, ".")
            plt.plot(cp2[:, 0], cp2[:, 1], ".")
            phi = np.linspace(0, 2*np.pi, 100)
            plt.plot(cx + R * np.cos(phi), cy + R * np.sin(phi), "-")
            plt.title("Fitted Circles")
            plt.axis("equal")

    if save:
        data_path += "/stage/"
        ensure_exists(data_path)

        if plot and save_figures:
            plt.savefig(data_path + "circles.png", dpi=160)

    p, dir = fit_line(c_centers)
    if dir[1] > 0:  # ensure up direction
        dir = -dir

    axis_errors = np.array([point_line_dist(c_centers[i], p, p + dir) for i in range(len(c_centers))])
    angle = 180 * np.arccos(np.dot([0, -1, 0], dir)) / np.pi
    print("Fitted circles:", len(c_centers))
    print("Mean axis error, mm:", np.mean(axis_errors))
    print("Incline, deg:", angle)

    if plot:
        plt.figure("Stage Calibration", (12, 12))
        plt.clf()
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Stage Calibration")

        skip = 10
        for i in range(len(charuco_frame)):
            if i % skip == 0:
                Ti, Ri = charuco_frame[i]
                board(ax, Ti, Ri, label="Charuco Boards" if i == 0 else "")

        scatter(ax, np.concatenate(charuco_3d[::skip], axis=0), c="g", s=5, label="Detected Corners")
        scatter(ax, np.array(c_centers), c="r", s=15, label="Circle Centers")
        line(ax, p - 150 * dir, p + 150 * dir, "-b", label="Stage Axis")

        ax.set_xlabel("x, mm")
        ax.set_ylabel("z, mm")
        ax.set_zlabel("-y, mm")
        plt.legend()
        plt.tight_layout()
        axis_equal_3d(ax)

        if save_figures:
            ax.view_init(elev=10, azim=-20)
            plt.savefig(data_path + "/calibration_view1.png", dpi=320)
            ax.view_init(elev=12, azim=26)
            plt.savefig(data_path + "/calibration_view2.png", dpi=320)

        plt.figure("Errors", (12, 7))
        plt.clf()
        plt.subplot(1, 3, 1, title="Camera reprojection")
        plt.hist(np.concatenate(charuco_errors[1]), bins=50)
        plt.xlabel("Error, pixels")
        plt.tight_layout()

        plt.subplot(1, 3, 2, title="Circle fit")
        plt.hist(c_errors, bins=50)
        plt.xlabel("Error, mm")
        plt.tight_layout()

        plt.subplot(1, 3, 3, title="Axis fit")
        plt.hist(axis_errors, bins=50)
        plt.xlabel("Error, mm")
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/errors.png", dpi=160)

    if save:
        with open("stage.json", "w") as f:
            json.dump({"p": p,
                       "dir": dir,
                       "mean_error, mm": np.mean(axis_errors),
                       "incline_angle, deg": angle}, f, indent=4, cls=NumpyEncoder)

    return p, dir, axis_errors


if __name__ == "__main__":
    camera_calib = load_camera_calibration("D:/Scanner/Calibration/camera_intrinsics/data/charuco/calibration.json")

    # data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_5_deg_before/merged/"
    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_2_deg_after/merged/"
    calibrate_axis(data_path, camera_calib, min_circle_points=70, save=True, plot=True, save_figures=True)

    plt.show()

