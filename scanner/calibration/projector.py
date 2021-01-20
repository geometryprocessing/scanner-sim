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
from detect import *


def scatter(ax, p, *args, **kwargs):
    if len(p.shape) > 1:
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1], *args, **kwargs)
    else:
        ax.scatter(p[0], p[2], -p[1], **kwargs)


def line(ax, p1, p2, *args, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], *args, **kwargs)


def basis(ax, T, R, *args, length=1, **kwargs):
    line(ax, T, T + length * R[:, 0], "r")
    line(ax, T, T + length * R[:, 1], "g")
    line(ax, T, T + length * R[:, 2], "b")


def board(ax, T, R, *args, label="", **kwargs):
    line(ax, T, T + 375 * R[:, 0], "orange", linestyle="--", label=label)
    line(ax, T, T + 270 * R[:, 1], "orange", linestyle="--")
    line(ax, T + 375 * R[:, 0], T + 375 * R[:, 0] + 270 * R[:, 1], "orange", linestyle="--")
    line(ax, T + 270 * R[:, 1], T + 375 * R[:, 0] + 270 * R[:, 1], "orange", linestyle="--")

    basis(ax, T, R, length=10)


def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def calibrate_geometry(data_path, camera_calib, max_planes=70, intrinsic=None, no_tangent=False, save=False, plot=False, save_figures=True, **kw):
    charuco, checker, plane_errors = reconstruct_planes(data_path, camera_calib, **kw)
    checker_3d, checker_2d, checker_local = checker
    avg_plane_errors, all_plane_errors = plane_errors
    w, h = 1920, 1080
    print("\nReconstructed:", len(checker_3d))
    print("Mean plane error", np.mean(avg_plane_errors))

    stride = len(checker_3d) // max_planes + 1
    checker_3d, checker_2d, checker_local = checker_3d[::stride], checker_2d[::stride], checker_local[::stride]

    initial_calib, initial_errors = calibrate(checker_local, checker_2d, (h, w), no_tangent=no_tangent,
                                              out_dir=data_path, plot=plot, save_figures=save_figures, **kw)
    mtx_guess, dist_guess, _, _ = initial_calib
    selected = initial_errors[2]

    all_obj = np.vstack([checker_3d[i] for i in selected]).astype(np.float32)
    all_img = np.vstack([checker_2d[i] for i in selected]).astype(np.float32)

    if intrinsic is None:
        flags = (cv2.CALIB_FIX_TANGENT_DIST if no_tangent else 0) | cv2.CALIB_USE_INTRINSIC_GUESS

        mtx_guess = np.array([[3000, 0, 1000], [0, 3000, 1000], [0, 0, 1]]).astype(np.float32)

        full_calib = cv2.calibrateCamera([all_obj], [all_img], (w, h), mtx_guess, None, flags=flags)

        ret, mtx, dist, rvecs, tvecs = full_calib
        print("\nCalibration matrix:\n", mtx)
        print("\nDistortions:", dist)

        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        print("\nOptimal calibration matrix:\n", new_mtx)
        print("\nRegion of interest:", roi)

        intrinsic = mtx, dist.ravel(), new_mtx, np.array(roi)
    else:
        mtx, dist, new_mtx, roi = intrinsic

    ret, rvec, tvec = cv2.solvePnP(all_obj, all_img, mtx, dist)
    full_errors = projection_errors([all_obj], [all_img], (ret, mtx, dist, [rvec], [tvec]))
    full_errors = full_errors[0][0], full_errors[1][0]
    print("\nMean full error:", full_errors[0])

    T, (R, _) = tvec.ravel(), cv2.Rodrigues(rvec)
    origin = np.matmul(R.T, -T)
    print("\nProjector Origin:", origin)
    print("\nProjector Basis [ex, ey, ez]:\n", R)
    extrinsic = origin, R

    if plot:
        plt.figure("Projector Calibration", (12, 12))
        plt.clf()
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Projector Calibration")
        skip = 3

        charuco_3d, charuco_id, charuco_frame = charuco
        to_plot = charuco_frame[::stride]
        for i in range(len(to_plot)):
            if i % skip == 0:
                Ti, Ri = to_plot[i]
                board(ax, Ti, Ri, label="Charuco Boards" if i == 0 else "")

        scatter(ax, np.concatenate(charuco_3d[::stride][::skip], axis=0), c="g", s=5, label="Charuco Corners")
        scatter(ax, np.concatenate(checker_3d[::skip], axis=0), c="b", s=8, label="Checker Corners")
        scatter(ax, origin, c="k", s=15, label="Projector Origin")
        basis(ax, origin, R.T, length=20)

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

        plt.figure("Distortions", (16, 9))
        plt.clf()
        points = np.mgrid[0:17, 0:8].T.reshape(-1, 2) * 100 + np.array([160, 190])
        u_points = cv2.undistortPoints(points.astype(np.float32), mtx, dist, None, new_mtx).reshape(-1, 2)
        plt.plot(points[:, 0], points[:, 1], ".r")
        plt.plot(u_points[:, 0], u_points[:, 1], ".b")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/distortions.png", dpi=160)

        plt.figure("Errors", (12, 7))
        plt.clf()
        plt.subplot(2, 1, 1, title="Camera projection")
        plt.hist(np.concatenate(all_plane_errors), bins=50)
        plt.xlabel("Error, pixels")
        plt.tight_layout()

        plt.subplot(2, 1, 2, title="Projector projection")
        plt.hist(full_errors[1], bins=50)
        plt.xlabel("Error, pixels")
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/calibration_errors.png", dpi=160)

    if save:
        save_projector_calibration(intrinsic, extrinsic, data_path + "/calibration.json", mean_error=full_errors[0])

    return intrinsic, extrinsic, full_errors


def save_projector_calibration(intrinsic, extrinsic, filename, mean_error=0.0):
    with open(filename, "w") as f:
        json.dump({"mtx": intrinsic[0],
                   "dist": intrinsic[1],
                   "new_mtx": intrinsic[2],
                   "roi": intrinsic[3],
                   "origin": extrinsic[0],
                   "basis": extrinsic[1],
                   "mean_projection_error, pixels": mean_error,
                   "projector": "Texas Instrument DPL4710LC",
                   "image_width, pixels": 1920,
                   "image_height, pixels": 1080,
                   "focus_distance, cm": 50,
                   "aperture, mm": 7.5}, f, indent=4, cls=NumpyEncoder)


def load_projector_calibration(filename):
    with open(filename, "r") as f:
        calib = numpinize(json.load(f))

    intrinsic = calib["mtx"], calib["dist"], calib["new_mtx"], calib["roi"]
    extrinsic = calib["origin"], calib["basis"]

    return intrinsic, extrinsic, calib


def calibrate_vignetting(data_path, camera_vignetting, light_on_filename, light_off_filename, dark_frame_filename, checker_filename, plot=False):
    on = cv2.imread(data_path + light_on_filename)[..., 0]
    off = cv2.imread(data_path + light_off_filename)[..., 0]
    dark = cv2.imread(data_path + dark_frame_filename)[..., 0]
    checker = cv2.imread(data_path + checker_filename)[..., 0]
    path_prefix = data_path + "/processed/" + light_on_filename[:-4] + "_"
    ensure_exists(data_path + "/processed/")

    clean = np.maximum(0, on - off)
    print("Original")
    vmin, vmax = img_stats(clean)

    clean = replace_hot_pixels(clean, dark)
    vmin, vmax = img_stats(clean)

    sigma = 5
    clean = gaussian_filter(clean, sigma=sigma)
    print("Applied Gaussian filter with sigma =", sigma)
    vmin, vmax = img_stats(clean)

    corrected = clean * 255.0 / camera_vignetting
    print("Corrected")
    vmin, vmax = img_stats(clean)

    points, new_filename, detected = detect_single(data_path + checker_filename, detect_checker, n=17, m=8, size=100,
                                                   draw=True, save=False, pre_scale=4, draw_scale=1)
    points, W, H = points[0], 1920, 1080
    tl, tr, bl, br = points[0, 0, :], points[0, 16, :], points[7, 0, :], points[7, 16, :]
    print(tl, tr, bl, br)

    px, py = 100, 130
    w, h = 1600 + 2*px, 700 + 2*py

    tl = tl - px*(tr-tl)/1600 - py*(bl-tl)/700
    tr = tr + px*(tr-tl)/1600 - py*(br-tr)/700
    bl = bl - px*(br-bl)/1600 + py*(bl-tl)/700
    br = br + px*(br-bl)/1600 + py*(br-tr)/700

    r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    a, b = r.ravel() / h, c.ravel() / w
    a, b = a[:, None], b[:, None]
    p = ((tl * (1-a) + bl * a) * (1-b) + (tr * (1-a) + br * a) * b).astype(np.int)
    print("\n", p, p.shape)

    r, c = r.ravel()+190-py, c.ravel()+160-px
    rc, z = np.vstack([r, c]), np.zeros((H, W))
    z[r, c] = corrected[p[:, 1], p[:, 0]]

    def surf_func(rc, *p):
        print(p)
        if len(p) != 6:  # a bug in curve_fit - passed numpy array on last iteration instead of a tuple
            p = p[0].tolist()
        cr, cc, p0, px2, pxy, py2 = p
        x, y = rc[1] - cc, rc[0] - cr
        return p0 + px2 * np.power(x, 2) + pxy * x * y + py2 * np.power(y, 2)

    p0 = [H/2, W/2, np.max(z), -1, 0, -1]

    popt, pcov = curve_fit(surf_func, rc, z[r, c].ravel(), p0)
    # print(popt)
    zopt = np.zeros_like(z)
    zopt[r, c] = surf_func(rc, popt)

    R, C = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Z = surf_func(np.vstack([R.ravel(), C.ravel()]), popt).reshape((H, W))

    save_ldr(path_prefix + "vignetting.png", (255 * Z / np.max(Z)).astype(np.uint8))

    if plot:
        plt.close("all")
        # plt.figure("Original Vignetting", (16, 9))
        # plt.imshow(clean, vmin=vmin, vmax=vmax)
        # plt.title("Original Vignetting")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(path_prefix + "original.png", dpi=160)

        plt.figure("Corrected Vignetting", (16, 9))
        plt.imshow(corrected, vmin=vmin, vmax=vmax)
        plt.title("Corrected Vignetting")
        points = points.reshape((-1, 2))
        plt.plot(points[:, 0], points[:, 1], ".r")
        plt.plot(tl[0], tl[1], ".g")
        plt.plot(tr[0], tr[1], ".g")
        plt.plot(bl[0], bl[1], ".g")
        plt.plot(br[0], br[1], ".g")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path_prefix + "corrected.png", dpi=160)

        plt.figure("Cropped", (16, 9))
        plt.imshow(z)
        plt.title("Cropped Vignetting")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path_prefix + "cropped.png", dpi=160)

        plt.figure("Smooth", (16, 9))
        plt.imshow(zopt)
        plt.title("Smooth Vignetting")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path_prefix + "smooth.png", dpi=160)

        plt.figure("Difference", (16, 9))
        plt.imshow(z - zopt)
        plt.title("Difference")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path_prefix + "smooth.png", dpi=160)

        # plt.figure("Checker", (16, 9))
        # plt.imshow(detected)
        # points = points.reshape((-1, 2))
        # plt.plot(points[:, 0], points[:, 1], ".r")
        # plt.plot(tl[0], tl[1], ".g")
        # plt.plot(tr[0], tr[1], ".g")
        # plt.plot(bl[0], bl[1], ".g")
        # plt.plot(br[0], br[1], ".g")
        # plt.title("Checker")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(path_prefix + "detected.png", dpi=160)

    return

    sigma = 5
    clean = gaussian_filter(clean, sigma=sigma)
    print("Applied Gaussian filter with sigma =", sigma)
    vmin, vmax = img_stats(clean)

    # Compute geometric correction
    h, w = clean.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    scale, light_origin = 11.5/6464, np.array([14.5*12, 0, 14.5*12])  # inch/pixel, inch
    x, y, z = (x - w/2)*scale, (h/2 - y)*scale, np.zeros_like(x)
    r = np.linalg.norm(np.stack([x, y, z], axis=2) - light_origin[None, None, :], axis=2)
    correction = np.power(r/np.average(r), 2)

    clean = clean * correction
    print("Applied Correction")
    vmin, vmax = img_stats(clean)

    r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    R = np.linalg.norm(np.stack([r - center[1], c - center[0]], axis=2), axis=2)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)  # center pixel might result in NaN after linalg.norm

    def profile_func(p, r):
        return p[0] + np.power(r, 2) * p[1] + np.power(r, 3) * p[2] + np.power(r, 4) * p[3]

    errfunc = lambda p, r, y: profile_func(p, r) - y

    idx = np.random.randint(w * h, size=100000)
    p0, r, y = [vmax, 0, 0, 0], R.ravel()[idx], clean.ravel()[idx]
    p = least_squares(errfunc, p0, bounds=([0, -1, -1, -1], [255, 1, 1, 1]), args=(r, y))['x']
    print("Fitted parameters:\n\t", p)

    with open(path_prefix + "profile.json", "w") as f:
        json.dump({"function": "intensity = p[0] + p[1]*r^2 + p[2]*r^3 + p[3]*r^4" +
                               ", where r is distance fron the center in pixels",
                   "center": center.tolist(),
                   "p": (p / p[0]).tolist()}, f, indent=4)

    smooth = profile_func(255 * p / p[0], R)
    cv2.imwrite(path_prefix + "smooth.png", np.repeat(smooth[:, :, None], 3, axis=2))

    if plot:
        plt.figure("Geometric Correction", (16, 9))
        plt.imshow(correction)
        plt.title("Geometric Correction")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path_prefix + "correction.png", dpi=160)

        plt.figure("Corrected Vignetting", (16, 9))
        plt.imshow(clean, vmin=vmin, vmax=vmax)
        plt.plot(center[0], center[1], ".r")
        plt.title("Corrected Vignetting")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path_prefix + "corrected.png", dpi=160)

        plt.figure("Radial Profile", (16, 9))
        idx = np.random.randint(0, w*h, size=3000)
        x = R.ravel()[idx]
        plt.plot(x, (clean/correction).ravel()[idx], ".r", markersize=4, label="No Correction")
        plt.plot(x, clean.ravel()[idx], ".b", markersize=4, label="With Correction")
        x = np.sort(x)
        plt.plot(x, profile_func(p, x), "-g", linewidth=2.5, label="Fitted")
        plt.xlim([0, np.max(R)])
        plt.xlabel("Radial distance, pixels")
        plt.ylabel("Intensity")
        plt.legend()
        plt.title("Radial Profile")
        plt.tight_layout()
        plt.savefig(path_prefix + "profile.png", dpi=160)


if __name__ == "__main__":
    camera_calib = load_camera_calibration("D:/Scanner/Calibration/camera_intrinsics/data/charuco/calibration.json")

    data_path = "D:/Scanner/Calibration/projector_intrinsics/data/charuco_checker_5mm/"
    # intrinsic, _, _ = calibrate_geometry(data_path, camera_calib, max_planes=500, no_tangent=True, save=True, plot=True, save_figures=True)

    data_path = "D:/Scanner/Calibration/projector_extrinsic/data/charuco_checker_5mm/"
    # _, extrinsic, errors = calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, max_planes=500, no_tangent=True, save=True, plot=True, save_figures=True)

    # save_projector_calibration(intrinsic, extrinsic, "projector/calibration.json", mean_error=errors[0])
    intrinsic, extrinsic, all = load_projector_calibration("projector/calibration.json")
    # center = all["mtx"][:2, 2]

    data_path = "D:/Scanner/Calibration/projector_vignetting/data/"
    camera_vignetting = load_ldr("camera/vignetting/inverted_softbox_smooth.png", make_gray=True)
    calibrate_vignetting(data_path, camera_vignetting, "White_200ms.png", "Room_200ms.png", "Dark_200ms.png", "White_Checker_200ms.png", plot=True)

    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_5_deg_before/merged/"
    # calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, center=True, no_tangent=True, save=True, plot=True, save_figures=True)

    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_2_deg_after/merged/"
    # calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, max_planes=50, center=True, no_tangent=True, save=True, plot=True, save_figures=True)

    plt.show()
