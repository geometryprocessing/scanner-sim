import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import least_squares


def reprojection_errors(obj_points, img_points, calibration):
    ret, mtx, dist, rvecs, tvecs = calibration

    errors = []
    for i, (obj_p, img_p) in enumerate(zip(obj_points, img_points)):
        img_points_2, _ = cv2.projectPoints(obj_p, rvecs[i], tvecs[i], mtx, dist)
        img_points_2 = img_points_2.reshape((obj_p.shape[0], 2))
        errors.append(np.average(np.linalg.norm(img_p - img_points_2, axis=1)))

    return np.array(errors)


def save_camera_calibration(calib, filename):
    with open(filename, "w") as f:
        json.dump({"mtx": calib[0].tolist(),
                   "dist": (calib[1]).tolist(),
                   "new_mtx": calib[2].tolist(),
                   "roi": calib[3].tolist()}, f, indent=4)


def load_camera_calibration(filename):
    with open(filename, "r") as f:
        calib = json.load(f)

    return np.array(calib["mtx"]), np.array(calib["dist"]), np.array(calib["new_mtx"]), np.array(calib["roi"])


def calibrate_intrinsic(data_path, max_images=70, min_points=80, max_error=1.0, no_tangent=False, plot=False, undistort=False):
    corners = json.load(open(data_path + "corners.json", "r"))

    names = [name for name, points in corners.items()]
    img_points = [np.array(points["img_points"]).reshape(-1, 2).astype(np.float32) for name, points in corners.items()]
    obj_points = [np.array(points["obj_points"]).reshape(-1, 3).astype(np.float32) for name, points in corners.items()]
    print("Detected:", len(img_points))

    img, name, points = cv2.imread(data_path + names[0]), names[0], img_points[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    idx = [i for i in range(len(img_points)) if img_points[i].shape[0] > min_points]
    print("Has enough points (>%d):" % min_points, len(idx))

    names, img_points, obj_points = [names[i] for i in idx], [img_points[i] for i in idx], [obj_points[i] for i in idx]

    if len(img_points) > max_images:
        stride = len(img_points) // max_images + 1
        names, img_points, obj_points = names[::stride], img_points[::stride], obj_points[::stride]

    print("\nUsing %d:" % len(names), names)

    # flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST
    flags = cv2.CALIB_FIX_TANGENT_DIST if no_tangent else 0

    full_calibration = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None, flags=flags)

    errors = reprojection_errors(obj_points, img_points, full_calibration)
    print("Mean full error:", np.mean(errors))

    if plot:
        plt.close("all")
        plt.figure("Calibration", (12, 9))
        plt.plot(np.arange(errors.shape[0]), errors, ".r", label="Full")

    idx = np.nonzero(errors < max_error)[0]
    names, img_points, obj_points = [names[i] for i in idx], [img_points[i] for i in idx], [obj_points[i] for i in idx]
    print("\nSelected %d:" % len(names), names)

    refined_calibration = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None, flags=flags)

    all_errors, errors = errors, reprojection_errors(obj_points, img_points, refined_calibration)
    print("Mean refined error:", np.mean(errors))

    ret, mtx, dist, rvecs, tvecs = refined_calibration
    print("\nmtx:\n", mtx)
    print("\ndist:", dist)

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print("\nnew_mtx:\n", new_mtx)
    print("\nroi:", roi)

    if plot:
        plt.plot(idx, errors, ".b", label="Refined")
        plt.plot([0, all_errors.shape[0]], [max_error, max_error], '--k', label="Threshold")
        plt.title("Reprojection Errors")
        plt.xlabel("Image #")
        plt.ylabel("Error, pixels")
        plt.ylim([0, 1.1 * np.max(all_errors)])
        plt.legend()
        plt.tight_layout()
        plt.savefig(data_path + "reprojection_errors.png", dpi=160)

    if undistort:
        undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)
        u_points = cv2.undistortPoints(points, mtx, dist, None, new_mtx).reshape(-1, 2)

        if plot:
            def plot_rec(p):
                plt.plot(p[:, 0], p[:, 1], ".g")
                tl, tr = np.argmin(p[:, 0] + p[:, 1]), np.argmin(-p[:, 0] + p[:, 1])
                bl, br = np.argmax(-p[:, 0] + p[:, 1]), np.argmax(p[:, 0] + p[:, 1])
                plt.plot(p[[tl, tr, br, bl, tl], 0], p[[tl, tr, br, bl, tl], 1], "-r")
                return [tl, tr, br, bl, tl]

            def draw_rec(img, p, idx):
                for i in range(len(idx)-1):
                    img = cv2.line(img, tuple(p[idx[i], :]), tuple(p[idx[i+1], :]), (0, 0, 255), thickness=1)
                return img

            def draw_points(img, p):
                for i in range(p.shape[0]):
                    img = cv2.circle(img, tuple(p[i, :]), 10, (0, 255, 0), thickness=2)
                return img

            plt.figure("Original", (12, 9))
            plt.imshow(img)
            plt.plot(mtx[0, 2], mtx[1, 2], ".b")

            idx = plot_rec(points)
            plt.tight_layout()
            # plt.savefig(data_path + "original.png", dpi=300)
            img = draw_rec(img, points, idx)
            img = draw_points(img, points)
            img = cv2.circle(img, tuple(mtx[:2, 2].astype(np.int)), 5, (255, 0, 0), thickness=10)


            plt.figure("Undistorted", (12, 9))
            plt.imshow(undistorted)
            plt.plot(new_mtx[0, 2], new_mtx[1, 2], ".b")

            idx = plot_rec(u_points)
            plt.tight_layout()
            # plt.savefig(data_path + "undistorted.png", dpi=300)
            undistorted = draw_rec(undistorted, u_points, idx)
            undistorted = draw_points(undistorted, u_points)
            undistorted = cv2.circle(undistorted, tuple(new_mtx[:2, 2].astype(np.int)), 5, (255, 0, 0), thickness=10)

        cv2.imwrite(data_path + name[:-4] + '_original.png', img)
        cv2.imwrite(data_path + name[:-4] + '_undistorted.png', undistorted)

    calibration = mtx, dist.ravel(), new_mtx, np.array(roi)
    save_camera_calibration(calibration, data_path + "calibration.json")

    return calibration


def replace_hot_pixels(img, dark, thr=32):
    h, w = img.shape[:2]
    rr, cc = np.nonzero(dark > thr)

    for r, c in zip(rr, cc):
        v, n = 0, 0
        if c > 0:
            v += img[r, c - 1]
            n += 1
        if c < w - 1:
            v += img[r, c + 1]
            n += 1
        if r > 0:
            v += img[r - 1, c]
            n += 1
        if r < h - 1:
            v += img[r + 1, c]
            n += 1
        img[r, c] = v / n

    print("Replaced %d hot/stuck pixels with average value of their neighbours" % rr.shape[0])

    return img


def calibrate_vignetting(data_path, light_on_filename, light_off_filename, dark_frame_filename, center, plot=False):
    on = cv2.imread(data_path + light_on_filename)[..., 0]
    off = cv2.imread(data_path + light_off_filename)[..., 0]
    dark = cv2.imread(data_path + dark_frame_filename)[..., 0]
    path_prefix = data_path + "/processed/" + light_on_filename[:-4] + "_"

    def stats(img, low=16, high=250):
        vmin, vmax = np.min(img), np.max(img)
        print("\tMin - Max range:", [vmin, vmax])
        print("\tDark (<%d) / Saturated (>%d): %d / %d" % (low, high, np.nonzero(img < low)[0].shape[0],
                                                                      np.nonzero(img > high)[0].shape[0]))
        return vmin, vmax

    clean = np.maximum(0, on - off)
    print("Original")
    vmin, vmax = stats(clean)

    clean = replace_hot_pixels(clean, dark)
    vmin, vmax = stats(clean)

    if plot:
        plt.close("all")
        plt.figure("Original Vignetting", (16, 9))
        plt.imshow(clean, vmin=vmin, vmax=vmax)
        plt.title("Original Vignetting")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path_prefix + "original.png", dpi=160)

    sigma = 5
    clean = gaussian_filter(clean, sigma=sigma)
    print("Applied Gaussian filter with sigma =", sigma)
    vmin, vmax = stats(clean)

    # Compute geometric correction
    h, w = clean.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    scale, light_origin = 11.5/6464, np.array([14.5*12, 0, 14.5*12])  # inch/pixel, inch
    x, y, z = (x - w/2)*scale, (h/2 - y)*scale, np.zeros_like(x)
    r = np.linalg.norm(np.stack([x, y, z], axis=2) - light_origin[None, None, :], axis=2)
    correction = np.power(r/np.average(r), 2)

    clean = clean * correction
    print("Applied Correction")
    vmin, vmax = stats(clean)

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
        json.dump({"function": "intensity = p[0] + p[1]*r^2 + p[2]*r^3 + p[3]*r^4, where r is distance fron the center in pixels",
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
    data_path = "D:/Scanner/Calibration/camera_intrinsics/data/"

    calibrate_intrinsic(data_path + "charuco/", max_error=0.9, no_tangent=True, plot=True, undistort=True)
    # calibrate_intrinsic(data_path + "checker/", max_error=0.8, no_tangent=True, plot=True, undistort=True)

    intrinsic = load_camera_calibration(data_path + "charuco/calibration.json")
    center = intrinsic[0][:2, 2]

    data_path = "D:/Scanner/Calibration/camera_vignetting/data/"
    # Run one at a time to prevent fitting failure on a second call (a bug)
    calibrate_vignetting(data_path, "inverted_softbox_8s.png", "dark_room_8s.png", "dark_frame_8s.png", center, plot=True)
    # calibrate_vignetting(data_path, "neatfi_light_10s.png", "dark_room_10s.png", "dark_frame_10s.png", center, plot=True)

    plt.show()
