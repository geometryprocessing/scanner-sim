import glob
import json
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from scipy.ndimage.filters import gaussian_filter

N, M = 11, 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_single_chessboard(filename, resize=False, n=N, m=M, plot=True):
    img = cv2.imread(filename)
    if resize:
        img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (n, m), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (n, m), (-1, -1), criteria)
        print(filename, img.shape, "Succeeded")

        if plot:
            img = cv2.drawChessboardCorners(img, (n, m), corners2, ret)
            cv2.imshow('img', img)
            while cv2.getWindowProperty('img', 0) >= 0:
                cv2.waitKey(50)

        return corners2.reshape((m, n, 2))
    else:
        print(filename, img.shape, "Failed")
        return None


def detect_all_chessboards(filenames, **kwargs):
    jobs = [joblib.delayed(detect_single_chessboard, check_pickle=False)(name, plot=False, **kwargs) for name in filenames]

    results = joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)
    results = {name: result for name, result in zip(filenames, results) if result is not None}

    return results


def reprojection_errors(objpoints, imgpoints, calibration):
    ret, mtx, dist, rvecs, tvecs = calibration

    errors = []
    for i, (objpt, imgpt) in enumerate(zip(objpoints, imgpoints)):
        imgpoints2, _ = cv2.projectPoints(objpt, rvecs[i], tvecs[i], mtx, dist)
        imgpoints2 = imgpoints2.reshape((objpt.shape[0], 2))
        # errors.append(cv2.norm(imgpt, imgpoints2, cv2.NORM_L2) / len(imgpoints2))
        errors.append(np.average(np.linalg.norm(imgpt - imgpoints2, axis=1)))

    return np.array(errors)


def combine(imgs, roi):
    img = np.copy(imgs[0])

    for i in range(len(imgs)-1):
        r = np.average(imgs[i+1][:, roi[0]:roi[1]] / imgs[i][:, roi[0]:roi[1]])
        print(i, r)
        img += imgs[i+1] / r

    return img / (len(imgs) * 255.)


def camera_vignetting(path):
    left = [np.average(imageio.imread(path + "brightness/" + name), axis=2) for name in ["left.png", "left2.png"]]
    right = [np.average(imageio.imread(path + "brightness/" + name), axis=2) for name in ["right.png"]]  # "right2.png" has dark br corner
    center = [np.average(imageio.imread(path + "brightness/" + name), axis=2) for name in ["center.png", "center2.png", "center3.png"]]

    left = combine(left, [0, 4500])
    right = combine(right, [2000, 6500])
    center = combine(center, [1000, 5500])

    l = np.average(left[:, 1000:1300] / center[:, 1000:1300])
    r = np.average(right[:, 5200:5500] / center[:, 5200:5500])
    print(l, r)
    left /= l
    right /= r

    full = np.copy(center)
    full[:, :1000] = left[:, :1000]
    full[:, 5500:] = right[:, 5500:]
    print(full.shape)
    full = gaussian_filter(full, sigma=10)
    full /= np.average(full[2300:2500, 3100:3300])
    np.save(path + "camera_vignetting", full.astype(np.float32))

    plt.figure("full", (12, 7))
    plt.imshow(full)
    plt.colorbar()
    plt.title("Camera Vignetting")
    plt.tight_layout()
    plt.savefig(path + "camera_vignetting.png", dpi=160)


if __name__ == "__main__":
    # camera_vignetting("camera/")
    # plt.show()
    # print("Done")
    # exit()

    # Uncomment this section to detect chessboards and corners. Cached results are used otherwise
    #
    # filenames = glob.glob('D:/calibration/regular/*.png')
    # print(len(filenames), "images")
    #
    # detect_single_chessboard(filenames[0], resize=True)
    #
    # corners = detect_all_chessboards(filenames, resize=False)
    #
    # with open("camera/corners.json", "w") as f:
    #     json.dump({n: c.tolist() for n, c in corners.items()}, f, indent=4)

    with open("camera/corners.json", "r") as f:
        corners = json.load(f)

    print("\n%d detected" % len(corners))

    # 20 mm checker size
    objp = np.zeros((N * M, 3), np.float32)
    objp[:, :2] = np.mgrid[0:N, 0:M].T.reshape(-1, 2)*20

    objpoints = [objp] * len(corners)
    imgpoints = []
    names = []

    for name, results in corners.items():
        imgpoints.append(np.array(results).reshape((N*M, 2)).astype(np.float32))
        names.append(name)

    img = cv2.imread("camera/original.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Uncomment this section to compute full calibration. Cached results are used otherwise
    #
    # full_calibration = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # with open("camera/full_calibration.pkl", "wb") as f:
    #     pickle.dump(full_calibration, f)

    with open("camera/full_calibration.pkl", "rb") as f:
        full_calibration = pickle.load(f)

    errors = reprojection_errors(objpoints, imgpoints, full_calibration)
    print(errors, "\nmean error:", np.mean(errors))

    plt.figure("Calibration", (12, 9))
    plt.plot(np.arange(errors.shape[0]), errors, ".r", label="All")

    # thr = np.mean(errors)
    thr = 0.8

    idx = np.nonzero(np.array(errors) < thr)[0]
    print("\n%d selected" % idx.shape[0])

    objpoints2 = [objpoints[i] for i in idx]
    imgpoints2 = [imgpoints[i] for i in idx]

    # Uncomment this section to compute refined calibration. Cached results are used otherwise
    #
    # refined_calibration = cv2.calibrateCamera(objpoints2, imgpoints2, gray.shape[::-1], None, None)
    # with open("camera/refined_calibration.pkl", "wb") as f:
    #     pickle.dump(refined_calibration, f)

    with open("camera/refined_calibration.pkl", "rb") as f:
        refined_calibration = pickle.load(f)

    errors2 = reprojection_errors(objpoints2, imgpoints2, refined_calibration)
    print(errors2, "\nmean error 2:", np.mean(errors2))

    ret, mtx, dist, rvecs, tvecs = refined_calibration
    print("\nmtx", mtx)
    print("dist", dist)

    plt.plot(idx, errors2, ".b", label="Refined")
    plt.plot([0, errors.shape[0]], [thr, thr], '--k', label="Threshold")
    plt.title("Camera Calibration Reprojection Errors")
    plt.xlabel("Image #")
    plt.ylabel("Error, pixels")
    plt.ylim([0, 1.1*np.max(errors)])
    plt.legend()
    plt.tight_layout()
    plt.savefig("camera/reprojection_errors.png", dpi=160)

    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print("\nnew_mtx", new_mtx)

    dst = cv2.undistort(img, mtx, dist, None, new_mtx)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Crop black edges
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]

    # cv2.imwrite('undistorted.png', dst)

    plt.figure("Image", (12, 9))
    plt.imshow(dst)
    plt.plot(mtx[0,2], mtx[1,2], ".r")
    plt.plot(new_mtx[0,2], new_mtx[1,2], ".b")

    plt.show()
