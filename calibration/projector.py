import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import cv2
from cv2 import aruco
from hdr import *


# n - horizontal, m - vertical. Detect in down-sampled image and then refine in original
def detect_chessboard(img, resize=4, n=17, m=8, plot=False):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 50, 0.001)

    if resize and resize != 1:
        img2 = cv2.resize(img, (img.shape[1] // resize, img.shape[0] // resize))
    else:
        img2 = img

    ret, corners0 = cv2.findChessboardCorners(img2, (n, m), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(img2, corners0, (n, m), (-1, -1), criteria)
        corners = cv2.cornerSubPix(img, corners2 * resize, (n, m), (-1, -1), criteria)

        if plot:
            img2 = cv2.drawChessboardCorners(np.repeat(img2[:, :, None], 3, axis=2), (n, m), corners / resize, ret)
            p = (np.round(corners[0, 0, :] / resize)).astype(np.int)
            img2 = cv2.rectangle(img2, tuple(p - [10, 10]), tuple(p + [10, 10]), (0, 255, 0), thickness=2)

            cv2.imshow('img', img2)
            while cv2.getWindowProperty('img', 0) >= 0:
                cv2.waitKey(50)

        objPoints = np.zeros((n * m, 3), np.float)
        objPoints[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2)
        return objPoints, corners.reshape((-1, 2))
    else:
        return None


# resize - for display only
def detect_charuco(img, resize=4, plot=False):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(25, 18, 15, 15 * 7 / 9, aruco_dict)

    m_pos, m_ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)

    if len(m_pos) > 0:
        count, c_pos, c_ids = cv2.aruco.interpolateCornersCharuco(m_pos, m_ids, img, board)

        if plot:
            img2 = cv2.resize(img, (img.shape[1] // resize, img.shape[0] // resize))
            img2 = aruco.drawDetectedMarkers(np.repeat(img2[:, :, None], 3, axis=2), np.array(m_pos) / resize, m_ids)
            img2 = aruco.drawDetectedCornersCharuco(img2, np.array(c_pos) / resize, c_ids, cornerColor=(0, 255, 0))

            cv2.imshow('img', img2)
            while cv2.getWindowProperty('img', 0) >= 0:
                cv2.waitKey(50)

        return board.chessboardCorners[c_ids].reshape((-1, 3)), np.array(c_pos).reshape((-1, 2))
    else:
        return None


def chessboard_vs_charuco(filename, plot=True):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

    obj1, charuco = detect_charuco(img, resize=4, plot=plot)
    obj2, chess = detect_chessboard(img, resize=4, n=24, m=17, plot=plot)

    if chess is not None and charuco is not None:
        charuco = np.flip(charuco.reshape((17, 24, 2)), axis=1).reshape((-1, 2))
        err = np.linalg.norm(chess - charuco, axis=1)
        print("Mean error, pix:", np.mean(err))
        if plot:
            plt.figure("Corner Detection Error")
            plt.hist(err, bins=100)
            plt.xlabel("Error, pixels")
            plt.title("Corner detection error (mean = %f)" % np.mean(err))
            plt.show()


def normalize(img, level=1):
    if level > 0:
        img = img / np.max(img)
        img /= level
    else:
        img = img / abs(level)

    img[img > 1] = 1
    img *= 255
    return np.round(img).astype(np.uint8)


def detect(path, folders, counts, calibration, plot=False):
    ret, mtx, dist, rvecs, tvecs = calibration
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (6464, 4852), alpha=1, centerPrincipalPoint=True)
    planes = []

    for folder, count in zip(folders, counts):
        for i in range(count):
            print(folder, i)
            img = load_openexr(path + folder + "/_%d.exr" % i)
            img = cv2.undistort(img, mtx, dist, None, new_mtx)

            charu_level = -1
            if folder == "full":
                charu_level = -0.75

            chess_level = 1
            if folder == "left":
                if i == 0:
                    chess_level = 0.5
                if i == 2:
                    chess_level = 1.5
            if folder == "full":
                if i == 0:
                    chess_level = 0.35

            objCharuco, ImgCharuco = detect_charuco(normalize(img, charu_level), resize=5, plot=plot)
            objChess, ImgChess = detect_chessboard(normalize(img, chess_level), resize=4, n=17 if folder == "full" else 9, m=8, plot=plot)

            planes.append({"folder": folder, "i": i,
                           "objCharuco": objCharuco.tolist(), "ImgCharuco": ImgCharuco.tolist(),
                           "objChess": objChess.tolist(), "ImgChess": ImgChess.tolist()})

    return planes, new_mtx

if __name__ == "__main__":
    # chessboard_vs_charuco("camera/original.png", plot=True)
    path = "projector/"

    # with open("camera/refined_calibration.pkl", "rb") as f:
    #     calibration = pickle.load(f)
    #
    # planes, new_mtx = detect(path, folders=["right", "left", "full"], counts=[8, 3, 4], calibration=calibration, plot=False)
    #
    # with open(path + "planes.json", "w") as f:
    #     json.dump({"planes": planes, "new_mtx": new_mtx.tolist()}, f, indent=4)

    with open(path + "planes.json", "r") as f:
        data = json.load(f)
        planes, new_mtx = data["planes"], data["new_mtx"]
        for i, p in enumerate(planes):
            for key, value in p.items():
                if key != "folder" and key != "i":
                    planes[i][key] = np.array(value)
        new_mtx = np.array(new_mtx)

    ax = plt.subplot(111, projection='3d', proj_type='ortho')

    for p in planes:
        ret, rvec, tvec = cv2.solvePnP(p["objCharuco"], p["ImgCharuco"], new_mtx, None)
        R, _ = cv2.Rodrigues(rvec)

        charuco3 = np.matmul(R, p["objCharuco"].T) + tvec
        ax.scatter(charuco3[0, :], charuco3[2, :], -charuco3[1, :], label=p["folder"] + " " + str(p["i"]))
        ax.scatter(charuco3[0, 0], charuco3[2, 0], -charuco3[1, 0] - 0.1, c='k')

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")
    plt.legend()

    plt.show()
    exit()
