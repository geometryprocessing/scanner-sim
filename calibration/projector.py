import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import cv2
from cv2 import aruco
from hdr import *
from camera import *
from display import *


# n - horizontal, m - vertical. Detect in down-sampled image and then refine in original
def detect_chessboard(img, resize=4, n=17, m=8, plot=False, save=None, roi=None):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 50, 0.001)

    if resize and resize != 1:
        img2 = cv2.resize(img, (img.shape[1] // resize, img.shape[0] // resize))
    else:
        img2 = img

    ret, corners0 = cv2.findChessboardCorners(img2, (n, m), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(img2, corners0, (n, m), (-1, -1), criteria)
        corners = cv2.cornerSubPix(img, corners2 * resize, (n, m), (-1, -1), criteria)

        if plot or save:
            img2 = cv2.drawChessboardCorners(np.repeat(img2[:, :, None], 3, axis=2), (n, m), corners / resize, ret)
            p = (np.round(corners[0, 0, :] / resize)).astype(np.int)
            img2 = cv2.rectangle(img2, tuple(p - [10, 10]), tuple(p + [10, 10]), (0, 255, 0), thickness=2)

            if save:
                roi = roi // resize if roi is not None else None
                cv2.imwrite(save, img2[roi[1]:roi[3], roi[0]:roi[2], :] if roi is not None else img2)

            if plot:
                cv2.imshow('img', img2)
                while cv2.getWindowProperty('img', 0) >= 0:
                    cv2.waitKey(50)

        objPoints = np.zeros((n * m, 3), np.float)
        objPoints[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2)
        return objPoints, corners.reshape((-1, 2))
    else:
        return None


# resize - for display only
def detect_charuco(img, resize=4, plot=False, save=None, roi=None):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(25, 18, 15, 15 * 7 / 9, aruco_dict)

    m_pos, m_ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)

    if len(m_pos) > 0:
        count, c_pos, c_ids = cv2.aruco.interpolateCornersCharuco(m_pos, m_ids, img, board)

        if plot or save:
            img2 = cv2.resize(img, (img.shape[1] // resize, img.shape[0] // resize))
            img2 = aruco.drawDetectedMarkers(np.repeat(img2[:, :, None], 3, axis=2), np.array(m_pos) / resize, m_ids)
            img2 = aruco.drawDetectedCornersCharuco(img2, np.array(c_pos) / resize, c_ids, cornerColor=(0, 255, 0))

            if save:
                roi = roi // resize if roi is not None else None
                cv2.imwrite(save, img2[roi[1]:roi[3], roi[0]:roi[2], :] if roi is not None else img2)

            if plot:
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
        print("Mean corner error:", np.mean(err))
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


def detect(path, folders, counts, calibration, plot=False, save=True, crop=True):
    ret, mtx, dist, rvecs, tvecs = calibration
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (6464, 4852), alpha=1, centerPrincipalPoint=True)
    planes, roi = [], np.array(roi)
    if crop:
        print(roi)

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

            filename =  path + folder + "/_%d_charuco.png" % i
            objCharuco, ImgCharuco = detect_charuco(normalize(img, charu_level), resize=4,
                                                    plot=plot, save=filename if save else None, roi=roi if crop else None)

            filename =  path + folder + "/_%d_chess.png" % i
            objChess, ImgChess = detect_chessboard(normalize(img, chess_level), resize=4,
                                                   n=17 if folder == "full" else 9, m=8,
                                                   plot=plot, save=filename if save else None, roi=roi if crop else None)

            planes.append({"folder": folder, "i": i,
                           "objCharuco": objCharuco.tolist(), "ImgCharuco": ImgCharuco.tolist(),
                           "objChess": objChess.tolist(), "ImgChess": ImgChess.tolist()})

    return planes, new_mtx


def scatter(ax, p, *args, **kwargs):
    if len(p.shape) > 1:
        ax.scatter(p[0, :], p[2, :], -p[1, :], *args, **kwargs)
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


def projector(ax, pos, rvec, mtx, p):
    T, R = pos.ravel(), cv2.Rodrigues(rvec)
    p = p.reshape((-1, 3))

    basis(ax, T, R)
    scatter(ax, p, c="m")


def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def trace_ray(T, R, p, d):
    A = np.stack((R[:, 0], R[:, 1], -d), axis=1)
    b = p - T
    uvt = np.matmul(np.linalg.inv(A), b)
    return p + uvt[2]*d


def lift_to_3d(p_img, mtx, T, R, offset=0.):
    p_world = np.zeros((p_img.shape[0], 3))
    for i in range(p_img.shape[0]):
        p_world[i, :] = trace_ray(T + offset * R[:, 2], R, np.zeros((3)), np.array([(p_img[i, 0] - mtx[0, 2]) / mtx[0, 0],
                                                                                    (p_img[i, 1] - mtx[1, 2]) / mtx[1, 1], 1]))
    return p_world


def fit_line(points):
    center = np.mean(points, axis=0)
    uu, dd, vv = np.linalg.svd(points - center)
    return center, vv[0]


def intersect_lines(P0, P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    """
    # generate all line direction vectors
    n = (P1-P0)/np.linalg.norm(P1-P0, axis=1)[:, np.newaxis]  # normalized

    # generate the array of all projectors
    projs = np.eye(n.shape[1]) - n[:, :, np.newaxis]*n[:, np.newaxis]  # I - n*n.T

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)

    # solve the least squares problem for the
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R, q, rcond=None)[0]

    return p.ravel()


def point_line_dist(p, l0, l1):
    return np.linalg.norm(np.cross(l1 - l0, p - l0)) / np.linalg.norm(l1 - l0)


def load_planes(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        planes, new_mtx = data["planes"], data["new_mtx"]
        for i, p in enumerate(planes):
            for key, value in p.items():
                if key != "folder" and key != "i":
                    planes[i][key] = np.array(value)

    return planes, np.array(new_mtx)


def calibrate_projector(planes, cam_new_mtx, plot=True, savefigs=True):
    charuco, chess = [], []
    chess_objs, chess_prjs = [], []
    chess_full = np.zeros((len(planes), 8, 17, 3))
    lines = np.zeros((2, 8, 17, 3))

    chess_board = np.stack(np.meshgrid(np.arange(8), np.arange(17), indexing="ij"), axis=2) * 100
    chess_board[:, :, 0] += 190
    chess_board[:, :, 1] += 160
    chess_board = chess_board[:, :, ::-1]

    if plot:
        plt.figure("Projector Calibration", (12, 12))
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        ax.set_title("Projector Calibration")

    cam_reproj_errors = []
    for i, p in enumerate(planes):
        ret, rvec, tvec = cv2.solvePnP(p["objCharuco"], p["ImgCharuco"], cam_new_mtx, None)
        T, (R, _) = tvec.ravel(), cv2.Rodrigues(rvec)

        reproj, _ = cv2.projectPoints(p["objCharuco"], rvec, tvec, cam_new_mtx, None)
        cam_reproj_errors.extend(np.linalg.norm(p["ImgCharuco"] - reproj.reshape(-1, 2), axis=1).tolist())

        charuco_3d = np.matmul(R, p["objCharuco"].T) + tvec
        charuco.append(charuco_3d.T)

        offset = 5.25
        chess_3d = lift_to_3d(p["ImgChess"], cam_new_mtx, T, R, offset=offset)
        chess.append(chess_3d)

        chess_obj = np.ones((chess_3d.shape[0], 3))
        chess_obj[:, 0] = np.dot(chess_3d - T, R[:, 0])
        chess_obj[:, 1] = np.dot(chess_3d - T, R[:, 1])
        chess_obj[:, 2] = 0
        chess_objs.append(chess_obj.astype(np.float32))

        if p["folder"] == "right":
            chess_full[i, :, 8:, :] = chess_3d.reshape((8, 9, 3))
            chess_prj = chess_board[:, 8:, :]
        if p["folder"] == "left":
            chess_full[i, :, :9, :] = chess_3d.reshape((8, 9, 3))
            chess_prj = chess_board[:, :9, :]
        if p["folder"] == "full":
            chess_full[i, :, :, :] = chess_3d.reshape((8, 17, 3))
            chess_prj = chess_board
        chess_prjs.append(chess_prj.reshape(-1, 2).astype(np.float32))

        if plot:
            board(ax, T, R, label="Charuco Boards" if i == 0 else "")

    print("Mean camera error:", np.average(cam_reproj_errors))

    line_fit_errors = []
    for i in range(8):
        for j in range(17):
            valid = chess_full[chess_full[:, i, j, 2] > 1, i, j, :].reshape(-1, 3)
            c, dir = fit_line(valid)
            lines[0, i, j, :] = c
            lines[1, i, j, :] = c + dir
            proj_err = np.array([point_line_dist(valid[k, :], c, c + dir) for k in range(valid.shape[0])])
            line_fit_errors.extend(proj_err)
    print("Mean line fit error:", np.mean(line_fit_errors))

    lines_origin = intersect_lines(lines[0, ...].reshape(-1, 3), lines[1, ...].reshape(-1, 3))
    print("Lines Origin:", lines_origin)

    origin_errors = []
    for i in range(8):
        for j in range(17):
            origin_errors.append(point_line_dist(lines_origin, lines[0, i, j, :], lines[1, i, j, :]))
    print("Mean lines origin error:", np.mean(origin_errors))

    calibration_estimate = cv2.calibrateCamera(chess_objs, chess_prjs, (1920, 1080), None, None)
    errors = reprojection_errors(chess_objs, chess_prjs, calibration_estimate)
    ret, mtx_guess, dist, rvecs, tvecs = calibration_estimate
    print("Reprojection errors:", errors, "\nMean error:", np.mean(errors))
    print("Calibration matrix guess:\n", mtx_guess)

    obj_all = np.vstack(chess).astype(np.float32)
    prj_all = np.vstack(chess_prjs).astype(np.float32)
    full_calibration = cv2.calibrateCamera([obj_all], [prj_all], (1920, 1080), mtx_guess, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    error = reprojection_errors([obj_all], [prj_all], full_calibration)[0]
    ret, mtx, dist, rvecs, tvecs = full_calibration
    T, (R, _) = tvecs[0].ravel(), cv2.Rodrigues(rvecs[0])
    origin = np.matmul(R.T, -T)
    print("Calibration matrix:\n", mtx)
    print("Mean projector error:", error)
    print("Projector Origin:", origin)
    print("Distance from lines origin:", np.linalg.norm(origin - lines_origin))

    if plot:
        scatter(ax, np.concatenate(charuco, axis=0).T, c="g", s=5, label="Charuco Corners")
        scatter(ax, np.concatenate(chess, axis=0).T, c="b", s=8, label="Checker Corners")
        scatter(ax, origin, c="k", s=15, label="Projector Origin")
        basis(ax, origin, R.T, length=20)

        for i in range(8):
            for j in range(17):
                if (i + j) % 2 != 0:
                    continue
                d = lines[0, i, j, :] - lines_origin
                d /= np.linalg.norm(d)
                line(ax, lines_origin + 0*d, lines_origin + 400*d, "r", label="Fitted Rays" if i == 0 and j == 0 else "")

        ax.set_xlabel("x, mm")
        ax.set_ylabel("z, mm")
        ax.set_zlabel("-y, mm")
        plt.legend()
        plt.tight_layout()
        axis_equal_3d(ax)

        if savefigs:
            ax.view_init(elev=10, azim=-20)
            plt.savefig("projector/calibration_view1.png", dpi=320)
            ax.view_init(elev=12, azim=26)
            plt.savefig("projector/calibration_view2.png", dpi=320)

        plt.figure("Original Pattern", (12, 7))
        pattern = gen_checker((1080, 1920), (90, 60), 100, (9, 18))
        plt.imshow(pattern)
        plt.plot(chess_board[:, :, 0], chess_board[:, :, 1], ".g")
        plt.title("Original Pattern")

        obj_all_proj, _ = cv2.projectPoints(obj_all, rvecs[0], tvecs[0], mtx, dist)
        obj_all_proj = obj_all_proj.reshape((-1, 2))
        proj_err = np.linalg.norm(prj_all - obj_all_proj, axis=1)
        plt.plot(obj_all_proj[:, 0], obj_all_proj[:, 1], ".r")
        plt.tight_layout()

        plt.figure("Distorted Pattern", (12, 7))
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (1920, 1080), 1, (1920, 1080))
        pattern_dist = cv2.undistort(pattern, mtx, dist, None, new_mtx)
        plt.imshow(pattern_dist)
        plt.title("Distorted Pattern")

        obj_all_proj2, _ = cv2.projectPoints(obj_all, rvecs[0], tvecs[0], new_mtx, None)
        obj_all_proj2 = obj_all_proj2.reshape((-1, 2))
        plt.plot(obj_all_proj2[:, 0], obj_all_proj2[:, 1], ".r")
        plt.tight_layout()

        if savefigs:
            plt.savefig("projector/calibration_pattern.png", dpi=160)

        plt.figure("Errors", (12, 7))
        plt.subplot(2, 2, 1, title="Camera Reprojection")
        plt.hist(cam_reproj_errors, bins=50)
        plt.xlabel("Error, pixels")

        plt.subplot(2, 2, 2, title="Projector Reprojection")
        plt.hist(proj_err, bins=50)
        plt.xlabel("Error, pixels")

        plt.subplot(2, 2, 3, title="Ray Fitting")
        plt.hist(line_fit_errors, bins=50)
        plt.xlabel("Error, mm")

        plt.subplot(2, 2, 4, title="Projector Origin")
        plt.hist(origin_errors, bins=50)
        plt.xlabel("Error, mm")
        plt.tight_layout()

        if savefigs:
            plt.savefig("projector/calibration_errors.png", dpi=160)

    return origin, R.T, mtx, dist


if __name__ == "__main__":
    # chessboard_vs_charuco("camera/original.png", plot=True)
    path = "projector/"

    with open("camera/refined_calibration.pkl", "rb") as f:
        calibration = pickle.load(f)

    # planes, new_mtx = detect(path, folders=["right", "left", "full"], counts=[8, 3, 4],
    #                          calibration=calibration, plot=False, save=True, crop=True)
    #
    # with open(path + "planes.json", "w") as f:
    #     json.dump({"planes": planes, "new_mtx": new_mtx.tolist()}, f, indent=4)

    planes, cam_new_mtx = load_planes(path + "planes.json")

    projector_calibration = calibrate_projector(planes, cam_new_mtx, plot=True, savefigs=True)

    with open("projector/calibration.pkl", "wb") as f:
        pickle.dump(projector_calibration, f)

    plt.show()
    exit()
