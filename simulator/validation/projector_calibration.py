from scanner.capture.display import *
from utils import *


if __name__ == "__main__":
    H, W = 1080, 1920
    n, m, size = 17, 8, 100

    checker = gen_checker((H, W), (90, 60), size, (m+1, n+1))
    checker[-1, :, 1:] = 0
    checker[:, -1, 1:] = 0
    checker[0, 0, :2] = 0

    (img, obj, ids), detected = detect_checker(checker[:, :, 0], n=n, m=m, size=size, draw_on=checker.copy())
    img, obj, ids = img.reshape((-1, 2)), obj.reshape((-1, 3)), ids.ravel()

    img_c = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    obj_c = np.array([size * (n-1) / 2.0, size * (m-1) / 2.0, 0])
    print("img_c:", img_c)

    obj_points, img_points, na = [], [], 11

    plt.figure("Input Boards", (12, 12))
    plt.clf()
    ax = plt.subplot(111, projection='3d', proj_type='ortho')
    ax.set_title("Input Boards")

    obj_test = None
    for i in range(na):
        # Translate
        obj_i = obj - obj_c[None, :]
        obj_i[:, 2] = obj_c[0]

        # Rotate
        a = np.pi * i / (8 * (na-1))
        r = obj_i[:, 0].copy()
        obj_i[:, 0] = r * np.cos(a)
        obj_i[:, 2] += r * np.sin(a)

        scatter(ax, obj_i, c="b", s=8)

        # Project
        obj_i[:, :2] = obj_i[:, :2] / obj_i[:, 2][:, None]
        # obj_i[:, 2] = 1

        # scatter(ax, obj_i, c="b", s=8)

        # Distort
        R2 = obj_i[:, 0]**2 + obj_i[:, 1]**2
        obj_i[:, :2] = obj_i[:, :2] * (1 + 0.1 * R2)[:, None]

        scatter(ax, obj_i * obj_c[0], c="r", s=8)

        # sc = img_c[0] - img[0, 0]
        # img_i = sc * obj_i[:, :2] + img_c[None, :]
        # img_points.append(img_i.astype(np.float32))
        # obj_points.append(obj.astype(np.float32))

        # Unproject
        obj_i[:, :2] = obj_i[:, :2] * obj_i[:, 2][:, None]

        if i == 0:
            obj_test = obj_i.copy()
            scatter(ax, obj_test, c="g", s=50)

        img_points.append(img.astype(np.float32))
        obj_points.append(obj_i.astype(np.float32))

    ax.set_xlabel("x, mm")
    ax.set_ylabel("z, mm")
    ax.set_zlabel("-y, mm")
    plt.legend()
    plt.tight_layout()
    axis_equal_3d(ax)

    calib, err = calibrate(obj_points, img_points, (H, W), error_thr=10, centerPrincipalPoint=None, plot=True, save_figures=False)

    plt.figure("Camera", (16, 9))
    plt.imshow(checker)
    plt.plot(img[:, 0], img[:, 1], "xr", markersize=25)
    for i in range(na):
        if i % 10 == 0:
            plt.plot(img_points[i][:, 0], img_points[i][:, 1], ".m:", markersize=9)
    plt.plot(img_c[0], img_c[1], "+g", markersize=25)
    plt.tight_layout()

    plt.figure("Board", (16, 9))
    plt.plot(obj[:, 0], obj[:, 1], ".b", markersize=16)
    plt.plot(obj_c[0], obj_c[1], "+r", markersize=25)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    mtx, dist, new_mtx, roi = calib

    predistorted = cv2.undistort(checker, mtx, dist, None, new_mtx)
    u_img = cv2.undistortPoints(img, mtx, dist, P=new_mtx).reshape(-1, 2)
    u_img_c = cv2.undistortPoints(img_c.reshape((-1, 2)), mtx, dist, None, new_mtx).ravel()

    # u_img2 = cv2.undistortPoints(img, mtx, dist, P=None).reshape(-1, 2)
    # uu_img = cv2.undistortPoints(u_img, new_mtx, None, P=None).reshape(-1, 2)
    # print(uu_img - u_img2)
    # print(uu_img)

    # proj = cv2.projectPoints(obj_points[0], cv2.Rodrigues(np.eye(3))[0], np.array([-obj_c[0], -obj_c[1], obj_c[0]]),
    proj = cv2.projectPoints(obj_test, cv2.Rodrigues(np.eye(3))[0], np.zeros((3)),
                                        mtx, dist)[0].reshape(-1, 2)
                                        # new_mtx, None)[0].reshape(-1, 2)

    plt.figure("Predistorted", (16, 9))
    plt.imshow(predistorted)
    plt.plot(u_img[:, 0], u_img[:, 1], "xr", markersize=25)
    plt.plot(proj[:, 0], proj[:, 1], ".g", markersize=16)
    plt.plot(u_img_c[0], u_img_c[1], "+g", markersize=25)
    plt.tight_layout()

    plt.show()
