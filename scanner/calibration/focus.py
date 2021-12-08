from scipy.optimize import curve_fit
from utils import *
from calibrate import *
from process import *


def process_checkers(checker_path, planes, projector_calib, plot=False):
    all_corners = [load_corners(checker_path + "/checker_%s/detected/corners.json" % c) for c in ["r", "g", "b"]]
    print("Detected:", [len(c) for c in all_corners])

    corners, avg_errors, all_errors = [], [], []
    mm_scales, projector_dist, camera_dist = [], [], []
    charuco, checker, plane_errors = planes
    origin, basis = projector_calib["origin"], projector_calib["basis"]

    if plot:
        plt.figure("Corners")
        plt.gca().invert_yaxis()

    for i, id in enumerate(sorted([int(name[name.rfind("_") + 1:-4]) for name in all_corners[0].keys()])):
        names = ["checker_%s_%d.png" % (c, i) for c in ["r", "g", "b"]]
        rgb = np.stack([all_corners[c][names[c]]["img"] for c in range(3)])
        avg = np.mean(rgb, axis=0)
        corners.append(avg.reshape(8, 17, 2))
        all_errors.append(np.linalg.norm(rgb - avg[None, :, :], axis=2).ravel())
        avg_errors.append(np.average(all_errors[-1]))

        T, R = charuco[2][i]
        p_int = trace_ray(T, R, origin, basis[2, :])
        projector_dist.append(np.linalg.norm(p_int - origin))

        c_int = trace_ray(T, R, np.array([0, 0, 0]), np.array([0, 0, 1]))
        camera_dist.append(np.linalg.norm(c_int))

        checker_3d = checker[0][i]
        mm_scales.append(np.linalg.norm(checker_3d[0, :] - checker_3d[16, :]) / 1600)

        if plot and i % 5 == 0:
            plt.plot(avg[:, 0], avg[:, 1], ".")

    corners, mm_scales = np.stack(corners), np.array(mm_scales)
    projector_dist, camera_dist = np.array(projector_dist), np.array(camera_dist)

    if plot:
        plt.figure("Average corner error")
        plt.plot(avg_errors)
        plt.figure("Corner errors")
        plt.hist(np.concatenate(all_errors), bins=100)

        plt.figure("Steps")
        plt.hist(np.diff(projector_dist), bins=20)

        plt.figure("Scales")
        plt.plot(0, 0, ".g")
        plt.plot(projector_dist, mm_scales, ".b")

        m, b = np.polyfit(projector_dist, mm_scales, 1)
        x = np.linspace(-1, np.max(projector_dist), 100)
        plt.plot(x, m * x + b, "-r")

    return corners, mm_scales, projector_dist, camera_dist


def crop_single(camera_filename, projector_filename, corners, y_off=300, x_pad=0, y_pad=30, size=150, id=0, plot=False, **kw):
    cam = load_openexr(camera_filename, make_gray=True)
    proj = load_openexr(projector_filename, make_gray=True)
    print("Loaded:", camera_filename, "and", projector_filename)

    y_off = y_off - id * 2  # avoid a spec of dust
    cam_crop = np.average(cam[y_off:y_off+10, cam. shape[1]//2: cam.shape[1]//2 + 400], axis=0)

    proj_crops = np.zeros((8, 17, size, size))
    for i in range(8):
        for j in range(17):
            c = corners[i, j, :].astype(np.int)
            proj_crops[i, j, :, :] = proj[c[1]+y_pad:c[1]+y_pad+size, c[0]+x_pad:c[0]+x_pad+size]

    if plot:
        plt.figure("Camera Profile(s)")
        plt.plot(cam_crop, label=str(id))
        plt.legend()

        plt.figure(projector_filename + " - Crop")
        plt.imshow(proj_crops[0, 0, :, :])
        plt.colorbar()

    return cam_crop, proj_crops


def crop_many(camera_template, projector_template, ids, corners, out_dir=None, save=True, **kw):
    jobs = [joblib.delayed(crop_single, check_pickle=False)
            (camera_template % id, projector_template % id, corners[id, ...], id=id, **kw) for id in ids]

    results = joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)

    cam_crops, proj_crops = np.stack([res[0] for res in results]), np.stack([res[1] for res in results])
    print(cam_crops.shape, proj_crops.shape)

    if save:
        if out_dir is not None:
            np.save(out_dir + "cam_crops.npy", cam_crops.astype(np.float32))
            np.save(out_dir + "proj_crops.npy", proj_crops.astype(np.float32))
        else:
            print("Specify out_dir!")

    return cam_crops, proj_crops


def calibrate_camera(crops, calib_params, skip=10, plot=False, save_figures=None, **kw):
    def sigmoid(x, a, b, scale, offset):
        return scale / (1 + np.exp(-(x - b) / a)) + offset

    x, params = np.arange(crops.shape[1]), []
    for i in range(crops.shape[0]):
        y = crops[i, :]
        par, cov = curve_fit(sigmoid, x, y, [1, 200, np.max(y), np.min(y)])
        print('Fitted:', par)
        params.append(par)

    dist, res = calib_params[3], [2 * p[0] for p in params]

    p = np.polyfit(dist, res, 4)
    lx = np.linspace(650, 950, int(3e+6))
    imin = np.argmin(np.polyval(p, lx))
    focus = lx[imin]
    best_res = np.polyval(p, focus)

    print("Camera focus distance, mm:", focus)
    print("Best camera resolution, pix:", best_res)

    if plot:
        plt.figure("Camera Profiles", (12, 9))
        for i, c in enumerate(range(crops.shape[0])[::skip]):
            plt.plot(crops[c, :], label=str(c))
            plt.plot(x, sigmoid(x, *params[i*skip]), "-r")
        plt.title("Camera Profiles")
        plt.xlabel("Pixels")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()

        if save_figures is not None:
            plt.savefig(save_figures + "camera_profiles.png", dpi=160)

        plt.figure("Camera Resolution", (12, 9))
        plt.plot(dist, res, ".b", label="Measured")
        plt.xlim([650, 950])
        plt.xlabel("Distance, mm")
        plt.ylabel("Resolution, pixels")

        sx = np.linspace(650, 950, 300)
        plt.plot(sx, np.polyval(p, sx), "-r", label="Fitted")
        plt.legend()
        plt.title("Camera Resolution")
        plt.tight_layout()

        if save_figures is not None:
            plt.savefig(save_figures + "camera_resolution.png", dpi=160)

    with open("camera/camera_focus.json", "w") as f:
        json.dump({"aperture, mm": 4.0,
                   "focus, mm": focus,
                   "best_res, pixels": best_res,
                   "dof (dist, res), pixels": (dist, res)}, f, indent=4, cls=NumpyEncoder)

    return focus, best_res, (dist, res)


def calibrate_projector(crops, calib_params, pos, plot=False, save_figures=None, **kw):
    def polar_sigmoid(rc, *p):
        # print(p)
        if len(p) != 6:  # a bug in curve_fit - passed numpy array on last iteration instead of a tuple
            p = p[0].tolist()
        cx, cy, R, sigma, scale, offset = p
        x, y = rc[1, :] - cx, rc[0, :] - cy
        r = np.sqrt(x**2 + y**2)
        return scale / (1 + np.exp(r - R) / sigma) + offset

    params = []
    dim = crops[0, 0, 0, :, :].shape
    r, c = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), indexing="ij")
    rc = np.vstack([r.ravel(), c.ravel()])

    for i in range(crops.shape[0]):
        z = crops[i, pos[0], pos[1], :, :].ravel()
        p0 = [dim[1]/2, dim[0]/2, 1, 1, np.max(z), np.min(z)]
        par, cov = curve_fit(polar_sigmoid, rc, z, p0, bounds=([0, 0, 0, 0, 0, 0],
                                                               [dim[1], dim[0], dim[0], dim[0], 10*np.max(z), np.max(z)]))
        print('Fitted:', par)
        params.append(par)

    N = len(params)
    corners, mm_scales = calib_params[:2]
    pix_scales = np.linalg.norm(corners[:, 0, 16, :] - corners[:, 0, 0, :], axis=1) / 1600
    dist, sigma, R = calib_params[2][:N], np.array([p[3] for p in params]), np.array([p[2] for p in params])
    sigma *= (mm_scales / pix_scales)[:N]
    R *= (mm_scales / pix_scales)[:N]

    thr = 2*N//3
    ap = np.polyfit(dist[thr:], 2*R[thr:], 1)
    aperture = np.max(np.polyval(ap, 0))
    print("Projector aperture, mm:", aperture)

    fp = np.polyfit(dist[:thr], 2*R[:thr], 4)
    x = np.linspace(dist[thr], dist[0], int(3e+6))
    imin = np.argmin(np.polyval(fp, x))
    focus = x[imin]
    best_res = np.polyval(fp, focus)
    print("Projector focus distance, mm:", focus)
    print("Best projector resolution, mm:", best_res)

    if plot:
        n = crops.shape[0]
        m = int(np.sqrt(n))

        # plt.figure("Original Crops (%d, %d)" % (pos[0], pos[1]), (16, 16))
        # for i in range(n):
        #     plt.subplot(m, m, i+1, title=str(i))
        #     plt.imshow(crops[i, pos[0], pos[1]])
        #     p = params[i]
        #     plt.plot(p[0], p[1], ".r")
        # plt.tight_layout()
        # # plt.suptitle("Original Crops (%d, %d)" % (pos[0], pos[1]))
        #
        # if save_figures is not None:
        #     plt.savefig(save_figures + "original_crops.png", dpi=160)
        #
        # plt.figure("Fitted Crops (%d, %d)" % (pos[0], pos[1]), (16, 16))
        # for i in range(n):
        #     plt.subplot(m, m, i+1, title=str(i))
        #     z = polar_sigmoid(rc, *params[i])
        #     plt.imshow(z.reshape(dim))
        # plt.tight_layout()
        # # plt.suptitle("Fitted Crops (%d, %d)" % (pos[0], pos[1]))
        #
        # if save_figures is not None:
        #     plt.savefig(save_figures + "fitted_crops.png", dpi=160)

        plt.figure("Projector Resolution", (5, 3))
        # plt.plot(dist, sigma, ".g", label="Measured Sigma")
        plt.plot(dist, 2*R, ".b", markersize=3.5, label="Measured Points")

        x = np.linspace(0, dist[thr], int(dist[thr]))
        plt.plot(x, np.polyval(ap, x), "-r", linewidth=1.25, label="Aperture Extrapolation")

        x = np.linspace(dist[thr], dist[0], int(dist[0] - dist[thr]))
        plt.plot(x, np.polyval(fp, x), "-m", linewidth=1.25, label="Polynomial Focus Fit")
        plt.xlim([0, 600])
        plt.ylim([0, aperture])
        plt.xlabel("Distance, mm")
        plt.ylabel("Diameter, mm")
        # plt.title("Projector Resolution")
        plt.legend()
        plt.tight_layout()

        if save_figures is not None:
            plt.savefig(save_figures + "projector_resolution.png", dpi=300)

    with open("projector/projector_focus.json", "w") as f:
        json.dump({"aperture, mm": aperture,
                   "focus, mm": focus,
                   "best_res, mm": best_res,
                   "dof (dist, res), mm": (dist, 2*R)}, f, indent=4, cls=NumpyEncoder)

    return aperture, focus, best_res, (dist, 2*R)


if __name__ == "__main__":
    data_path = "D:/Scanner/Calibration/projector_intrinsics/data/"
    checker_path = data_path + "color_checker_5mm/"
    dots_path = data_path + "color_dots_5mm/"

    camera_calib = load_calibration("D:/Scanner/Calibration/camera_intrinsics/data/charuco/calibration.json")
    projector_calib = load_calibration(data_path + "charuco_checker_5mm/calibration.json")

    planes = reconstruct_planes(data_path + "charuco_checker_5mm/", camera_calib)
    print("\nReconstructed:", len(planes[0][0]))

    # for c in ["r", "g", "b"]:
    #     process_all(checker_path + "checker_%s_*.exr" % c, checker_path + "blank_*.exr", None, out_dir=("checker_" + c),
    #                     auto_map=gamma_map, return_images=False, are_gray=True, save=True, plot=False)
    #     detect_all(checker_path + "/checker_%s/*.png" % c, detect_checker, n=17, m=8, size=100, out_dir="detected",
    #                draw=True, save=True, pre_scale=5, draw_scale=1)

    calib_params = process_checkers(checker_path, planes, projector_calib, plot=False)

    # process_all(dots_path + "dots_*.exr", dots_path + "blank_*.exr", None, out_dir="dots",
    #                 auto_map=None, return_images=False, are_gray=True, save=True, plot=False)

    # for id in range(65)[::10]:
    #     crop_single(checker_path + "blank_%d.exr" % id, dots_path + "dots/dots_%d.exr" % id,
    #                 calib_params[0][id, ...], id=id, plot=True)

    # cam_crops, proj_crops = crop_many(checker_path + "blank_%d.exr", dots_path + "dots/dots_%d.exr", range(65)[::1],
    #                                   calib_params[0], out_dir=data_path, save=True, plot=False)

    cam_crops, proj_crops = np.load(data_path + "cam_crops.npy"), np.load(data_path + "proj_crops.npy")

    # Run one at a time to prevent fitting failure on a second call (a bug)
    # cam_focus, cam_res, cam_dof = calibrate_camera(cam_crops, calib_params, plot=True, save_figures=checker_path)
    proj_aperture, proj_focus, proj_res, proj_dof = calibrate_projector(proj_crops[:64, ...], calib_params, (4, 8), plot=True, save_figures=dots_path)

    plt.show()
