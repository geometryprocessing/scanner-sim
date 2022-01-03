from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from utils import *
from calibrate import *
from detect import *


def calibrate_geometry(data_path, camera_calib, max_planes=70, intrinsic=None, no_tangent=False,
                                        centerPrincipalPoint=None, save=False, plot=False, save_figures=True, **kw):
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

        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h),
                                                                centerPrincipalPoint=centerPrincipalPoint)
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

        plt.figure("Errors", (3.3, 3))
        plt.clf()
        # plt.subplot(2, 1, 1, title="Camera projection")
        # plt.hist(np.concatenate(all_plane_errors), bins=50)
        # plt.xlabel("Error, pixels")
        # plt.tight_layout()

        # plt.subplot(2, 1, 2, title="Projector projection")
        plt.hist(full_errors[1], bins=40, range=[0, 0.8])
        plt.xlabel("Error, pixels")
        plt.ylabel("Counts")
        plt.xlim([0, 0.8])
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/all_projector_reprojection_errors.png", dpi=300)

    if save:
        save_projector_calibration(intrinsic, extrinsic, data_path + "/projector_geometry.json", mean_error=full_errors[0])

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
                   # "focus_distance, cm": 50,
                   # "aperture, mm": 7.5,
                   "image_width, pixels": 1920,
                   "image_height, pixels": 1080}, f, indent=4, cls=NumpyEncoder)


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

    # TODO: Save projector vignetting profile as json file similar to camera vignetting (not just evaluated png image)
    # with open(path_prefix + "profile.json", "w") as f:
    #     json.dump({"function": "intensity = p[0] + p[1]*r^2 + p[2]*r^3 + p[3]*r^4" +
    #                            ", where r is distance fron the center in pixels",
    #                "center": center.tolist(),
    #                "p": (p / p[0]).tolist()}, f, indent=4)


    if plot:
        plt.close("all")
        # plt.figure("Original Vignetting", (16, 9))
        # plt.imshow(clean, vmin=vmin, vmax=vmax)
        # plt.title("Original Vignetting")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(path_prefix + "original.png", dpi=160)

        # plt.figure("Corrected Vignetting", (16, 9))
        # plt.imshow(corrected, vmin=vmin, vmax=vmax)
        # plt.title("Corrected Vignetting")
        # points = points.reshape((-1, 2))
        # plt.plot(points[:, 0], points[:, 1], ".r")
        # plt.plot(tl[0], tl[1], ".g")
        # plt.plot(tr[0], tr[1], ".g")
        # plt.plot(bl[0], bl[1], ".g")
        # plt.plot(br[0], br[1], ".g")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(path_prefix + "corrected.png", dpi=160)

        plt.figure("Cropped", (6.7, 3.5))
        plt.imshow(z, vmin=100)
        # plt.title("Cropped Vignetting")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path_prefix + "cropped.png", dpi=300)

        # plt.figure("Smooth", (16, 9))
        # plt.imshow(zopt)
        # plt.title("Smooth Vignetting")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(path_prefix + "smooth.png", dpi=160)

        # plt.figure("Difference", (16, 9))
        # plt.imshow(z - zopt)
        # plt.title("Difference")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(path_prefix + "smooth.png", dpi=160)


def calibrate_white_balance(data_path, R_filename, G_filename, B_filename, exposures=None, plot=False):
    rgb = [None, None, None]
    for i, (name, exp) in enumerate(zip([R_filename, G_filename, B_filename], exposures)):
        img = cv2.imread(data_path + name)[..., 0]
        black = cv2.imread(data_path + "Black" + name[1:])[..., 0]
        dark = cv2.imread(data_path + "Dark" + name[1:])[..., 0]

        clean = np.maximum(0, img - black)
        clean = replace_hot_pixels(clean, dark)
        rgb[i] = clean / exp

    [img_stats(rgb[i], low=16/exposures[i], high=250/exposures[i]) for i in range(3)]

    print("\nApplying gauss filter")
    rgb = [gaussian_filter(rgb[i], sigma=5) for i in range(3)]

    [img_stats(rgb[i], low=16/exposures[i], high=250/exposures[i]) for i in range(3)]

    mask = rgb[1] > 0.5 * np.max(rgb[1])
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    mask = morph.binary_erosion(mask, struct, 5)
    idx = np.nonzero(mask.ravel())[0]

    print("\nApplying mask")
    rgb = [rgb[i].ravel()[idx] for i in range(3)]

    [img_stats(rgb[i], low=16/exposures[i], high=250/exposures[i]) for i in range(3)]

    save_path = data_path + "/processed/"
    ensure_exists(save_path)

    r, g, b = rgb

    if plot:
        plt.figure("Balance Mask", (16, 9))
        plt.imshow(mask)
        plt.title("White Balance Mask")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path + "balance_mask.png", dpi=160)

        plt.figure("White Balance", (12, 7))
        plt.clf()
        plt.subplot(1, 2, 1, title="R")
        plt.hist(g / r, bins=200)
        plt.xlabel("Ratio")
        plt.tight_layout()

        plt.subplot(1, 2, 2, title="B")
        plt.hist(g / b, bins=200)
        plt.xlabel("Ratio")
        plt.tight_layout()
        plt.savefig(save_path + "white_balance.png", dpi=160)

    with open("projector/white_balance.json", "w") as f:
        json.dump({"g/r": np.mean(g / r),
                   "g/b": np.mean(g / b)}, f, indent=4)


def calibrate_response(data_path, cache=True, save=False, plot=False, save_figures=True, **kw):
    colors = ["gray", "red", "green", "blue"]

    def process_single(filename, id, plot=False):
        print("Loading %d:" % id, filename)
        img = load_openexr(filename)

        ambient = np.average(img[100:150, 5000:5050])
        parasitic = np.average(img[1000:1050, 3250:3300])
        signal = np.average(img[2000:2050, 3250:3300])

        print(id, "-", ambient, parasitic, signal)

        if plot:
            m = np.max(img)
            img[100:150, 5000:5050] = m
            img[1000:1050, 3250:3300] = m
            img[2000:2050, 3250:3300] = m

            plt.figure(filename, (16, 10))
            plt.imshow(img, vmax=m/0.9)
            plt.colorbar()
            plt.tight_layout()

        return id, ambient, parasitic, signal

    intensities = {}
    for color in colors:
    # for color in ["gray"]:
        filename = data_path + "/%s.json" % color

        if os.path.exists(filename) and cache:
            print("Reuse cache:", filename)
            with open(filename, "r") as f:
                intensities[color] = numpinize(json.load(f))
        else:
            files = glob.glob(data_path + "/%s_*.exr" % color)
            ids = [int(file[file.rfind("_")+1:-4]) for file in files]

            order = [i[0] for i in sorted(enumerate(ids), key=lambda x:x[1])]
            files = [files[i] for i in order]
            ids = [ids[i] for i in order]

            print(len(files), color, "files:", files)
            print("ids:", ids)

            # process_single(files[-1], ids[-1], plot=True)

            jobs = [joblib.delayed(process_single)(file, id) for file, id in zip(files, ids)]
            results = joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)

            results = {"ids": np.array(ids),
                       "ambient": np.array([r[1] for r in results]),
                       "parasitic": np.array([r[2] for r in results]),
                       "signal": np.array([r[3] for r in results])}

            intensities[color] = results
            # print(results)

            if cache:
                with open(filename, "w") as f:
                    json.dump(results, f, indent=4, cls=NumpyEncoder)

    gammas, polies = {"func": "log10(y) = p[0] * log10(x) + p[1]"}, {"func": "y = p[0] * x^2 + p[1] * x + p[2]"}
    print("\nFitting...")

    for i, color in enumerate(colors):
        x = intensities[color]["ids"][1:]
        y = intensities[color]["signal"][1:] - intensities[color]["signal"][0]
        gam = np.polyfit(np.log10(x), np.log10(y), 1)
        poly = np.polyfit(x, y, 2)
        print(color, gam, poly)
        gammas[color] = gam
        polies[color] = poly

    if save:
        with open(data_path + "/projector_gammas.json", "w") as f:
            json.dump(gammas, f, indent=4, cls=NumpyEncoder)

        with open(data_path + "/projector_polies.json", "w") as f:
            json.dump(polies, f, indent=4, cls=NumpyEncoder)

    if plot:
        plt.figure("Projector Response - Paper Edition", (4, 3.5))
        data = intensities["gray"]
        ids, amb, par, sig = data["ids"], data["ambient"], data["parasitic"], data["signal"]
        resp = sig - sig[0]
        m = np.max(resp)
        plt.plot(ids, resp / m, "b.", label="Measurements")
        plt.plot(ids, np.polyval(polies["gray"], ids) / m, "r-", label="Polynomial Fit", linewidth=1.2)
        plt.legend()
        plt.xlabel("Pixel value")
        plt.ylabel("Projected intensity (relative)")
        plt.tight_layout()
        plt.savefig(data_path + "/projector_response_paper.png", dpi=300)

        plt.figure("Projector Response", (16, 10))
        for i, color in enumerate(colors):
            data = intensities[color]
            ids, amb, par, sig = data["ids"], data["ambient"], data["parasitic"], data["signal"]
            resp = sig - sig[0]

            print("\n%s:" % color.capitalize())
            print("sig_0:", sig[0])
            print("sig_250:", sig[-2])
            print("sig_250 / sig_0:", sig[-2] / sig[0])
            print("par_250 / par_0:", par[-2] / par[0])
            print("par_0 / sig_0:", par[0] / sig[0])
            print("amb_250 / amb_0:", amb[-2] / amb[0])
            print("amb_0 / sig_0:", amb[0] / sig[0])

            # plt.figure(color.capitalize(), (16, 10))
            plt.subplot(221 + i)
            plt.plot(ids, amb, "-", label="Ambient")
            plt.plot(ids, par, "-", label="Parasitic")
            plt.plot(ids, sig, "-", label="Signal")
            plt.plot(ids, resp, "b.", label="Response")
            plt.plot(ids, resp[-2] * np.power(ids / ids[-2], gammas[color][0]), "r-", label="Gamma (g = %.3f)" % gammas[color][0])
            plt.plot(ids, np.polyval(polies[color], ids), "m-", label="Polynomial (deg = 2)")
            plt.legend()
            plt.xlabel("Pixel value")
            plt.ylabel("Intensity")
            # plt.loglog()
            # plt.semilogy()
            plt.title(color.capitalize())
            plt.tight_layout()

            # if save_figures:
            #     plt.savefig(data_path + "%s_response.png" % color, dpi=120)

        plt.savefig(data_path + "/projector_response.png", dpi=120)


if __name__ == "__main__":
    # camera_calib = load_calibration("D:/Scanner/Calibration/camera_intrinsics/data/charuco/calibration.json")
    camera_calib = load_calibration("camera/camera_geometry.json")

    data_path = "D:/Scanner/Calibration/projector_intrinsics/data/charuco_checker_5mm/"
    # intrinsic, _, _ = calibrate_geometry(data_path, camera_calib, max_planes=500, no_tangent=True, save=True, plot=True, save_figures=True)

    data_path = "D:/Scanner/Calibration/projector_extrinsic/data/charuco_checker_5mm/"
    # _, extrinsic, errors = calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, max_planes=500, no_tangent=True, save=True, plot=True, save_figures=True)

    # save_projector_calibration(intrinsic, extrinsic, "projector/projector_geometry.json", mean_error=errors[0])
    proj_calib = load_calibration("projector/projector_geometry_test.json")
    intrinsic = proj_calib["mtx"], proj_calib["dist"], proj_calib["new_mtx"], proj_calib["roi"]
    extrinsic = proj_calib["origin"], proj_calib["basis"]

    # center = proj_calib["mtx"][:2, 2]
    # print("center:", center)

    data_path = "D:/Scanner/Calibration/projector_vignetting/data/"
    # camera_vignetting = load_ldr("camera/vignetting/inverted_softbox_smooth.png", make_gray=True)
    # calibrate_vignetting(data_path, camera_vignetting, "White_200ms.png", "Black_200ms.png", "Dark_200ms.png", "White_Checker_200ms.png", plot=True)

    # calibrate_white_balance(data_path, "R_800ms.png", "G_400ms.png", "B_800ms.png", exposures=[0.8, 0.4, 0.8], plot=True)

    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_5_deg_before/merged/"
    # calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, center=True, no_tangent=True, save=True, plot=True, save_figures=True)

    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_2_deg_after/merged/"
    # calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, max_planes=50, center=True, no_tangent=True, save=True, plot=True, save_figures=True)

    data_path = "D:/scanner_sim/calibration/accuracy_test/projector_calib/"
    # _, extrinsic, errors = calibrate_geometry(data_path, camera_calib, intrinsic=intrinsic, max_planes=500, error_thr=0.8, no_tangent=True, save=True, plot=True, save_figures=True)
    # save_projector_calibration(intrinsic, extrinsic, "projector/projector_geometry_test.json", mean_error=errors[0])

    data_path = "E:/scanner_sim/calibration/projector_response/data/"
    calibrate_response(data_path, save=True, plot=True, save_figures=True)

    plt.show()
