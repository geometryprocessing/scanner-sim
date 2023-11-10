from simulator.rendering.configuration import *
from simulator.rendering import *
from scipy.optimize import curve_fit


calib_path = "../../data/calibrations/"
valid_path = "../../data/validation/projector_focus/"


def simulate_projector_focus(data_path, mitsuba_path, range_cm=(61, 94), scale=0.05, ideal_camera=False, verbose=True, **kw):
    config, cam_geom = {}, load_calibration(calib_path + "camera_geometry.json")
    # Render small crop (120 pixels high) only for optimal performance
    w, h = 120, 120
    cam_geom["image_width, pixels"] = w
    cam_geom["image_height, pixels"] = h
    cam_geom["new_mtx"][:2, 2] = (w-1)/2, (h-1)/2

    configure_camera_geometry(config, cam_geom, **kw)
    configure_camera_focus(config, calib_path + "camera_focus.json", **kw)

    configure_projector_geometry(config, calib_path + "projector_geometry.json", **kw)
    configure_projector_focus(config, calib_path + "projector_focus.json", **kw)
    # config["pro_diff_limit"] = 0

    if ideal_camera:
        config["cam_aperture_radius"] = 0
        config["cam_diff_limit"] = 0
        config["cam_pixel_aspect"] = 1.0

    if verbose:
        print("Config:", config)
        print("Range:", range_cm)

    ensure_exists(data_path + "/")

    h, w = 1080, 1920
    pattern = np.zeros((h, w, 3), dtype=np.uint8)
    # pattern[:, w//2 - 10, :] = 255
    # pattern[:, w//2 - 8, :] = 255
    # pattern[h-10, :, :] = 255
    # pattern[h-8, :, :] = 255
    pattern[h-1, w//2, :] = 255
    imageio.imwrite(data_path + "/pattern.png", pattern)
    config["pro_offset_x"], config["pro_offset_y"] = w/2, h

    config["scale"] = scale
    config["offset"] = config["cam_focus_distance"] - config["pro_focus_distance"]

    for dist_cm in range(*range_cm):
        # Geometry cube is 2*scale meters in size
        config["dist"] = config["scale"] + dist_cm / 100.
        write_scene_file(config, data_path + "/dist_%d.xml" % dist_cm, valid_path + "projector_focus.xml")

    source(mitsuba_path + "/setpath.sh")
    render_scenes(data_path + "/dist_*.xml", verbose=verbose)


# dist_offset_mm = offset parameter from projector_focus.xml and
# cam_res_rad = atan(1 / new_mtx[0,0]) from camera_geometry.json
def analyze_projector_focus(data_path, reference=None, dist_offset_mm=320, cam_res_rad=6.8e-5, print_version=True):
    if reference is not None:
        ref = load_calibration(reference)
        ref_data = ref["dof (dist, res), mm"]
    else:
        ref, ref_data = None, None

    files = sorted(glob.glob(data_path + "/dist_*.exr"))

    def polar_sigmoid(rc, *p):
        # print(p)
        if len(p) != 6:  # a bug in curve_fit - passes numpy array on last iteration instead of a tuple
            p = p[0].tolist()
        cx, cy, R, sigma, scale, offset = p
        x, y = rc[1, :] - cx, rc[0, :] - cy
        r = np.sqrt(x**2 + y**2)
        return scale / (1 + np.exp(r - R) / sigma) + offset

    dist_mm, res_pix, sigmas_pix, fits = [], [], [], []

    plt.figure("Original Crops", (16, 16))
    m = int(np.ceil(np.sqrt(len(files))))

    for i, file in enumerate(files):
        d = int(file[file.rfind("_")+1:][:-4])
        print("Dist: %d cm" % d)

        img = load_openexr(file)
        h, w = img.shape

        r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rc = np.vstack([r.ravel(), c.ravel()])
        z = img.ravel()

        p0 = [w/2, h/2, 1, 1, np.max(z), np.min(z)]
        par, cov = curve_fit(polar_sigmoid, rc, z, p0, bounds=([0, 0, 0, 0, 0, 0],
                                                               [w, h, h, h, 10*np.max(z), np.max(z)]))
        print('Fitted:', par)
        dist_mm.append(d * 10 - dist_offset_mm)
        res_pix.append(2 * par[2])
        sigmas_pix.append(par[3])
        fits.append(polar_sigmoid(rc, *par).reshape((h, w)))

        plt.subplot(m, m, i + 1, title=str(dist_mm[i]) + " mm")
        plt.imshow(img)
        plt.plot(par[0], par[1], ".r")

    dist_mm, res_pix, sigmas_pix = np.array(dist_mm), np.array(res_pix), np.array(sigmas_pix)
    cam_res_mm = (dist_mm + dist_offset_mm) * np.tan(cam_res_rad)
    res_mm, sigmas_mm = res_pix * cam_res_mm, sigmas_pix * cam_res_mm
    plt.tight_layout()

    plt.figure("Fitted Crops", (16, 16))
    for i, file in enumerate(files):
        plt.subplot(m, m, i + 1, title=str(dist_mm[i]) + " mm")
        plt.imshow(fits[i])
    plt.tight_layout()

    plt.figure("Projector Resolution", (6, 4.5) if print_version else (12, 9))
    if reference is not None:
        plt.plot(ref_data[0, :] / 10, ref_data[1, :], ".b", label="Measured")
    if not print_version:
        plt.plot(dist_mm / 10, sigmas_mm, ".g", label="Fitted Sigma")
    plt.plot(dist_mm / 10, res_mm, "-r", label="Simulated")
    plt.xlim([28, 61.5])
    if reference is not None:
        # plt.ylim([0, ref["aperture, mm"] / 2])
        plt.ylim([0, 2.75])
    if not print_version:
        plt.title("Projector resolution (pixel diameter)")
    plt.xlabel("Distance, cm")
    plt.ylabel("Resolution, mm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(valid_path + "projector_focus.png", dpi=300 if print_version else 200)


if __name__ == "__main__":
    mitsuba_path = "/media/vice-oqton/Vice_SSD/01. Projects/01. THEIA/Tools/scanner-sim/mitsuba"
    data_path = mitsuba_path + "/scenes"
    ensure_exists(data_path)

    # simulate_projector_focus(data_path + "/projector_focus", mitsuba_path, ideal_camera=False, cam_samples=1024)

    analyze_projector_focus(data_path + "/projector_focus", reference=calib_path + "projector_focus.json")

    # pos = 81
    # simulate_projector_focus(data_path + "/projector_focus", mitsuba_path,
    #                          ideal_camera=True, diff_limit=None, range_cm=(pos, pos+1))
    # rgb, d = load_openexr(data_path + "/projector_focus/dist_%d.exr" % pos, make_gray=False, load_depth=True)
    #
    # # plt.figure("Depth", (12, 10))
    # # plt.imshow(d)
    # # plt.colorbar()
    # # plt.tight_layout()
    #
    # plt.figure("Img", (12, 10))
    # plt.imshow(rgb[:, :, 0])
    # plt.colorbar()
    # plt.tight_layout()

    plt.show()
