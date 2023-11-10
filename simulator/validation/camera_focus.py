from simulator.rendering.configuration import *
from simulator.rendering import *
from scipy.optimize import curve_fit


calib_path = "../../data/calibrations/"
valid_path = "../../data/validation/camera_focus/"


def simulate_camera_focus(data_path, mitsuba_path, range_cm=(65, 96), verbose=True, **kw):
    config, cam_geom = {}, load_calibration(calib_path + "camera_geometry.json")
    # Render small crop (100 pixels high) only for optimal performance
    w, h = 200, 100
    cam_geom["image_width, pixels"] = w
    cam_geom["image_height, pixels"] = h
    cam_geom["new_mtx"][:2, 2] = (w-1)/2, (h-1)/2

    configure_camera_geometry(config, cam_geom, **kw)
    configure_camera_focus(config, calib_path + "camera_focus.json", **kw)

    if verbose:
        print("Config:", config)
        print("Range:", range_cm)

    ensure_exists(data_path + "/")

    for dist_cm in range(*range_cm):
        # Geometry cube is 1 meter in size
        config["dist"] = 0.5 + dist_cm / 100.
        write_scene_file(config, data_path + "/dist_%d.xml" % dist_cm, valid_path + "camera_focus.xml")

    source(mitsuba_path + "/setpath.sh")
    render_scenes(data_path + "/dist_*.xml", verbose=verbose)


def analyze_camera_focus(data_path, reference=None, print_version=True):
    if reference is not None:
        ref = load_calibration(reference)["dof (dist, res), pixels"]
    else:
        ref = None

    files = sorted(glob.glob(data_path + "/dist_*.exr"))

    def sigmoid(x, a, b, scale, offset):
        return scale / (1 + np.exp(-(x - b) / a)) + offset

    plt.figure("Profiles", (12, 9))

    dist, res, contrast = [], [], 0

    for i, file in enumerate(files):
        d = int(file[file.rfind("_")+1:][:-4])
        print("Dist: %d cm" % d)

        img = load_openexr(file)
        h, w = img.shape
        prof = np.average(img, axis=0)
        prof /= prof[-1]

        x = np.arange(prof.shape[0])
        y = prof

        par, cov = curve_fit(sigmoid, x, y, [1, w/2, np.max(y), np.min(y)])
        print('Fitted:', par)
        dist.append(d * 10)
        res.append(2 * par[0])
        a, b = par[:2]
        contrast = (1.0 - 2 * sigmoid(b - a, *par)) * 100

        plt.plot(x, y, label=str(i))

    dist, res = np.array(dist), np.array(res)

    plt.legend()
    plt.tight_layout()

    plt.figure("Camera Resolution", (6, 4.5) if print_version else (12, 9))
    if ref is not None:
        plt.plot(ref[0, :] / 10, ref[1, :], ".b", label="Measured")
    plt.plot(dist / 10, res, "-r", label="Simulated")
    plt.xlim([64.5, 95.5])
    # plt.ylim([0, 6])
    plt.xlabel("Distance, cm")
    plt.ylabel("Resolution, pixels")
    if not print_version:
        plt.title("Camera resolution (%.1f %% contrast)" % contrast)
    plt.legend()
    plt.tight_layout()
    plt.savefig(valid_path + "camera_focus.png", dpi=300 if print_version else 200)


if __name__ == "__main__":
    mitsuba_path = "/media/vice-oqton/Vice_SSD/01. Projects/01. THEIA/Tools/scanner-sim/mitsuba"
    data_path = mitsuba_path + "/scenes"
    ensure_exists(data_path)

    # simulate_camera_focus(data_path + "/camera_focus", mitsuba_path)

    analyze_camera_focus(data_path + "/camera_focus", reference=calib_path + "camera_focus.json")
    plt.show()
