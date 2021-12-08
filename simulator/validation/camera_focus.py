from configuration import *
from rendering import *
from scipy.optimize import curve_fit


def simulate_camera_focus(data_path, mitsuba_path, range_cm=(65, 96), verbose=True, **kw):
    config, cam_geom = {}, load_calibration("../../scanner/calibration/camera/camera_geometry.json")
    # Render small crop (200 pixels high) only for optimal performance
    cam_geom["image_width, pixels"] = 200
    cam_geom["image_height, pixels"] = 100

    configure_camera_geometry(config, cam_geom)
    configure_camera_focus(config, "../../scanner/calibration/camera/camera_focus.json", **kw)

    if verbose:
        print("Config:", config)
        print("Range:", range_cm)

    header, body = load_template("camera_focus.xml")
    ensure_exists(data_path + "/")

    for dist_cm in range(*range_cm):
        # Geometry cube is 1 meter in size
        config["dist"] = 0.5 + dist_cm / 100.
        generate_scene(header, body, config, data_path + "/dist_%d.xml" % dist_cm)

    source(mitsuba_path + "/setpath.sh")
    render_scenes(data_path + "/dist_*.xml", verbose=verbose)


def analyze_camera_focus(data_path, reference=None):
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

    plt.figure("Camera Resolution", (12, 9))
    if ref is not None:
        plt.plot(ref[0, :], ref[1, :], ".b", label="Measured")
    plt.plot(dist, res, "-r", label="Simulated")
    plt.xlim([650, 950])
    plt.xlabel("Distance, mm")
    plt.ylabel("Resolution, pixels")
    plt.title("Camera resolution (%.1f %% contrast)" % contrast)
    plt.legend()
    plt.tight_layout()
    plt.savefig("camera_focus.png", dpi=200)


if __name__ == "__main__":
    mitsuba_path = "/home/yurii/software/mitsuba"
    data_path = mitsuba_path + "/scenes"
    ensure_exists(data_path)

    simulate_camera_focus(data_path + "/camera_focus", mitsuba_path)

    analyze_camera_focus(data_path + "/camera_focus",
                         reference="../../scanner/calibration/camera/camera_focus.json")
    plt.show()
