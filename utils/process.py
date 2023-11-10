from utils.detect import *
from shutil import copyfile
import matplotlib.pyplot as plt


def linear_map(img, thr=None, mask=None, gamma=1.0):
    if thr is None:
        pixels = img[mask].ravel() if mask is not None else img.ravel()

        if pixels.shape[0] > 1e+6:
            pixels = pixels[::int(pixels.shape[0] / 1e+6)]

        thr = 1.2 * np.sort(pixels)[int(0.99*pixels.shape[0])]  # threshold at 99th percentile

    img = img / thr
    if abs(gamma - 1.0) > 1e-6:
        img = np.power(img, 1/gamma)

    return np.minimum(255 * img, 255).astype(np.uint8), thr


def gamma_map(img, thr=None, mask=None, gamma=2.2):
    return linear_map(img, thr=thr, mask=mask, gamma=gamma)


def tone_map(img, thr=None, mask=None):
    if thr is None:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) > 2 else img
        pixels = gray[mask].ravel() if mask is not None else gray.ravel()
        thr = np.average(pixels)  # optimize for average brightness

    return (255 * (1 - np.exp(-img / thr))).astype(np.uint8), thr


# Load single HDR image and map it to HDR using a method of choice. HDRs are assumed to be gray scale by default
def map_single(filename, method=None, return_image=True, is_gray=True, suffix="", save=False, plot=False):
    assert(method is not None)

    img = load_openexr(filename, make_gray=is_gray)  # our HDRs always have 3 channels (even for gray scale images)
    print("Loaded", filename)

    ldr, thr = method(img)
    print("Threshold:", thr)

    if save:
        new_filename = filename[:-4] + suffix + ".png"
        save_ldr(new_filename, ldr)
        print("Saved", new_filename)
    else:
        new_filename = None

    if plot:
        plt.figure((new_filename or filename) + " - Image", (12, 9))
        plt.imshow(ldr)
        plt.colorbar()
        plt.tight_layout()

        plt.figure((new_filename or filename) + " - Hist", (12, 9))
        plt.hist(ldr.ravel(), bins=256)
        plt.tight_layout()

    return ldr if return_image else None, new_filename


def map_all(filename_template, method, return_images=True, **kw):
    filenames = glob.glob(filename_template)

    jobs = [joblib.delayed(map_single)
            (filename, method=method, return_image=return_images, **kw) for filename in filenames]

    return joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)


# Process a charuco-checker calibration pair. Extract clean checker image projected onto a charuco board
def process_single(image_filename, blank_filename, texture_filename, auto_map=None, out_dir="processed", return_image=True, are_gray=True, save=False, plot=False):
    image = load_openexr(image_filename, make_gray=are_gray)
    blank = load_openexr(blank_filename, make_gray=are_gray)
    clean = np.maximum(0, image - blank)

    print("Loaded", image_filename)

    if texture_filename is not None:
        texture = load_openexr(texture_filename, make_gray=are_gray)
        texture = np.maximum(0, texture - blank)
        texture_ldr, texture_thr = linear_map(texture)
        mask = texture > texture_thr * 0.002

        processed = clean / (texture + 1e-6)
        processed[~mask] = 0
    else:
        processed = clean

    if save:
        path = os.path.dirname(image_filename) + "/" + out_dir + "/"
        ensure_exists(path)

        new_filename = path + os.path.basename(image_filename)[:-4] + ".exr"
        save_openexr(new_filename, processed)
        print("Saved", new_filename)

        if auto_map is not None:
            processed_ldr, processed_thr = auto_map(processed)
            save_ldr(new_filename[:-4] + ".png", processed_ldr)
            print("Mapped", new_filename)
    else:
        new_filename = None

    if plot:
        name = new_filename or image_filename

        plot_image(processed, name + " - Processed HDR", vmin=0, vmax=1)
        plot_image(linear_map(processed)[0], name + " - Processed LDR")
        if texture_filename is not None:
            plot_image(texture_ldr, name + " - Texture LDR")
            plot_image(mask, name + " - Mask")

    return processed if return_image else None, new_filename


def process_all(image_template, blank_template, texture_template, auto_map=None, out_dir="processed", return_images=True, **kw):
    images = glob.glob(image_template)
    blanks = glob.glob(blank_template)
    textures = glob.glob(texture_template) if texture_template is not None else [None] * len(images)

    jobs = [joblib.delayed(process_single)
            (image, blank, texture, auto_map=auto_map, out_dir=out_dir, return_image=return_images, **kw)
            for image, blank, texture in zip(images, blanks, textures)]

    return joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)


# Analyse one image from the stage calibration data
def test_stage(data_path, id=33, n=9, m=8):
    filename = data_path + "blank_%d.exr" % id

    map_single(filename, method=linear_map, is_gray=True, suffix="_linear_map", save=True, plot=True)
    map_single(filename, method=tone_map, is_gray=True, suffix="_tone_map", save=True, plot=True)

    detect_single(filename[:-4] + "_linear_map.png", detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)
    detect_single(filename[:-4] + "_tone_map.png", detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)

    process_single(data_path + "checker_%d.exr" % id, data_path + "blank_%d.exr" % id, data_path + "white_%d.exr" % id,
                   are_gray=True, save=True, plot=True)
    map_single(data_path + "/processed/checker_%d.exr" % id, method=linear_map,
                   is_gray=True, save=True, plot=True)
    detect_single(data_path + "/processed/checker_%d.png" % id, detect_checker, n=n, m=m, size=100,
                   draw=True, save=True, pre_scale=5, draw_scale=1)


def process_stage(data_path):
    map_all(data_path + "blank_*.exr", method=linear_map, return_images=False, save=True, plot=False)
    detect_all(data_path + "blank_*[!map].png", detect_charuco, out_dir="charuco", draw=True, save=True, pre_scale=2, draw_scale=1)

    process_all(data_path + "checker_*[0-9].exr", data_path + "blank_*.exr", data_path + "white_*.exr", out_dir="checker",
                    auto_map=linear_map, return_images=False, are_gray=True, save=True, plot=False)
    detect_all(data_path + "/checker/*.png", detect_checker, n=9, m=8, size=100, out_dir="detected_half",
                draw=True, save=True, pre_scale=5, draw_scale=1)
    detect_all(data_path + "/checker/*.png", detect_checker, n=17, m=8, size=100, out_dir="detected_full",
                draw=True, save=True, pre_scale=5, draw_scale=1)


def rename_group(data_path):
    files = glob.glob(data_path + "/checker_*.exr")
    print("Renaming %d files:" % len(files), files)

    for file in files:
        id = int(file[file.rfind("_") + 1:-4])
        copyfile(file, data_path + "/checker_%d.exr" % id)


def merge_positions(data_path, out_dir="merged"):
    folders = glob.glob(data_path + "/position_*")
    print("Merging %d folders:" % len(folders), folders)

    out_dir += "/"
    if not os.path.exists(data_path + out_dir):
        os.makedirs(data_path + out_dir, exist_ok=True)

    for folder in folders:
        id = int(folder[folder.rfind("_") + 1:])
        files = glob.glob(folder + "/*.exr")

        for file in files:
            new_file = data_path + out_dir + os.path.basename(file)[:-4] + "_%d.exr" % id
            copyfile(file, new_file)


if __name__ == "__main__":
    data_path = "D:/Scanner/Calibration/projector_extrinsic/data/charuco_checker_5mm/"

    # test_stage(data_path, id=33, n=17, m=8)
    # process_stage(data_path)

    data_path = "D:/Scanner/Calibration/projector_intrinsics/data/charuco_checker_5mm/"
    # rename_group(data_path)

    # test_stage(data_path, id=33, n=17, m=8)
    # process_stage(data_path)

    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_5_deg_before/"
    # merge_positions(data_path)

    # test_stage(data_path + "/merged/", id=70, n=9, m=8)
    # process_stage(data_path + "/merged/")

    # data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_2_deg_after/"
    data_path = "D:/scanner_sim/captures/stage_batch_3/stage_calib_2_deg_before/"
    # merge_positions(data_path)

    # test_stage(data_path + "/merged/", id=70, n=9, m=8)
    # process_stage(data_path + "/merged/")

    # data_path = "D:/scanner_sim/calibration/accuracy_test/projector_calib/"
    # data_path = "D:/scanner_sim/calibration/accuracy_test/charuco_plane/combined/"
    data_path = "/media/yurii/EXTRA/scanner-sim-data/calibration/accuracy_test/charuco_plane/combined/"
    process_stage(data_path)

    plt.show()
