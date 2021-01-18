import Imath
import OpenEXR
from detect import *
from shutil import copyfile
import matplotlib.pyplot as plt

# Default color channels order: RGB


# Gray scale by default
def save_openexr(filename, image, keep_rgb=False):
    if len(image.shape) > 2:
        if keep_rgb:
            R = image[:, :, 0].astype(np.float16).tobytes()
            G = image[:, :, 1].astype(np.float16).tobytes()
            B = image[:, :, 2].astype(np.float16).tobytes()
        else:
            R = G = B = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float16).tobytes()
    else:
        R = G = B = image.astype(np.float16).tobytes()

    header = OpenEXR.Header(image.shape[1], image.shape[0])
    header['Compression'] = Imath.Compression(Imath.Compression.PXR24_COMPRESSION)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(OpenEXR.HALF)),
                          'G': Imath.Channel(Imath.PixelType(OpenEXR.HALF)),
                          'B': Imath.Channel(Imath.PixelType(OpenEXR.HALF))}

    exr = OpenEXR.OutputFile(filename, header)
    exr.writePixels({'R': R, 'G': G, 'B': B})   # need to duplicate channels for grayscale anyways
                                                # (to keep it readable by LuminanceHDR)
    exr.close()


# Gray scale by default
def load_openexr(filename, make_gray=True):
    with open(filename, "rb") as f:
        in_file = OpenEXR.InputFile(f)
        try:
            dw = in_file.header()['dataWindow']
            dim = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
            (r, g, b) = in_file.channels("RGB", pixel_type=Imath.PixelType(Imath.PixelType.FLOAT))

            r = np.reshape(np.frombuffer(r, dtype=np.float32), dim)
            g = np.reshape(np.frombuffer(g, dtype=np.float32), dim)
            b = np.reshape(np.frombuffer(b, dtype=np.float32), dim)
            rgb = np.stack([r, g, b], axis=2)

            if make_gray:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                return rgb
        finally:
            in_file.close()


# Unaffected by default
def save_ldr(filename, image, ensure_rgb=False):
    if len(image.shape) == 2 and ensure_rgb:
        image = np.repeat(image[:, :, None], 3, axis=2)

    if len(image.shape) > 2:
        image = image[:, :, ::-1]  # RGB to BGR for cv2 (if color)

    cv2.imwrite(filename, image)  # expects BGR or Gray


# Unaffected by default
def load_ldr(filename, make_gray=False):
    img = cv2.imread(filename)[:, :, ::-1]  # BGR by default

    if make_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


def linear_map(img, thr=None, mask=None):
    if thr is None:
        pixels = img[mask].ravel() if mask is not None else img.ravel()

        if pixels.shape[0] > 1e+6:
            pixels = pixels[::int(pixels.shape[0] / 1e+6)]

        thr = 1.1 * np.sort(pixels)[int(0.99*pixels.shape[0])]  # threshold at 99th percentile

    return np.minimum(255 * img / thr, 255).astype(np.uint8), thr


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

    jobs = [joblib.delayed(map_single, check_pickle=False)
            (filename, method=method, return_image=return_images, **kw) for filename in filenames]

    return joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)


# Process a charuco-checker calibration pair. Extract clean checker image projected onto a charuco board
def process_single(image_filename, blank_filename, texture_filename, auto_map=None, out_dir="processed", return_image=True, are_gray=True, save=False, plot=False):
    image = load_openexr(image_filename, make_gray=are_gray)
    blank = load_openexr(blank_filename, make_gray=are_gray)
    texture = load_openexr(texture_filename, make_gray=are_gray)
    print("Loaded", image_filename)

    texture = np.maximum(0, texture - blank)
    texture_ldr, texture_thr = linear_map(texture)
    mask = texture > texture_thr * 0.002

    clean = np.maximum(0, image - blank)
    processed = clean / (texture + 1e-6)
    processed[~mask] = 0

    if save:
        path = os.path.dirname(image_filename) + "/" + out_dir + "/"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

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

        def plot_image(img, title, **kw):
            plt.figure(title, (12, 9))
            plt.imshow(img, **kw)
            plt.colorbar()
            plt.tight_layout()

        plot_image(processed, name + " - Processed HDR", vmin=0, vmax=1)
        plot_image(linear_map(processed)[0], name + " - Processed LDR")
        plot_image(texture_ldr, name + " - Texture LDR")
        plot_image(mask, name + " - Mask")

    return processed if return_image else None, new_filename


def process_all(image_template, blank_template, texture_template, auto_map=None, out_dir="processed", return_images=True, **kw):
    images = glob.glob(image_template)
    blanks = glob.glob(blank_template)
    textures = glob.glob(texture_template)

    jobs = [joblib.delayed(process_single, check_pickle=False)
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

    process_all(data_path + "checker_*.exr", data_path + "blank_*.exr", data_path + "white_*.exr", out_dir="checker",
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
    process_stage(data_path)

    data_path = "D:/Scanner/Calibration/projector_intrinsics/data/charuco_checker_5mm/"
    # rename_group(data_path)

    # test_stage(data_path, id=33, n=17, m=8)
    process_stage(data_path)

    data_path = "D:/Scanner/Captures/stage_batch_2/stage_calib_5_deg_before/"
    # merge_positions(data_path)

    # test_stage(data_path + "/merged/", id=70, n=9, m=8)
    process_stage(data_path + "/merged/")

    plt.show()
