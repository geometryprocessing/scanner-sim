import itertools
import os
import cv2
import json
import Imath
import OpenEXR
import numpy as np
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def numpinize(data):
    return {k: (np.array(v) if (type(v) is list or type(v) is tuple) else
               (numpinize(v) if type(v) is dict else v)) for k, v in data.items()}


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def img_stats(img, low=16, high=250):
    vmin, vmax = np.min(img), np.max(img)
    print("\tMin - Max range:", [vmin, vmax])
    print("\tDark (<%d) / Saturated (>%d): %d / %d" % (low, high, np.nonzero(img < low)[0].shape[0],
                                                                  np.nonzero(img > high)[0].shape[0]))
    return vmin, vmax


def replace_hot_pixels(img, dark, thr=32):
    h, w = img.shape[:2]
    rr, cc = np.nonzero(dark > thr)

    for r, c in zip(rr, cc):
        v, n = 0, 0
        if c > 0:
            v += img[r, c - 1]
            n += 1
        if c < w - 1:
            v += img[r, c + 1]
            n += 1
        if r > 0:
            v += img[r - 1, c]
            n += 1
        if r < h - 1:
            v += img[r + 1, c]
            n += 1
        img[r, c] = v / n

    print("Replaced %d hot/stuck pixels with average value of their neighbours" % rr.shape[0])

    return img


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


if __name__ == "__main__":
    data = {"l": [1, 2, 3],
            "t": (4, 5, 6),
            "i": 7,
            "f": 8.0,
            "a": np.array([9, 10, 11]),
            "d": {"a2": np.array([12, 13, 14])}}

    print(data)

    with open("test.json", "w") as f:
        json.dump(data, f, cls=NumpyEncoder)

    with open("test.json", "r") as f:
        data = numpinize(json.load(f))

    print(data)
