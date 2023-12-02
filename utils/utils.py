import itertools
import os
import cv2
import glob
import json
import re
import Imath
import shutil
import OpenEXR
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
# import meshio
import open3d as o3d
# import imageio

import matplotlib
matplotlib.use('TkAgg')
# font = {'family': 'serif', 'weight': 'normal', 'size': 32}
font = {'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)


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
    
def transform2string(transform):
    # TODO: add tests for validity of transform
    transform = np.array2string(transform.flatten(), prefix="", suffix="", max_line_width=1000)
    transform = transform.replace("[", "").replace("]", "")
    transform = re.sub(' +', ' ', transform)
    return transform

def string2transform(t_string):
    transform = np.fromstring(t_string, sep=' ')
    transform = transform.reshape(4, 4)
    return transform


def numpinize(data):
    return {k: (np.array(v) if (type(v) is list or type(v) is tuple) else
               (numpinize(v) if type(v) is dict else v)) for k, v in data.items()}


def ensure_exists(path):
    path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def img_stats(img, low=16, high=250):
    vmin, vmax = np.min(img), np.max(img)
    print("\tMin - Max range:", [vmin, vmax])
    print("\tDark (<%d) / Saturated (>%d): %d / %d" % (low, high, np.nonzero(img < low)[0].shape[0],
                                                                  np.nonzero(img > high)[0].shape[0]))
    return vmin, vmax


def plot_image(img, figure_name, title=None, size=(16, 9), save_as=None, **kw):
    plt.figure(figure_name, size)
    plt.clf()
    plt.imshow(img, **kw)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as + ".png", dpi=160)


def plot_hist(data, figure_name, title=None, size=(12, 12), save_as=None, **kw):
    plt.figure(figure_name, size)
    plt.clf()
    plt.hist(data, **kw)
    if title:
        plt.title(title)
    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as + ".png", dpi=160)


def plot_3d(points, figure_name, title=None, size=(12, 9), axis_equal=True, save_as=None, **kw):
    plt.figure(figure_name, size)
    plt.clf()
    ax = plt.subplot(111, projection='3d', proj_type='ortho')
    ax.set_title(title if title else figure_name)

    scatter(ax, points, s=5, **kw)

    ax.set_xlabel("x, mm")
    ax.set_ylabel("z, mm")
    ax.set_zlabel("-y, mm")
    plt.tight_layout()
    if axis_equal:
        axis_equal_3d(ax)

    if save_as is not None:
        plt.savefig(save_as + ".png", dpi=160)

    return ax


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


# # Gray scale by default
# def load_openexr(filename, make_gray=True):
#     with open(filename, "rb") as f:
#         in_file = OpenEXR.InputFile(f)
#         try:
#             dw = in_file.header()['dataWindow']
#             dim = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
#             (r, g, b) = in_file.channels("RGB", pixel_type=Imath.PixelType(Imath.PixelType.FLOAT))
#
#             r = np.reshape(np.frombuffer(r, dtype=np.float32), dim)
#             g = np.reshape(np.frombuffer(g, dtype=np.float32), dim)
#             b = np.reshape(np.frombuffer(b, dtype=np.float32), dim)
#             rgb = np.stack([r, g, b], axis=2)
#
#             if make_gray:
#                 return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#             else:
#                 return rgb
#         finally:
#             in_file.close()

# Gray scale by default
def load_openexr(filename, make_gray=True, load_depth=False):
    with open(filename, "rb") as f:
        in_file = OpenEXR.InputFile(f)
        try:
            dw = in_file.header()['dataWindow']
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            dim = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
            # print(dim)
            if len(in_file.header()['channels']) == 3:  # Scan
                (r, g, b) = in_file.channels("RGB", pixel_type=pt)
                d = None
            elif len(in_file.header()['channels']) >= 4:  # Sim
                r = in_file.channel('color.R', pt)
                g = in_file.channel('color.G', pt)
                b = in_file.channel('color.B', pt)

                if load_depth:
                    d = in_file.channel("distance.Y", pt)
                    d = np.reshape(np.frombuffer(d, dtype=np.float32), dim)
                else:
                    d = None

            r = np.reshape(np.frombuffer(r, dtype=np.float32), dim)
            g = np.reshape(np.frombuffer(g, dtype=np.float32), dim)
            b = np.reshape(np.frombuffer(b, dtype=np.float32), dim)
            rgb = np.stack([r, g, b], axis=2)

            ret = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if make_gray else rgb
            if load_depth:
                return ret, d
            else:
                return ret
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


def load_calibration(filename):
    with open(filename, "r") as f:
        return numpinize(json.load(f))


def copy_to(path, files=[]):
    if type(files) is str:
        files = glob.glob(files)
    assert type(files) is list

    ensure_exists(path)
    for file in files:
        shutil.copy2(file, path)


def save_ply(filename, points, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))

    #print(colors.shape, normals.shape, colors.dtype, normals.dtype)
    if type(normals) != type(None):
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float32))
    if type(colors) != type(None):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    o3d.io.write_point_cloud(filename, pcd, compressed=False, print_progress=True)


def load_ply(filename):
    pc = o3d.io.read_point_cloud(filename, print_progress=True)
    
    return np.asarray(pc.points), np.asarray(pc.normals), np.asarray(pc.colors)


def scatter(ax, p, *args, **kwargs):
    if len(p.shape) > 1:
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1], *args, **kwargs)
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


def axis_equal_3d(ax, zoom=1):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r/zoom, ctr + r/zoom)


def fit_plane(p, ax=None, **kwargs):
    pca = PCA(n_components=3)
    p2 = pca.fit_transform(p)
    print(pca.mean_, pca.singular_values_, "\n", pca.components_)

    if ax:
        scatter(ax, p[::10, :], s=5, label="p", **kwargs)
        basis(ax, pca.mean_, pca.components_.T, length=20, **kwargs)

    return p, p2, pca.mean_, pca.singular_values_, pca.components_


def fit_circle(points, p0, ax=None, **kwargs):
    def circle_loss(p, xy):
        # print(p)
        cx, cy, R = p
        x, y = xy[:, 0] - cx, xy[:, 1] - cy
        r = np.sqrt(x**2 + y**2)
        return r - R

    p = least_squares(circle_loss, p0, args=(points,))['x']
    # p = least_squares(circle_loss, p0, bounds=([0, -1, 0], [255, 1, 1]), args=(points,))['x']
    print("Fitted parameters:\n\t", p)
    return p

    # if ax:
    #     scatter(ax, p[::10, :], s=5, label="p", **kwargs)
    #     basis(ax, pca.mean_, pca.components_.T, length=20, **kwargs)
    #
    # return p, p2, pca.mean_, pca.singular_values_, pca.components_


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
