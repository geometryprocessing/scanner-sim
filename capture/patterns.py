import numpy as np
import imageio
import png
import cv2

eps = 1e-1

def gen_blank(dim):
    return np.zeros((*dim, 3), dtype=np.uint8)

def gen_circle(dim, pos, r, R, lum, shrink=1):
    img = np.zeros((*dim, 3), dtype=np.uint8)
    i, j = np.meshgrid(np.arange(0, dim[0]), np.arange(0, dim[1]), indexing='ij')
    index = ((i.ravel() - int(pos[0])) ** 2 + ((j.ravel() - int(pos[1]))/shrink) ** 2 < R ** 2 + R*eps) & \
            ((i.ravel() - int(pos[0])) ** 2 + ((j.ravel() - int(pos[1]))/shrink) ** 2 > r ** 2 - eps)
    i, j = np.nonzero(np.reshape(index, dim))
    img[i, j, :] = lum
    return img

def gen_checker(dim, pos, size, n, m, lum):
    img = np.ones((*dim, 3), dtype=np.uint8) * int(lum)
    for i in range(n):
        for j in range(m):
            if (i + j) % 2 == 0:
                t = pos[0] + i * size
                l = pos[1] + j * size
                img[t:t+size, l:l+size, :] = 0;

    return img

def gen_line(dim, start, stop, width, lum):
    img = np.zeros((*dim, 3), dtype=np.uint8)
    cv2.line(img, (int(start[0]), int(start[1])), (int(stop[0]), int(stop[1])), (255, 255, 255), max(1, int(width)))
    img[img != 0] = lum
    return img

def gen_stripes(dim, stride, dir, lum):
    img = np.zeros((*dim, 3), dtype=np.uint8)
    for i in range(0, dim[dir], 2 * stride):
        if dir == 0:
            img[i:i + stride, :, :] = lum
        if dir == 1:
            img[:, i:i + stride, :] = lum
    return img

def save_pattern(pattern, filename):
    # png.from_array(pattern, 'RGB').save(filename)
    imageio.imsave(filename, pattern)

def load_pattern(filename):
    return imageio.imread(filename)
