import json
import Imath
import OpenEXR
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.ndimage.morphology as morph
from scipy.ndimage.filters import gaussian_filter


def load_images(path):
    with open(path + "exposures.json", "r") as f:
        exposures = json.load(f)
        print("Found", len(exposures), "images in", path)

    images = [np.load(path + str(i) + ".npy") / (2 ** 12 - 1) for i in range(len(exposures))]

    return np.array(exposures)[:, 1], np.array(images)


def find_gamma(path, plot=False):
    exposures, images = load_images(path)

    gammas = []
    for i in np.arange(images.shape[1]):
        for j in np.arange(images.shape[2]):
            x = np.log10(exposures)
            y = np.log10(images[:, i, j])

            idx = (y > np.log10(0.01)) & (y < np.log10(0.8))

            fit_func = lambda p, x: p[0] + p[1] * x
            err_func = lambda p, x, y: (y - fit_func(p, x))

            p, _ = optimize.leastsq(err_func, [-1.0, 1.0], args=(x[idx], y[idx]))
            gammas.append(p[1])

    if plot:
        plt.figure("Gamma")
        plt.hist(gammas, bins=1000)
        plt.title("Gamma = %f" % np.mean(gammas))

    return np.mean(gammas)


def correct_images(images, gamma=1.00792):
    # gamma correction
    images = np.power(images, 1 / gamma)

    # broken pixel removal
    h, w = 4852, 6464
    broken_r, broken_c = 3123, 2862
    if images.shape[1] == h and images.shape[2] == w:
        images[:, broken_r, broken_c] = images[:, broken_r, broken_c + 1]

    return images


def compute_hdr(exposures, images, plot=False):
    order = np.argsort(exposures)[::-1]

    exp = exposures[order[0]]
    res = np.copy(images[order[0], :, :])
    thr0 = 0.85
    thr = thr0

    for i in range(exposures.shape[0] - 1):
        # mask = res > thr
        mask_g = gaussian_filter(res, 1) > thr

        struct = scipy.ndimage.generate_binary_structure(2, 1)
        mask_e = morph.binary_erosion(mask_g, struct, 1)
        mask_dd = morph.binary_dilation(mask_e, struct, 2)

        r, c = np.nonzero(mask_dd)
        print("%d: replace %d" % (i, r.shape[0]))

        if r.shape[0] == 0:
            break

        exp2 = exposures[order[i + 1]]
        img2 = images[order[i + 1], :, :]

        # exposures are known with sub us precision
        ratio = exp / exp2

        res[r, c] = img2[r, c] * ratio
        thr = thr0 * ratio

        if plot:
            plt.figure(str(i), (14, 8))
            plt.subplot2grid((1, 3), (0, 1), colspan=2)
            masked = res.copy()
            masked[r, c] = 0.5 * masked[r, c]
            plt.imshow(masked)
            plt.gca().set_title("%.6f seconds exposure (ratio %.6f)" % (exp2, ratio))
            plt.colorbar()

            r0, c0 = np.nonzero(~mask_dd)
            ratios = res[r0, c0] / img2[r0, c0]
            std, mean = np.std(ratios), np.mean(ratios)

            y, bins = np.histogram(ratios, bins=1000, range=[ratio - 3 * std, ratio + 3 * std])
            x = 0.5 * (bins[:-1] + bins[1:])

            def gauss_function(x, a, mu, sigma):
                return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

            p, _ = optimize.curve_fit(gauss_function, x, y, p0=[np.max(y), ratio, std])

            plt.subplot2grid((1, 3), (0, 0))
            plt.bar(x, y, width=x[1] - x[0])
            plt.plot(x, gauss_function(x, *p), 'r')
            plt.gca().set_title("%.6f -> %.6f" % (mean, p[1]))
            plt.tight_layout()

    return res


def save_openexr(file, image):
    G = image.astype(np.float16).tostring()

    header = OpenEXR.Header(image.shape[1], image.shape[0])
    header['Compression'] = Imath.Compression(Imath.Compression.PXR24_COMPRESSION)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(OpenEXR.HALF)),
                          'G': Imath.Channel(Imath.PixelType(OpenEXR.HALF)),
                          'B': Imath.Channel(Imath.PixelType(OpenEXR.HALF))}

    exr = OpenEXR.OutputFile(file, header)
    exr.writePixels({'R': G, 'G': G, 'B': G})
    exr.close()


if __name__ == "__main__":
    path = "hdr/"

    gamma = find_gamma("gamma/", plot=True)
    print("gamma =", gamma)

    exposures, images = load_images(path)
    images = correct_images(images, gamma)

    hdr = compute_hdr(exposures, images, plot=True)

    np.save(path + "hdr", hdr.astype(np.float32))
    save_openexr(path + "hdr.exr", hdr)

    plt.figure('HDR', (12, 8))
    plt.imshow(hdr, vmax=np.max(hdr)/10.)
    plt.colorbar()
    plt.show()
    print("Done")
