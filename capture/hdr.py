import json
import Imath
import OpenEXR
import imageio
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.ndimage.morphology as morph
from scipy.ndimage.filters import gaussian_filter


def load_images(path, numpify=True):
    with open(path + "exposures.json", "r") as f:
        exposures = json.load(f)
        print("Found", len(exposures), "images in", path)

    images = [np.load(path + str(i) + ".npy") for i in range(len(exposures))]
    print("\tLoaded")

    if numpify:
        return np.array(exposures)[:, 1], np.array(images)
    else:
        return exposures, images


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


def explore_dark_frames(path):
    exposures, images = load_images(path, numpify=False)

    titles = []
    for i, exp in enumerate(exposures):
        r, c = np.nonzero(np.greater(images[i], 1.5))
        r2, c2 = np.nonzero(np.greater(images[i], 8.5))
        r3, c3 = np.nonzero(np.greater(images[i], 1024.5))
        titles.append(str(exp[0]) + " sec with %d noisy pixels (%d > 8 and %d > 1024)" % (r.shape[0], r2.shape[0], r3.shape[0]))
        print("\n", titles[-1], "\nr:", r3, "\nc:", c3)

    for i, exp in enumerate(exposures):
        plt.figure(str(exp[0]) + " sec", (16, 9))
        plt.imshow(images[i], vmin=1, vmax=2**([2, 2, 3, 6][i]))
        plt.title(titles[i])
        plt.colorbar()
        plt.get_current_fig_manager().window.state('zoomed')
        plt.tight_layout()
        plt.savefig(path + str(exp[0]) + " sec.png", dpi=150)

    files = [path + str(exp[0]) + " sec.png" for exp in exposures]
    imageio.mimsave(path + "dark_frame_noise.gif", [imageio.imread(f) for f in files], duration=1)

    plt.figure("distributions", (16, 9))
    for i, exp in enumerate(exposures):
        plt.subplot(2, 2, i+1)
        plt.hist(images[i].ravel(), bins=2**10, range=[-0.5, 2**12 - 0.5])
        plt.ylim([0.1, 1e+7])
        plt.title(titles[i])
        plt.yscale("log")

    plt.get_current_fig_manager().window.state('zoomed')
    plt.tight_layout()
    plt.savefig(path + "distributions.png", dpi=150, bbox_inches='tight')


def average_dark_frame(path, save=False):
    exposures, images = load_images(path, numpify=False)
    img = np.average(np.array(images), axis=0)

    if save:
        name = "dark_frame_" + str(exposures[0][0]) + "_sec"
        np.save(path + name, img.astype(np.float32))
        save_openexr(path + name + ".exr", img)

    return img

def plot_dark_frame(img, exp, path=None, scale=12):
    name = "dark_frame_" + str(exp) + "_sec"
    title = str(exp) + " sec dark frame (average of 100)"

    plt.figure(name, (16, 9))
    plt.imshow(img, vmin=1, vmax=2**scale)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    if path:
        plt.savefig(path + name + ".png", dpi=120)

    plt.figure(name + "_distribution", (16, 9))
    plt.hist(img.ravel(), bins=2**12, range=[-0.5, 2**12 - 0.5])
    plt.title(title)
    plt.xlim([-32, 2**12 + 32])
    plt.ylim([0.1, 1e+7])
    plt.yscale("log")
    plt.tight_layout()
    if path:
        plt.savefig(path + name + "_distribution.png", dpi=300, bbox_inches='tight')


def generate_dark_frames(dark_path):
    with open(dark_path + "exposures.json", "r") as f:
        exposures = json.load(f)
        print("Found", len(exposures), "dark frames in", dark_path)

    frames = []
    distributions = []
    for exp in exposures:
        name = "dark_frame_" + str(exp) + "_sec"
        print(name)

        full_path = dark_path + name + "/"
        frames.append(full_path + name + ".png")
        distributions.append(full_path + name + "_distribution.png")

        # img = average_dark_frame(full_path, save=True)
        img = np.load(full_path + name + ".npy")

        if exp < 0.9:
            continue

        s = 2
        if exp > 0.9:
            s = 3
        if exp > 2.9:
            s = 4
        if exp > 5.9:
            s = 5

        plot_dark_frame(img, exp, path=full_path, scale=s)

    imageio.mimsave(dark_path + "dark_frames.gif", [imageio.imread(f) for f in frames], duration=1)
    imageio.mimsave(dark_path + "dark_frame_distributions.gif", [imageio.imread(d) for d in distributions], duration=1)


if __name__ == "__main__":
    # generate_dark_frames("D:/scanner_sim/dark_frames/")
    #
    # plt.show()
    # exit()

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
    plt.tight_layout()
    plt.show()
    print("Done")
