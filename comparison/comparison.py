import cv2
import matplotlib
import matplotlib.pyplot as plt

from hdr import *
from process import *

# font = {'family' : 'serif',
#         'weight' : 'normal',
#         'size'   : 22}
#
# matplotlib.rc('font', **font)


# Gray scale by default
def load_openexr_2(filename, make_gray=True, load_depth=False):
    with open(filename, "rb") as f:
        in_file = OpenEXR.InputFile(f)
        try:
            dw = in_file.header()['dataWindow']
            dim = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
            # print(dim)
            if len(in_file.header()['channels']) == 3:  # Scan
                (r, g, b) = in_file.channels("RGB", pixel_type=Imath.PixelType(Imath.PixelType.FLOAT))
                d = None
            elif len(in_file.header()['channels']) >= 4:  # Sim
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
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

            if make_gray:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), d
            else:
                return rgb, d
        finally:
            in_file.close()

if __name__ == "__main__":
    # l, r, b, t, save_to = 2000, 4300, 3700, 900, "full_3/"
    l, r, b, t, save_to = 2750, 3250, 3025, 2725, "crop_3/"

    render, _ = load_openexr_2("img_000_2.exr")
    print("render", render.shape, render.dtype)
    render = render[:, 2:6466]
    render = np.roll(render, -4, axis=0)
    render = np.roll(render, 4, axis=1)
    save_openexr(save_to + "render.exr", render)
    # render = render[:, 1:3233]
    render_lm, render_thr = linear_map(render)
    cv2.imwrite(save_to + "1_render.png", render_lm[t:b, l:r])

    scan = load_openexr("checker.exr")
    # scan = scan[::2, ::2]
    print("scan", scan.shape, scan.dtype)
    scan_lm, scan_thr = linear_map(scan)
    cv2.imwrite(save_to + "scan.png", scan_lm)

    background = load_openexr("img_01.exr")
    # background = background[::2, ::2]
    print("background", background.shape, background.dtype)

    # mask = cv2.cvtColor(cv2.imread("pawn_mask.png"), cv2.COLOR_BGR2GRAY)
    # mask = np.ones_like(scan, dtype=np.bool)
    # mask = mask > 0
    mask = render > 0
    print("mask", mask.shape, mask.dtype)
    cv2.imwrite(save_to + "mask.png", mask.astype(np.uint8)*255)

    vignetting = cv2.cvtColor(cv2.imread("inverted_softbox_smooth.png"), cv2.COLOR_BGR2GRAY)
    vignetting = vignetting / np.max(vignetting)

    clean = scan - background
    clean[~mask] = 0
    clean /= vignetting
    save_openexr(save_to + "clean.exr", clean)

    clean_lm, clean_thr = linear_map(clean)
    cv2.imwrite(save_to + "0_clean.png", clean_lm[t:b, l:r])

    clean = clean / clean_thr / 1.2
    render = render / render_thr / 1.2

    # plt.figure("Scan", (16, 12))
    # plt.imshow(scan)
    # plt.figure("Background", (16, 12))
    # plt.imshow(background)
    plt.figure("Mask", (16, 12))
    plt.imshow(mask)

    plt.figure("Vignetting", (16, 12))
    plt.imshow(vignetting)

    lim = 1.5
    plt.figure("Clean", (16, 9))
    plt.imshow(clean, vmin=0, vmax=lim)
    plt.xlim([l, r])
    plt.ylim([b, t])
    plt.title("Clean")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_to + "clean_plot.png", dpi=300)

    plt.figure("Render", (16, 9))
    plt.imshow(render, vmin=0, vmax=lim)
    plt.xlim([l, r])
    plt.ylim([b, t])
    plt.title("Render")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_to + "render_plot.png", dpi=300)

    plt.figure("Diff", (16, 9))
    plt.imshow(render-clean, cmap="bwr", vmin=-lim, vmax=lim)
    plt.xlim([l, r])
    plt.ylim([b, t])
    plt.title("Render - Clean")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_to + "diff.png", dpi=300)

    diff_thr = 0.05
    plt.figure("Hist All", (16, 9))
    diff = (render-clean).ravel()
    diff = diff[np.abs(diff) > diff_thr]
    plt.hist(diff, bins=1000, range=[-0.5, 0.5])
    plt.title("Diff (std=%.3f)" % np.std(diff))
    plt.semilogy()
    plt.tight_layout()
    plt.savefig(save_to + "diff_hist_all.png", dpi=200)

    plt.figure("Hist Crop", (16, 9))
    diff = (render-clean)[t:b, l:r].ravel()
    # diff = diff[np.abs(diff) > diff_thr]
    plt.hist(diff, bins=1000, range=[-0.2, 0.4])
    plt.title("Diff (std=%.3f)" % np.std(diff))
    plt.semilogy()
    plt.tight_layout()
    plt.savefig(save_to + "diff_hist_crop.png", dpi=200)

    plt.figure("Hist Patch", (16, 9))
    row, col, sz = 2800, 3000, 50
    diff = (render-clean)[row:row+sz, col:col+sz].ravel()
    # diff = diff[np.abs(diff) > diff_thr]
    plt.hist(diff, bins=500, range=[-0.1, 0.1])
    plt.title("Diff Patch (std=%.3f, mean=%.3f)" % (np.std(diff), np.mean(diff)))
    # plt.semilogy()
    plt.tight_layout()
    plt.savefig(save_to + "diff_hist_patch.png", dpi=200)

    plt.figure("Hist Noise", (16, 9))
    row, col, sz = 2825, 3200, 20
    noise = clean[row:row+sz, col:col+sz].ravel()
    level = np.mean(noise)
    noise -= level
    plt.hist(noise, bins=500, range=[-0.003, 0.003])
    plt.title("Noise (std=%.6f, level=%.6f)" % (np.std(noise), level))
    # plt.semilogy()
    plt.tight_layout()
    plt.savefig(save_to + "hist_noise.png", dpi=200)

    plt.figure("Diff Mask Crop", (16, 9))
    plt.imshow(np.abs(render-clean) > diff_thr)
    plt.title("Diff > %.2f" % (diff_thr))
    plt.xlim([l, r])
    plt.ylim([b, t])
    plt.tight_layout()
    plt.savefig(save_to + "diff_mask_crop.png", dpi=200)

    plt.show()
