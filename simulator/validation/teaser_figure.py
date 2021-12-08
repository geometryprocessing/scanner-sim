from process import *

import matplotlib

font = {'family': 'serif', 'weight': 'normal', 'size': 32}
matplotlib.rc('font', **font)


# Gray scale by default
def load_rendered_openexr(filename, make_gray=True, load_depth=False):
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


def preprocess_images(img_path, blank, cam_geom, cam_vignetting, wb=None, **kw):
    # TODO: Refactored prepare_images()
    pass


def prepare_images():
    rendering, _ = load_rendered_openexr("rendering_final.exr")
    print("rendering", rendering.shape, rendering.dtype)

    rendering = rendering[:, :6464]  # Shift by 2 was already accomplished by adjusting projected pattern
    rendering = np.roll(rendering, -4, axis=0)

    save_openexr("rendering_cropped.exr", rendering)

    scan = load_openexr("scan_checker.exr")
    print("scan", scan.shape, scan.dtype)

    parasitic = load_openexr("scan_parasitic.exr")
    print("parasitic", parasitic.shape, parasitic.dtype)

    vignetting = cv2.cvtColor(cv2.imread("camera_vignetting.png"), cv2.COLOR_BGR2GRAY)
    print("vignetting", vignetting.shape, vignetting.dtype)

    clean = np.maximum(0, scan - parasitic)
    clean /= vignetting / np.max(vignetting)
    save_openexr("scan_clean.exr", clean)


def compare(lim=1.03, thr=0.05, crop=False, hist=False, save=False):
    render = load_openexr("rendering_cropped.exr")
    clean = load_openexr("scan_clean.exr")
    print("Loaded")

    mask = render > 1.e-6
    clean[~mask] = 0

    l, r, t, b = 2000, 4250, 900, 3700
    cl, cr, ct, cb = 2750, 3250, 2725, 3050

    nt, nl, ns = 1425, 3200, 50
    render_thr = np.average(render[nt:(nt+ns), nl:(nl+ns)])
    clean_thr = np.average(clean[nt:(nt+ns), nl:(nl+ns)])
    print("Thresholds:", render_thr, clean_thr)

    render_lm = np.minimum(255 * render / (0.8 * render_thr), 255).astype(np.uint8)
    clean_lm = np.minimum(255 * clean / (0.8 * clean_thr), 255).astype(np.uint8)

    render /= render_thr
    clean /= clean_thr

    def plot_img(img, name, lims=None, title=True, bar=True, axis='on', **kw):
        plt.figure(name, (16, 9))
        plt.imshow(img, **kw)

        if title:
            plt.title(name)
        if bar:
            cbar = plt.colorbar()
            cbar.ax.locator_params(nbins=5)
        if lims:
            plt.xlim([lims[0], lims[1]])
            plt.ylim([lims[3], lims[2]])

        plt.axis(axis)
        plt.tight_layout()

    plot_img(render, "Render")
    # plot_img(clean, "Clean")
    # plot_img(mask, "Mask")

    if save:
        ensure_exists("plots/")

        img = np.dstack([255 - clean_lm]*3)
        cv2.rectangle(img, (cl, ct), (cr, cb), (0, 0, 255), thickness=15)
        cv2.imwrite("plots/0_clean.png", img[t:b, l:r])

        img = np.dstack([255 - render_lm]*3)
        cv2.rectangle(img, (nl, nt), (nl+2*ns, nt+2*ns), (0, 255, 0), thickness=15)
        cv2.imwrite("plots/1_render.png", img[t:b, l:r])

        if crop:
            img = np.dstack([clean_lm] * 3)
            cv2.rectangle(img, (2850, 2780), (2950, 2880), (0, 0, 255), thickness=2)
            cv2.putText(img, "6.5mm", (2960, 2840), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite("plots/0_clean_crop.png", img[ct:cb, cl:cr])

            # cv2.imwrite("plots/0_clean_crop.png", clean_lm[ct:cb, cl:cr])
            cv2.imwrite("plots/1_render_crop.png", render_lm[ct:cb, cl:cr])

    plot_img(render-clean, "Diff", lims=[l, r, t, b], title=False, axis='off', cmap="bwr", vmin=-lim, vmax=lim)
    if save:
        plt.savefig("plots/diff.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plot_img(np.abs(render-clean) > thr, "Diff Mask", lims=[l, r, t, b], title=False, bar=False, axis='off')
    if save:
        plt.savefig("plots/diff_mask_" + str(thr) + ".png", dpi=200, bbox_inches='tight', pad_inches=0.22)

    if hist:
        def plot_hist(name, data, bins, range, title=None, semilog=False):
            plt.figure("Hist " + name, (16, 9))
            plt.hist(data, bins=bins, range=range)

            if title:
                plt.title(title)
            if semilog:
                plt.semilogy()
            plt.tight_layout()

            if save:
                plt.savefig("plots/hist_" + name.lower() + ".png", dpi=200)

        diff = (render - clean)[t:b, l:r].ravel()
        # diff = diff[np.abs(diff) > thr]
        plot_hist("All", diff, bins=1000, range=[-0.5, 0.5],
                  title="Diff (std=%.3f)" % np.std(diff), semilog=True)

        diff = (render - clean)[ct:cb, cl:cr].ravel()
        plot_hist("Crop", diff, bins=1000, range=[-0.5, 0.5],
                  title="Diff (std=%.3f)" % np.std(diff), semilog=True)

        row, col, sz = 2800, 3000, 50
        diff = (render - clean)[row:row + sz, col:col + sz].ravel()
        plot_hist("Patch", diff, bins=500, range=[-0.1, 0.1],
                  title="Diff Patch (mean=%.3f, std=%.3f)" % (np.mean(diff), np.std(diff)))

        row, col, sz = 2825, 3200, 20
        noise = clean[row:row + sz, col:col + sz].ravel()
        plot_hist("Noise", noise - np.mean(noise), bins=500, range=[-0.003, 0.003],
                  title="Noise (level=%.6f, std=%.6f)" % (np.mean(noise), np.std(noise)))

    if crop:
        font = {'family': 'serif', 'weight': 'normal', 'size': 48}
        matplotlib.rc('font', **font)

        plot_img(render - clean, "Diff Crop", lims=[cl, cr, ct, cb], title=False, axis='off', cmap="bwr", vmin=-lim, vmax=lim)
        if save:
            filename = "plots/diff_crop.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
            img = cv2.imread(filename)
            cv2.rectangle(img, (2100, 100), (2750, 550), (0, 255, 0), thickness=18)
            cv2.imwrite(filename, img)

        plot_img(np.abs(render-clean) > thr, "Diff Mask Crop", lims=[cl, cr, ct, cb], title=False, bar=False, axis='off')
        if save:
            filename = "plots/diff_mask_" + str(thr) + "_crop.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0)
            img = cv2.imread(filename)
            cv2.rectangle(img, (1520, 900), (1950, 1180), (255, 200, 0), thickness=15)
            cv2.imwrite(filename, img)


if __name__ == "__main__":
    prepare_images()

    compare(crop=True, hist=True, save=True)

    plt.show()
