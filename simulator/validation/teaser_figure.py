from configuration import *
from rendering import *
from display import *

font = {'family': 'serif', 'weight': 'normal', 'size': 32}
matplotlib.rc('font', **font)

calib_path = "../../data/calibrations/"
valid_path = "../../data/validation/teaser_figure/"


def simulate_teaser_figure(data_path, mitsuba_path, pawn_geometry, verbose=True, **kw):
    config = configure_camera_and_projector(calib_path=calib_path, **kw)

    H, W = 1080, 1920
    pattern = gen_checker((H, W), (90, 60), 100, (9, 18))
    pattern[:30, :, :] = 0
    pattern[H - 30:, :, :] = 0

    cv2.imwrite(data_path + "checker.png", pattern[:, :, ::-1])
    process_patterns(data_path + "checker.png", calib_path, verbose=True)

    copy_to(data_path, valid_path + "pawn.obj")
    configure_object(config, pawn_geometry, calib_path=calib_path, obj_mat="material/rough_plastic.xml")

    config["pro_pattern_file"] = "checker.png"
    config["amb_radiance"] = 0.0
    write_scene_file(config, data_path + "pawn.xml", calib_path + "../scenes/scene_default.xml")

    source(mitsuba_path + "/setpath.sh")
    render_scenes(data_path + "/*.xml", verbose=verbose)


def analyze_teaser_figure(data_path, lim=1.03, thr=0.05, roi=True, crop=False, hist=False, save=False):
    data_path += "/"
    render = load_openexr(data_path + "rendered.exr")
    clean = load_openexr(data_path + "captured.exr")
    print("Loaded")

    mask = render > 1.e-6
    clean[~mask] = 0

    l, r, t, b = 2000, 4250, 900, 3700
    cl, cr, ct, cb = 2750, 3250, 2725, 3050
    nt, nl, ns = 1425, 3200, 50

    if roi:
        render_thr = np.average(render[nt:(nt+ns), nl:(nl+ns)])
        clean_thr = np.average(clean[nt:(nt+ns), nl:(nl+ns)])
    else:
        _, render_thr = linear_map(render)
        _, clean_thr = linear_map(clean)
        render_thr, clean_thr = render_thr/1.2, clean_thr/1.2

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
    plot_img(clean, "Clean")
    plot_img(mask, "Mask")

    if save:
        ensure_exists(data_path + "plots/")

        img = np.dstack([255 - clean_lm]*3)
        cv2.rectangle(img, (cl, ct), (cr, cb), (0, 0, 255), thickness=15)
        cv2.imwrite(data_path + "plots/0_clean.png", img[t:b, l:r])

        img = np.dstack([255 - render_lm]*3)
        cv2.rectangle(img, (nl, nt), (nl+2*ns, nt+2*ns), (0, 255, 0), thickness=15)
        cv2.imwrite(data_path + "plots/1_render.png", img[t:b, l:r])

        if crop:
            img = np.dstack([clean_lm] * 3)
            cv2.rectangle(img, (2850, 2780), (2950, 2880), (0, 0, 255), thickness=2)
            cv2.putText(img, "6.5mm", (2960, 2840), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(data_path + "plots/0_clean_crop.png", img[ct:cb, cl:cr])

            # cv2.imwrite(data_path + "plots/0_clean_crop.png", clean_lm[ct:cb, cl:cr])
            cv2.imwrite(data_path + "plots/1_render_crop.png", render_lm[ct:cb, cl:cr])

    plot_img(render-clean, "Diff", lims=[l, r, t, b], title=False, axis='off', cmap="bwr", vmin=-lim, vmax=lim)
    if save:
        plt.savefig(data_path + "plots/diff.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plot_img(np.abs(render-clean) > thr, "Diff Mask", lims=[l, r, t, b], title=False, bar=False, axis='off')
    if save:
        plt.savefig(data_path + "plots/diff_mask_" + str(thr) + ".png", dpi=200, bbox_inches='tight', pad_inches=0.22)

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
                plt.savefig(data_path + "plots/hist_" + name.lower() + ".png", dpi=200)

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
            filename = data_path + "plots/diff_crop.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
            img = cv2.imread(filename)
            cv2.rectangle(img, (2100, 100), (2750, 550), (0, 255, 0), thickness=18)
            cv2.imwrite(filename, img)

        plot_img(np.abs(render-clean) > thr, "Diff Mask Crop", lims=[cl, cr, ct, cb], title=False, bar=False, axis='off')
        if save:
            filename = data_path + "plots/diff_mask_" + str(thr) + "_crop.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0)
            img = cv2.imread(filename)
            cv2.rectangle(img, (1520, 900), (1950, 1180), (255, 200, 0), thickness=15)
            cv2.imwrite(filename, img)


def prepare_captured_teaser_figure(data_path):
    scan = load_openexr(data_path + "/position_0/color/checker.exr")
    parasitic = load_openexr(data_path + "/position_0/gray/img_01.exr")
    vignetting = cv2.cvtColor(cv2.imread(calib_path + "camera_vignetting.png"), cv2.COLOR_BGR2GRAY)
    cam_calib = load_calibration(calib_path + "camera_geometry.json")
    print("captured", scan.shape, scan.dtype)

    clean = np.maximum(0, scan - parasitic)
    clean /= vignetting / np.max(vignetting)
    undistorted = cv2.undistort(clean, cam_calib["mtx"], cam_calib["dist"], newCameraMatrix=cam_calib["new_mtx"])

    save_openexr(valid_path + "captured.exr", undistorted)


if __name__ == "__main__":
    # Point to the unzipped location of pawn_30_deg_no_ambient.zip
    # from Physical Scans (https://archive.nyu.edu/handle/2451/63306)
    # prepare_captured_teaser_figure("/media/yurii/EXTRA/scanner-sim-data/pawn_30_deg_no_ambient")

    mitsuba_path = "/home/vida/software/mitsuba/"
    rendered_path = mitsuba_path + "scenes/teaser_figure/"
    ensure_exists(rendered_path)

    pawn_geometry = load_calibration(valid_path + "pawn_geometry.json")
    simulate_teaser_figure(rendered_path, mitsuba_path, pawn_geometry, cam_samples=(256))
    copy_to(valid_path + "rendered.exr", rendered_path + "pawn.exr")

    analyze_teaser_figure(valid_path, crop=True, roi=False, hist=True, save=False)

    # rgb, d = load_openexr(rendered_path + "pawn.exr", make_gray=False, load_depth=True)
    #
    # plt.figure("Depth", (12, 10))
    # plt.imshow(d)
    # plt.colorbar()
    # plt.tight_layout()
    #
    # plt.figure("Img", (12, 10))
    # plt.imshow(rgb[:, :, 0])
    # plt.colorbar()
    # plt.tight_layout()

    plt.show()
