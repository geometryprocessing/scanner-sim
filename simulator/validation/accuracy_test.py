from configuration import *
from rendering import *
from display import *
from cv2 import aruco

import matplotlib
matplotlib.use('TkAgg')

calib_path = "../../data/calibrations/"
valid_path = "../../data/validation/accuracy_test/"


# units = mm
def gen_charuco_texture(n=25, m=18, size=(400, 300), checker_size=15, pixels_per_unit=10, contrast=15):
    w, h = size[0] * pixels_per_unit, size[1] * pixels_per_unit
    cp = checker_size * pixels_per_unit

    pad_x = (w - n * cp) // 2
    pad_y = (h - m * cp) // 2

    img = np.zeros((h, w), dtype=np.uint8)
    img[...] = 255

    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(n, m, cp, cp * 12 // 15, aruco_dict)

    b_w, b_h, ms = n * cp, m * cp, cp * 3 // (2 * 15)
    img[pad_y:pad_y+b_h, pad_x:pad_x+b_w] = aruco.drawPlanarBoard(board, (b_w, b_h), marginSize=ms, borderBits=1)
    
    for i in range(m):
        for j in range(n):
            if (i + j) % 2 == 1:
                img[pad_y + i * cp:pad_y + (i+1) * cp, pad_x + j * cp:pad_x + (j+1) * cp] = 0

    img[img == 0] = round(255 / contrast)

    return img


# def test_accuracy(data_path, camera_calib, proj_calib, captured=None, rendered=None, save=False, plot=False, save_figures=True, **kw):
#     cam_mtx, cam_dist, cam_new_mtx = camera_calib["mtx"], camera_calib["dist"], camera_calib["new_mtx"]
#     charuco, checker, plane_errors = reconstruct_planes(data_path, camera_calib, extra_output=True, **kw)
#     charuco_img, charuco_3d, charuco_id, charuco_frame = charuco
#     checker_img, checker_3d, checker_2d, checker_local = checker
#     avg_plane_errors, all_plane_errors = plane_errors
#     w, h = 1920, 1080
#
#     print("\nReconstructed:", len(checker_3d))
#     print("Mean plane error", np.mean(avg_plane_errors))
#     assert(len(checker_3d) == 1)
#
#     T, R = charuco_frame[0]
#
#     print("Plane location:")
#     print("T:", T, "\nR:\n", R, "\n")
#
#     cam_offsets = np.zeros(2)
#     if rendered is not None:
#         cam_offsets = np.array([(6464 - 1)/2, (4852 - 1)/2]) - cam_new_mtx[:2, 2]
#     print("Camera offsets:", cam_offsets)
#
#     u_cam_xy = cv2.undistortPoints(checker_img[0].astype(np.float).reshape((-1, 1, 2)), cam_new_mtx, None).reshape((-1, 2))
#     cam_rays = np.concatenate([u_cam_xy, np.ones((u_cam_xy.shape[0], 1))], axis=1)
#
#     # proj_offsets = np.array([0., 0.])
#     proj_offsets = np.array([0.5, 0.5])
#     print("Projector offsets:", proj_offsets)
#
#     triangulated_3d = triangulate(cam_rays, checker_2d[0] + proj_offsets[None, :], proj_calib)
#     prj_triangulated = cv2.projectPoints(triangulated_3d, np.eye(3), np.zeros(3), cam_new_mtx, None)[0].reshape(-1, 2)
#
#     d0, d = np.linalg.norm(checker_3d[0], axis=1), np.linalg.norm(triangulated_3d, axis=1)
#     err = np.linalg.norm(triangulated_3d - checker_3d[0], axis=1)
#     print(err, "\nMean triangulation error:", np.mean(err), "mm")
#     print("Mean triangulation bias:", np.mean(d - d0), "mm")
#
#     if plot:
#         plt.figure("Plane Reconstruction", (12, 12))
#         plt.clf()
#         ax = plt.subplot(111, projection='3d', proj_type='ortho')
#         ax.set_title("Plane Reconstruction")
#
#         scatter(ax, charuco_3d[0], c="g", s=5, label="Charuco Corners")
#         scatter(ax, checker_3d[0], c="b", s=8, label="Checker Corners")
#         board(ax, T, R, label="Charuco Board")
#
#         ax.set_xlabel("x, mm")
#         ax.set_ylabel("z, mm")
#         ax.set_zlabel("-y, mm")
#         plt.legend()
#         plt.tight_layout()
#         axis_equal_3d(ax)
#
#         if save_figures:
#             ax.view_init(elev=10, azim=-20)
#             plt.savefig(data_path + "/reconstruction_view1.png", dpi=120)
#             ax.view_init(elev=12, azim=26)
#             plt.savefig(data_path + "/reconstruction_view2.png", dpi=120)
#
#         if captured is not None and rendered is not None:
#             img_1 = np.minimum(1.2 * captured/np.max(captured), 1.)
#             img_2 = rendered/np.max(rendered)
#             img_1[1:, 3:] = img_1[:-1, :-3]  # Correct for optical axis offset
#
#             plt.figure("Image Overlay", (16, 10))
#             plt.imshow(img_1 - img_2)
#             # plt.imshow((img_1 + img_2) * 0.5)
#             # plt.imshow(np.stack([img_1, img_2, np.zeros_like(img_1)], axis=2))
#             plt.colorbar()
#             plt.tight_layout()
#
#             if save_figures:
#                 plt.savefig(data_path + "/image_overlay.png", dpi=120)
#
#             if rendered is not None:
#                 detected, img = detect_checker(rendered, draw_on=None, n=17, m=8, size=100)
#                 prj_detected = detected[0].reshape((-1, 2)) if detected[0] is not None else None
#             else:
#                 prj_detected = None
#
#             plt.figure("Marker Overlay", (16, 10))
#             plt.imshow(rendered if rendered is not None else captured)
#             plt.colorbar()
#
#             plt.plot(charuco_img[0][:, 0] + cam_offsets[0], charuco_img[0][:, 1] + cam_offsets[1], "b+", markersize=12, label="Detected Charuco")
#             plt.plot(checker_img[0][:, 0] + cam_offsets[0], checker_img[0][:, 1] + cam_offsets[1], "bx", markersize=12, label="Detected Checker")
#
#             prj_charuco = cv2.projectPoints(charuco_3d[0], np.eye(3), np.zeros(3), cam_new_mtx, None)[0].reshape(-1, 2)
#             prj_checker = cv2.projectPoints(checker_3d[0], np.eye(3), np.zeros(3), cam_new_mtx, None)[0].reshape(-1, 2)
#
#             plt.plot(prj_charuco[:, 0] + cam_offsets[0], prj_charuco[:, 1] + cam_offsets[1], "r+", markersize=9, label="Projected Charuco")
#             if rendered is not None:
#                 if prj_detected is not None:
#                     plt.plot(prj_detected[:, 0], prj_detected[:, 1], "rx", markersize=9, label="Rendered Checker")
#             else:
#                 plt.plot(prj_triangulated[:, 0] + cam_offsets[0], prj_triangulated[:, 1] + cam_offsets[1], "rx", markersize=9, label="Triangulated Checker")
#
#             plt.tight_layout()
#             plt.legend()
#
#             if save_figures:
#                 plt.savefig(data_path + "/marker_overlay.png", dpi=120)
#
#             if prj_detected is not None:
#                 ch_d = np.linalg.norm(checker_img[0] + cam_offsets[None, :] - prj_detected, axis=1)
#                 print("Mean rendering error:", np.mean(ch_d), "camera pixels")
#
#                 plt.figure("Rendering errors", (16, 9))
#                 plt.subplot(211)
#                 plt.imshow((ch_d).reshape(8, 17))
#                 plt.title("Rendering errors, camera pixels")
#                 plt.colorbar()
#                 plt.subplot(212)
#                 plt.hist(ch_d, bins=50)
#                 plt.title("Mean = %.3f camera pixels" % np.mean(ch_d))
#                 plt.tight_layout()
#
#                 if save_figures:
#                     plt.savefig(data_path + "/rendering_errors.png", dpi=120)
#
#                 plt.figure("Rendering Errors - Paper Edition", (4, 3.5))
#                 plt.hist(ch_d, bins=50, range=[0, 5])
#                 plt.xlabel("Rendering error, camera pixels")
#                 plt.ylabel("Checker corner counts")
#                 plt.tight_layout()
#
#                 if save_figures:
#                     plt.savefig(data_path + "/rendering_errors_paper.png", dpi=300)
#
#
#         plt.figure("Triangulation errors", (16, 9))
#         plt.subplot(211)
#         plt.imshow((d-d0).reshape(8, 17))
#         plt.title("Triangulation errors, mm depth")
#         plt.colorbar()
#         plt.subplot(212)
#         plt.hist(d-d0, bins=50)
#         plt.title("Mean = %.3f mm" % np.mean(d-d0))
#         plt.tight_layout()
#
#         if save_figures:
#             plt.savefig(data_path + "/triangulation_errors.png", dpi=120)
#
#     if save:
#         with open(data_path + "plane_location.json", "w") as f:
#             json.dump({"T": T,
#                        "R": R,
#                        "T_Help": "[x, y, z] in mm in camera's frame of reference",
#                        "R_Help": "[ex, ey, ez]",
#                        "Camera": "https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html"},
#                       f, indent=4, cls=NumpyEncoder)
#
#     return T, R


def process_accuracy_test(data_path, camera_calib, undistorted=False, reuse_corners=False):
    if not reuse_corners:
        # For accuracy_test data captured with physical setup "combined/" subfolder needs to be created
        # with gray/img_01.exr copied as blank_0.exr, gray/img_00.exr as white_0.exr, and
        # color/checker.exr as checker_0.exr to conform to naming convention of process_stage()
        process_stage(data_path)

    charuco, checker, plane_errors = reconstruct_planes(data_path, camera_calib, undistorted=undistorted)
    charuco_3d, charuco_id, charuco_frame = charuco

    print("\nReconstructed:", len(charuco_3d), "plane(s)")
    print("Mean plane error:", np.mean(plane_errors[0]))
    assert len(charuco_3d) == 1, "Found more than one board!"

    T, R = charuco_frame[0]
    R = R.T  # match convention (row-wise) when saving to json
    print("\nBoard location:")
    print("T:", T, "\nR:\n", R, "\n")

    with open(data_path + "/board_geometry.json", "w") as f:
        json.dump({"obj_file": "charuco_board.obj",
                   "obj_type": "obj",
                   "obj_translation": T / 1000.,
                   "obj_rotation": R,
                   "obj_scale": 1.0}, f, indent=4, cls=NumpyEncoder)


def simulate_accuracy_test(data_path, mitsuba_path, board_geometry, reuse_patterns=False, verbose=True, **kw):
    config = configure_camera(calib_path=calib_path, **kw)
    configure_projector(config, calib_path=calib_path, **kw)

    data_path += "/"
    ensure_exists(data_path)

    if not reuse_patterns:
        pattern = np.zeros((1080, 1920, 3), dtype=np.uint8)
        imageio.imwrite(data_path + "blank_0.png", pattern)

        pattern[...] = 255
        imageio.imwrite(data_path + "white_0.png", pattern)

        pattern = gen_checker((H, W), (90, 60), 100, (9, 18))
        imageio.imwrite(data_path + "checker_0.png", pattern)

        process_patterns(data_path + "*_0.png", calib_path, verbose=True)
    else:
        # blank_0.png might get overwritten by process_stage()
        if verbose:
            print("Restoring pattern:", data_path + "blank_0.png")
        pattern = np.zeros((1080 * 4, 1920 * 4, 3), dtype=np.uint8)
        imageio.imwrite(data_path + "blank_0.png", pattern)
        # process_patterns(data_path + "blank_0.png", calib_path, verbose=True)

    copy_to(data_path, calib_path + "../objects/charuco_board/charuco_board.*")
    configure_object(config, board_geometry, calib_path=calib_path, obj_mat="materials/textured_obj.xml")

    for pattern in ["blank_0.png", "white_0.png", "checker_0.png"]:
        config["pro_pattern_file"] = pattern
        config["amb_radiance"] = 0.5
        write_scene_file(config, data_path + "%s.xml" % pattern[:-4], valid_path + "accuracy_test.xml")

        if pattern == "checker_0.png":
            config["board_texture_name"] = "white_0.png"
            write_scene_file(config, data_path + "checker_clean.xml", valid_path + "accuracy_test.xml")

    source(mitsuba_path + "/setpath.sh")
    render_scenes(data_path + "/*.xml", verbose=verbose)
    # render_scenes(data_path + "/*_0.xml", verbose=verbose)
    # render_scenes(data_path + "/checker_0.xml", verbose=verbose)


def analyze_accuracy_test(captured_path, rendered_path, camera_calib, board_geom, better_checker_ref=False,
                          proj_calib=None, real_img_ref=True, plot=True, savefigs=False):
    charuco_cap = load_corners(captured_path + "/charuco/corners.json")["blank_0.png"]
    checker_cap = load_corners(captured_path + "/checker/detected_full/corners.json")["checker_0.png"]

    charuco_cap["img"] = cv2.undistortPoints(charuco_cap["img"].reshape((-1, 1, 2)),
                                             camera_calib["mtx"], camera_calib["dist"],
                                             P=camera_calib["new_mtx"]).reshape((-1, 2))

    checker_cap["img"] = cv2.undistortPoints(checker_cap["img"].reshape((-1, 1, 2)),
                                             camera_calib["mtx"], camera_calib["dist"],
                                             P=camera_calib["new_mtx"]).reshape((-1, 2))

    T, R = board_geom["obj_translation"] * 1000.0, board_geom["obj_rotation"].T
    charuco_ref = cv2.projectPoints(charuco_cap["obj"], cv2.Rodrigues(R)[0], T, camera_calib["new_mtx"], None)[0].reshape(-1, 2)

    charuco_ren = load_corners(rendered_path + "/charuco/corners.json")["blank_0.png"]
    checker_ren = load_corners(rendered_path + "/checker/detected_full/corners.json")["checker_0.png"]

    if not os.path.exists(rendered_path + "/checker_clean.png"):
        save_ldr(rendered_path + "/checker_clean.png",
                 linear_map(load_openexr(rendered_path + "/checker_clean.exr"))[0])
    clean_ldr = load_ldr(rendered_path + "/checker_clean.png", make_gray=True)

    checker_ref, _ = detect_checker(clean_ldr, n=17, m=8, size=100, draw_on=clean_ldr)
    checker_ref = checker_ref[0].reshape((-1, 2))

    if better_checker_ref and proj_calib is not None:
        checker_cap["img"] = checker_ref
        proj_rays = cv2.undistortPoints((checker_cap["obj"][:, :2].astype(np.float) +
                                         np.array([160 + 0.5, 190 + 0.5])[None, :]).reshape((-1, 1, 2)),
                                         proj_calib["mtx"], proj_calib["dist"]).reshape((-1, 2))
        proj_rays = np.concatenate([proj_rays, np.ones((proj_rays.shape[0], 1))], axis=1)
        proj_rays = np.matmul(proj_calib["basis"].T, proj_rays.T).T

        ref_3d = np.zeros((proj_rays.shape[0], 3))
        for i in range(proj_rays.shape[0]):
            ref_3d[i, :] = trace_ray(T, R, proj_calib["origin"], proj_rays[i, :])

        checker_ref = cv2.projectPoints(ref_3d, cv2.Rodrigues(np.eye(3))[0], np.zeros((3)),
                                        camera_calib["new_mtx"], None)[0].reshape(-1, 2)

        # if plot:
        #     plt.figure("Checker Reconstruction", (12, 12))
        #     plt.clf()
        #     ax = plt.subplot(111, projection='3d', proj_type='ortho')
        #     ax.set_title("Plane Reconstruction")
        #
        #     scatter(ax, ref_3d, c="b", s=8, label="Checker Corners")
        #     board(ax, T, R, label="Charuco Board")
        #
        #     # scatter(ax, proj_rays + proj_calib["origin"][None, :], c="c", s=8, label="Projector Rays")
        #     # basis(ax, proj_calib["origin"], proj_calib["basis"].T)
        #
        #     ax.set_xlabel("x, mm")
        #     ax.set_ylabel("z, mm")
        #     ax.set_zlabel("-y, mm")
        #     plt.legend()
        #     plt.tight_layout()
        #     axis_equal_3d(ax)
        #
        #     # if savefigs:
        #     #     ax.view_init(elev=10, azim=-20)
        #     #     plt.savefig(valid_path + "/reconstruction_view1.png", dpi=120)
        #     #     ax.view_init(elev=12, azim=26)
        #     #     plt.savefig(valid_path + "/reconstruction_view2.png", dpi=120)

    if real_img_ref:
        if not os.path.exists(captured_path + "/checker_0.png"):
            save_ldr(captured_path + "/checker_0.png",
                     tone_map(load_openexr(captured_path + "/checker_0.exr"))[0])
        ref_img = load_ldr(captured_path + "/checker_0.png", make_gray=False)
        ref_img = cv2.undistort(ref_img, cam_calib["mtx"], cam_calib["dist"], newCameraMatrix=cam_calib["new_mtx"])
    else:
        ref_img = load_ldr(rendered_path + "/blank_0.png", make_gray=True)
        ref_img = np.stack([ref_img, clean_ldr, ref_img], axis=2)
        # ref_img = np.stack([clean_ldr, clean_ldr, clean_ldr], axis=2)
        ref_img = np.repeat(cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)[:, :, None], 3, axis=2)

    if plot:
        plt.figure("Corners", (32, 24))
        plt.imshow(ref_img)

        plt.plot(charuco_ref[:, 0], charuco_ref[:, 1], "g*", markersize=9, label="charuco_ref")
        plt.plot(charuco_ren["img"][:, 0], charuco_ren["img"][:, 1], "b+", markersize=12, label="charuco_ren")
        plt.plot(charuco_cap["img"][:, 0], charuco_cap["img"][:, 1], "rx", markersize=9, label="charuco_cap")

        plt.plot(checker_ref[:, 0], checker_ref[:, 1], "g*", markersize=12, label="checker_ref")
        plt.plot(checker_ren["img"][:, 0], checker_ren["img"][:, 1], "b+", markersize=16, label="checker_ren")
        plt.plot(checker_cap["img"][:, 0], checker_cap["img"][:, 1], "rx", markersize=12, label="checker_cap")

        plt.title("Corners")
        plt.legend()
        plt.tight_layout()

        if savefigs:
            plt.savefig(valid_path + "/accuracy_test_corners.png", dpi=150)


        plt.figure("Errors", (16, 8))
        plt.subplot(121, title="Charuco corner displacements (camera pixels)")

        first = True
        for i, id in enumerate(charuco_cap["idx"]):
            if not id in charuco_ren["idx"]:
                continue

            j = np.nonzero(charuco_ren["idx"] == id)[0][0]
            cap, ren, ref = charuco_cap["img"][i, :], charuco_ren["img"][j, :], charuco_ref[i, :]
            # if ref[0] > 3232:
            #     continue

            plt.plot([0, cap[0] - ref[0]], [0, cap[1] - ref[1]], "r:.", markersize=9,
                                                                 label="Captured" if first else None)
            plt.plot([0, ren[0] - ref[0]], [0, ren[1] - ref[1]], "b:.", markersize=9,
                                                                 label="Rendered" if first else None)
            first = False

        sc = 3
        plt.xlim([-sc, sc])
        plt.ylim([-sc, sc])
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()
        plt.subplot(122, title="Checker corner displacements (camera pixels)")

        first = True
        for i, id in enumerate(checker_cap["idx"]):
            cap, ren, ref = checker_cap["img"][i, :], checker_ren["img"][i, :], checker_ref[i, :]
            # if ref[0] > 3232:
            #     continue

            plt.plot([0, cap[0] - ref[0]], [0, cap[1] - ref[1]], "r:.", markersize=9,
                                                                 label="Captured" if first else None)
            plt.plot([0, ren[0] - ref[0]], [0, ren[1] - ref[1]], "b:.", markersize=9,
                                                                 label="Rendered" if first else None)
            # plt.plot([0, ren[0] - cap[0]], [0, ren[1] - cap[1]], "g:.", markersize=9,
            #                                                      label="Relative" if first else None)
            first = False

        plt.xlim([-sc, sc])
        plt.ylim([-sc, sc])
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if savefigs:
            plt.savefig(valid_path + "/accuracy_test_errors.png", dpi=300)


if __name__ == "__main__":
    # img = gen_charuco_texture()
    # cv2.imwrite(calib_path + "../objects/charuco_board/charuco_board.png", img)
    # plt.imshow(img)
    # plt.show()
    # exit(0)

    cam_calib = load_calibration(calib_path + "camera_geometry.json")
    proj_calib = load_calibration(calib_path + "projector_geometry.json")

    captured_path = "/media/yurii/EXTRA/scanner-sim-data/calibration/accuracy_test/charuco_plane/combined/"
    process_accuracy_test(captured_path, cam_calib, reuse_corners=True)

    copy_to(valid_path, captured_path + "board_geometry.json")
    board_geometry = load_calibration(valid_path + "board_geometry.json")

    mitsuba_path = "/home/yurii/software/mitsuba/"
    rendered_path = mitsuba_path + "scenes/accuracy_test/"
    ensure_exists(rendered_path)

    # simulate_accuracy_test(rendered_path, mitsuba_path, board_geometry, reuse_patterns=False, cam_samples=768)
    process_accuracy_test(rendered_path, cam_calib, reuse_corners=True, undistorted=True)

    analyze_accuracy_test(captured_path, rendered_path, cam_calib, board_geometry, better_checker_ref=True,
                          proj_calib=proj_calib, real_img_ref=True, savefigs=True)
    plt.show()
