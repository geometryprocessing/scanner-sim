import itertools
import os
import cv2
import json
import Imath
import OpenEXR
import joblib
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.morphology as morph
from skimage import measure
from skimage import filters
from scipy.spatial.transform import Rotation as R

from utils import ensure_exists, save_ply, plot_3d, plot_image, load_openexr, load_ply
from decoding import load_decoded


def img_to_ray(p_img, mtx):
    p_img = (p_img - mtx[:2, 2]) / mtx[[0, 1], [0, 1]]
    return np.concatenate([p_img, np.ones((p_img.shape[0], 1))], axis=1)


def triangulate(cam_rays, proj_xy, proj_calib, undistort=True):
    if undistort:
        u_proj_xy = cv2.undistortPoints(proj_xy.astype(np.float), proj_calib["mtx"], proj_calib["dist"]).reshape((-1, 2))
    else:
        #print("No undistort")
        u_proj_xy = cv2.undistortPoints(proj_xy.astype(np.float), proj_calib["new_mtx"], None).reshape((-1, 2))
    #print(u_proj_xy[:10, :])
    proj_rays = np.concatenate([u_proj_xy, np.ones((u_proj_xy.shape[0], 1))], axis=1)
    proj_rays = np.matmul(proj_calib["basis"].T, proj_rays.T).T
    proj_origin = proj_calib["origin"]

    v12 = np.sum(np.multiply(cam_rays, proj_rays), axis=1)
    v1, v2 = np.linalg.norm(cam_rays, axis=1)**2, np.linalg.norm(proj_rays, axis=1)**2
    L = (np.matmul(cam_rays, proj_origin) * v2 + np.matmul(proj_rays, -proj_origin) * v12) / (v1 * v2 - v12**2)

    return cam_rays * L[:, None]

def calculate_normals_from_p3d(points, mask):
    dx = (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)) / 1
    dy = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 1
    normals = - np.cross(dx, dy)
    #normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    return normals
    
def calculate_normals_from_dm(dm):
    zy, zx = np.gradient(dm)
    normals = np.dstack((zx, zy, -np.ones_like(dm)))
#    n = np.linalg.norm(normals, axis=2)
    normals = normals / np.linalg.norm(normals, axis=2)[:, :, None]
    return normals


def reconstruct_single(data_path, cam_calib, proj_calib, out_dir="/reconstructed", max_group=25, gen_depth_map=True,
                       save=True, plot=False, save_figures=True, verbose=False, extract_normals=True, extract_colors=True, sim=False, **kw):
    
    if sim:
        white_path = data_path + "/img_000.exr"
    else: 
        white_path = data_path + "/img_color.exr"
    
    if save:
        save_path = data_path + out_dir + "/"
        ensure_exists(save_path)
        data_path += "/decoded/"
    else:
        save_path = None

    all, groups = load_decoded(data_path)
    cam_xy, proj_xy, mask = all
    
    if groups:
        group_cam_xy, group_proj_xy, group_counts, group_rcs = groups
    undistorted = bool(open(data_path + "undistorted.txt", "r").read())
    #print("Loaded:", data_path)

    if undistorted:
        u_cam_xy = cv2.undistortPoints(cam_xy.astype(np.float), cam_calib["new_mtx"], None).reshape((-1, 2))
        if groups:
            u_group_cam_xy = cv2.undistortPoints(group_cam_xy.astype(np.float), cam_calib["new_mtx"], None).reshape((-1, 2))
    else:
        u_cam_xy = cv2.undistortPoints(cam_xy.astype(np.float), cam_calib["mtx"], cam_calib["dist"]).reshape((-1, 2))
        if groups:
            u_group_cam_xy = cv2.undistortPoints(group_cam_xy.astype(np.float), cam_calib["mtx"], cam_calib["dist"]).reshape((-1, 2))
            
    cam_rays = np.concatenate([u_cam_xy, np.ones((u_cam_xy.shape[0], 1))], axis=1)
    if groups:
        group_cam_rays = np.concatenate([u_group_cam_xy, np.ones((u_group_cam_xy.shape[0], 1))], axis=1)

    #print("Triangulating...")
    #print(np.sum(mask))#.shape)
    
    #proj_xy = proj_xy +
    offset = 6.0
    proj_xy[:, 0] = proj_xy[:, 0] + offset # 2.5 5.5
    proj_xy[:, 1] = proj_xy[:, 1] + offset # 2.5 5.5
    group_proj_xy[:, 0] = group_proj_xy[:, 0] + offset # 2.5 5.5
    group_proj_xy[:, 1] = group_proj_xy[:, 1] + offset # 2.5 5.5
#    print(proj_xy[:10, :])

    all_points = triangulate(cam_rays, proj_xy, proj_calib)
    if groups:
        group_points = triangulate(group_cam_rays, group_proj_xy, proj_calib)
        idx = np.nonzero(group_counts < max_group)[0]
        #print("%d groups larger than %d excluded" % (group_counts.shape[0] - idx.shape[0], max_group))
        group_points = group_points[idx, :]
        group_cam_xy = group_cam_xy[idx, :]
        
    # Extract colors        
    all_colors = group_colors = None
    if extract_colors:
        
        white, _ = load_openexr(white_path, make_gray=False, load_depth=False)
        #print(np.min(white), np.max(white))
        ma = np.max(white)
        mi = np.min(white)
        white = (white - mi) / (ma - mi)
        all_colors = white
        all_colors = all_colors[mask]
        all_colors = all_colors.reshape(-1, 3)

        group_idxs = group_cam_xy.astype("int32")
        #print(group_idxs.shape)
        group_colors = white[group_idxs[:, 1], group_idxs[:, 0]]


    # Generate depth maps
    if gen_depth_map:
        #print("Generating depth map(s)")
        full_depth_map = np.zeros_like(mask, dtype=np.float32)
        full_depth_map[cam_xy[:, 1], cam_xy[:, 0]] = np.linalg.norm(all_points, axis=1)

        if groups:
            group_depth_map = np.zeros_like(mask, dtype=np.float32)
            for i, id in enumerate(idx):
                rcs = group_rcs[id]
                if i % 100000 == 0 and verbose:
                    print("Group", i)
                group_depth_map[rcs[:, 0], rcs[:, 1]] = np.linalg.norm(group_points[i, :])
                
    # Generate normals
    all_normals = group_normals = None
    if extract_normals:
        full_points = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        full_points[cam_xy[:, 1], cam_xy[:, 0], :] = all_points
        all_normals = calculate_normals_from_p3d(full_points, mask)
        #all_normals = calculate_normals_from_dm(full_depth_map)

        group_idxs = group_cam_xy.astype("int32")
        group_normals = all_normals[group_idxs[:, 1], group_idxs[:, 0]]
        
        all_normals = all_normals[mask]
        
        all_norm = np.linalg.norm(all_normals, axis=1)
        all_nonzero = all_norm > 0
        all_normals[all_nonzero] /= all_norm[all_nonzero, None]
        
        group_norm = np.linalg.norm(group_normals, axis=1)
        group_nonzero = group_norm > 0
        group_normals[group_nonzero] /= group_norm[group_nonzero, None]


    if save:
        save_ply(save_path + "all_points.ply", all_points, all_normals, all_colors)
        if groups:
            #print(group_points.shape, group_normals.shape, group_colors.shape)
            save_ply(save_path + "group_points.ply", group_points, group_normals, group_colors)
            with open(save_path + "max_group_size.txt", "w") as f:
                f.write(str(max_group))

        if gen_depth_map:
            np.save(save_path + "full_depth_map.npy", full_depth_map.astype(np.float32))
            if groups:
                np.save(save_path + "group_depth_map.npy", group_depth_map.astype(np.float32))

    if plot:
        if not save_figures:
            save_path = None

        #plot_3d(all_points[::1000, :], "All Points", save_as=save_path + "all_points" if save_path else None)
        #if groups:
        #    plot_3d(group_points[::100, :], "Group Points", save_as=save_path + "group_points" if save_path else None)

        if gen_depth_map:
            vmin = np.min(full_depth_map[full_depth_map > 1])
            #plot_image(full_depth_map, "Full Depth Map", data_path + " - Full Depth Map", vmin=vmin, save_as=save_path + "full_depth_map" if save_path else None)
            if groups:
                vmin = np.min(full_depth_map[group_depth_map > 1])
                plot_image(group_depth_map, "Depth Map", data_path + " - Depth Map", vmin=vmin, save_as=save_path + "group_depth_map" if save_path else None)

    return group_points, group_colors, group_depth_map/1000.0 if groups else None


def reconstruct_many(path_template, cam_calib, proj_calib, suffix="gray/", **kw):
    paths = glob.glob(path_template)
    print("Found %d directories:" % len(paths), paths)

    jobs = [joblib.delayed(reconstruct_single, check_pickle=False)
            (path + "/" + suffix, cam_calib, proj_calib, **kw) for path in paths]

    results = joblib.Parallel(verbose=15, n_jobs=8, batch_size=1, pre_dispatch="all")(jobs)

    return {path: result for path, result in zip(paths, results)}
    
    



def merge_single(data_path, filename_template, stage_calib, max_dist=100, title="Merged", skip=1000, plot=False, sim=False, max_range=12, **kw):
    if sim:
        files = [data_path + filename_template % a for a in range(max_range)]
    else:
        files = [data_path + filename_template % (a * 30) for a in range(max_range)]
      
    points = []
    normals = []
    colors = []
    
    for fi in files:
        poi, nor, col = load_ply(fi)
        points.append(poi)
        normals.append(nor)
        colors.append(col)
   
    p0, dir = stage_calib["p"], stage_calib["dir"]
    points = list(map(lambda p: p - p0, points))
    merged = [points[0]]
    merged_normals = [normals[0]]

    if plot:
        ax = plot_3d(points[0][::skip, :], title, label=str(0) + " deg")
        # line(ax, p0 - 10 * dir, p0 + 100 * dir, "-r")
        

    angle = int(360/max_range)
    for i in range(len(points) - 1):
        rot = R.from_rotvec((-(i + 1) * angle * np.pi / 180) * dir)
        p_rot = rot.apply(points[i+1])
        n_rot = rot.apply(normals[i+1])
        merged.append(p_rot)
        merged_normals.append(n_rot)

        #if plot:
            #plt.scatter(ax, p_rot[::skip, :], s=5, label=str(30*(i + 1)) + " deg")

    merged = np.concatenate(merged, axis=0)
    m_normals = np.concatenate(merged_normals, axis=0)
    colors = np.concatenate(colors, axis=0)
    proj = dir[None, :] * np.matmul(merged, dir)[:, None]
    dist = np.linalg.norm(merged - proj, axis=1)
    merged = merged[dist < max_dist, :] + p0
    #print(merged.shape, colors.shape, m_normals.shape, len(merged_normals))
    m_normals = m_normals[dist < max_dist, :]
    colors = colors[dist < max_dist, :]

    if plot:
        plt.legend()

    return merged, m_normals, colors


def merge_both(data_path, object_name, stage_calib, save=True, plot=False, save_figures=True, sim=False, **kw):
    if save:
        save_path = data_path + "/reconstructed/"
        ensure_exists(save_path)

    #for suffix, skip in zip(["all", "group"], [10000, 1000]):
    for suffix in ["group"]:
        print("Merging object: %s (%s)" % (object_name, suffix))
        if sim:
            merged, normals, colors = merge_single(data_path, "/rot_%s/reconstructed/%s_points.ply" % ("%03i", suffix),
                                     stage_calib, title=object_name + "_" + suffix, plot=plot, sim=sim, **kw)
        else:
            merged, normals, colors = merge_single(data_path, "/position_%s/gray/reconstructed/%s_points.ply" % ("%d", suffix), stage_calib, title=object_name + "_" + suffix, plot=plot, sim=sim, **kw)
                                     
        #print("/rot_%s/reconstructed/%s_points.ply" % ("%d", suffix))

        if save:
            filename = object_name + "_%s.ply" % suffix
            print("Saving", filename)
            save_ply(save_path + filename, merged, normals, colors)

            if plot and save_figures:
                plt.savefig(save_path + object_name + "_%s.png" % suffix, dpi=160)
