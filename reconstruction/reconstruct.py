import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

def reconstruct_object(path, calibration, threshold=127, max_depth=100.0, pro=[1920, 1080], map_type="DIST", dm_name="depth_rec.txt", pc_name="pointcloud_rec.ply",
                       color_image=0, prefix="img", suffix=".png", dec_name="decoded.xml", min_name="min_max.xml"):
    if type(color_image) == int:
        c_image = sorted(glob.glob("%s/%s_*%s"%(path, prefix, suffix)))[color_image]
    else:
        c_image = "%s/%s"%(path, color_image)
    if map_type == "DIST":
        m_type = 0
    elif map_type == "DEPTH":
        m_type = 1
    
    cmd = "./reconstruct '%s/%s' '%s/%s' '%s' %s '%s/%s' '%s/%s' %i %0.1f %i %i %i 1"%(path, dec_name, path, min_name, c_image, calibration, path, dm_name, path, pc_name, threshold, max_depth, pro[0], pro[1], m_type)
    os.system(cmd)
    
    
#reconstruct_object("rendering/results/last_abc/obj_00190086_00", "rendering/calibration_abc.yml", pro=[2048, 2048], threshold=100, prefix="cam")


def reconstruct_python(calibration_path, decoded_path, proj=[1920, 1080], cam=[2048, 2048], threshold = 70, max_dist = 1000):

    with open(decoded_path, "rb") as fi:
        dec = np.load(fi)
    
    #TODO load calibration
    
    

    
    depth_image = np.zeros((dec.shape[1], dec.shape[2]), dtype="float64")
    pro_points = dec[:2] # decoded projector row and column indices
    mask_i = dec[2].astype("bool")
    mask = (1-dec[2]).astype("bool")
    
    # Check if we need this to accomodate for non-square projector pixels
    scale_factor_x = 1.0
    scale_factor_y = 1.0
    pro_points[0] = pro_points[0] / scale_factor_x
    pro_points[1] = pro_points[1] / scale_factor_y
    
    vg = np.tile(np.arange(2048), (2048, 1)) # 1920, 1080
    hg = np.tile(np.arange(2048).reshape(2048,1), (1, 2048)) #1080, 1080, 1920
    cam_points = np.stack([vg, hg]) # camera image row and column indices
    
#     cam_points[0][mask_i] = 0
#     cam_points[1][mask_i] = 0   

#     plt.figure(figsize = (20,10))
#     plt.imshow(cam_points[0])#, cmap="gray")
#     plt.show()
#     plt.figure(figsize = (20,10))
#     plt.imshow(cam_points[1])#, cmap="gray")
#     plt.show()
    
#     plt.figure(figsize = (20,10))
#     plt.imshow(pro_points[0])#, cmap="gray")
#     plt.show()
#     plt.figure(figsize = (20,10))
#     plt.imshow(pro_points[1])#, cmap="gray")
#     plt.show()

    c_points = cam_points[:, mask]#.reshape(2, -1)
    p_points = pro_points[:, mask]#.reshape(2, -1)
    
#calib.cam_K, calib.cam_kc, calib.proj_K, calib.proj_kc, Rt, calib.T, cam, proj, p, &distance);
#def triangulate_stereo(cam_points, pro_points, calibration):
    
    

#calib_path = "rendering/configs/calibration.json"
#reconstruct(calib_path, "rendering/results/obj_gray/rot_fixed/decoded.npy")