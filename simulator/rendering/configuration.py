from utils import *
import re
import cv2
from PIL import Image
import lxml.etree as ET


SCALE_FACTOR = 1 / 1000.0  # configuration is in [mm], mitsuba uses [m]


def transform2string(transform):
    # TODO: add tests for validity of transform
    transform = np.array2string(transform.flatten(), prefix="", suffix="", max_line_width=1000)
    transform = transform.replace("[ ", "").replace(" ]", "")
    transform = re.sub(' +', ' ', transform)
    return transform


# OpenCV pixel coordinates convention:
# (Origin O = (0, 0) is in the *middle* of the top-left pixel)
#  +---+---+---+--
#  | O |x=1|x=2|
#  +---+---+---+--     +---+
#  |y=1|   |   |       |   | - Real image / pattern pixel
#  +---+---+---+--     +---+
#  |y=2|   |   |

# Mitsuba pixel coordinates convention:
# (Origin O = (0, 0) is in the top-left corner of the top-left pixel)
#  O--x=1-x=2-x=3-
#  |   |   |   |
# y=1--+---+---+--     + - +
#  |   |   |   |       .   . - Virtually padded pixel (see below)
# y=2--+---+---+--     + - +
#  |   |   |   |

# For convenience, Camera calibration is in OpenCV convention, but Projector calibration uses Mitsuba convention.
# Reconstruction is done in OpenCV convention, so projector calibration needs to be adjusted (add half a pixel
# to the projector optical axis location at triangulation stage as can be seen in reconstruction/reconstruct.py)
# In contrast, rendering is done in Mitsuba, so camera calibration needs to be adjusted for Mitsuba convention
# while projector calibration values can be used as is. Furthermore, we leverage Mitsuba's cropping function
# for simulation of the camera optical axis offset, so it needs to be rounded to nearest integer value, but
# one can double the precision of optical axis definition by using odd or even rendering resolution:

#                         Odd
#            0   1   2   3   4   5   6   7   8   9   10  - OpenCV convention (camera calibration)
#      + - +---+---+---+---+---+---+---+---+---+---+       input: size = 10, center = 4.0
#      .   |   |   |   |   | C |   |   |   |   |   |
#      + - +---+---+---+---+---+---+---+---+---+---+       output: new_size = 11, offset = 1
#      0   O  O+1 O+2 O+3 O+4 O+5 O+6 O+7 O+8 O+9 O+10   - Mitsuba convention (image rendering)

#                         Even
#            0   1   2   3   4   5   6   7   8   9   10  - OpenCV convention (camera calibration)
#  + - + - +---+---+---+---+---+---+---+---+---+---+       input: size = 10, center = 3.5
#  .   .   |   |   |   |   C   |   |   |   |   |   |
#  + - + - +---+---+---+---+---+---+---+---+---+---+       output: new_size = 12, offset = 2
#  0   1   O  O+1 O+2 O+3 O+4 O+5 O+6 O+7 O+8 O+9 O+10   - Mitsuba convention (image rendering)

def pad_dimension(size, center):
    assert type(size) is int
    assert 0 <= center <= size - 1

    center_i = int(np.floor(center))
    center_f = center - float(center_i)

    if 0.25 < center_f < 0.75:  # even
        center_i += 1
        new_size = 2 * max(center_i, size - center_i)
        offset = new_size - size if center_i < size - center_i else 0
    else:
        # odd
        if center_f > 0.5:
            center_i += 1

        new_size = 1 + 2 * max(center_i, size - center_i - 1)
        offset = new_size - size if center_i < size - center_i - 1 else 0

    return new_size, offset


def configure_camera_geometry(config, cam_geom, half_res=False, quarter_res=False, **kw):
    if type(cam_geom) is str:
        cam_geom = load_calibration(cam_geom)
    assert type(cam_geom) is dict

    w, h = cam_geom["image_width, pixels"], cam_geom["image_height, pixels"]
    if quarter_res:
        w, h = w//4, h//4
        cam_geom["new_mtx"] /= 4
    elif half_res:
        w, h = w//2, h//2
        cam_geom["new_mtx"] /= 2

    new_w, off_w = pad_dimension(w, cam_geom["new_mtx"][0, 2])
    new_h, off_h = pad_dimension(h, cam_geom["new_mtx"][1, 2])

    config["cam_crop_width"], config["cam_crop_height"] = w, h
    config["cam_image_width"], config["cam_image_height"] = new_w, new_h
    config["cam_crop_offset_x"], config["cam_crop_offset_y"] = off_w, off_h

    config["cam_fov_y"] = 2 * np.arctan((new_h / 2.) / cam_geom["new_mtx"][1, 1]) * 180. / np.pi
    config["cam_pixel_aspect"] = cam_geom["new_mtx"][0, 0] / cam_geom["new_mtx"][1, 1]


def configure_camera_focus(config, cam_focus, tolerances_factor=1.07, wavelength_nm=550, **kw):
    if type(cam_focus) is str:
        cam_focus = load_calibration(cam_focus)
    assert type(cam_focus) is dict

    # Convert values to mitsuba scene units (meters)
    config["cam_focus_distance"] = cam_focus["focus, mm"] * SCALE_FACTOR
    # Convert aperture diameter (focus_calib convention) to aperture radius (mitsuba convention)
    config["cam_aperture_radius"] = cam_focus["aperture, mm"] / 2. * SCALE_FACTOR
    # Diffraction limit (in radians) depends on camera aperture (in meters)
    config["cam_diff_limit"] = 1.22 * wavelength_nm * 1.e-9 / (2. * config["cam_aperture_radius"])
    # Account for manufacturing tolerances of the lens and convert from radians to degrees
    config["cam_diff_limit"] *= tolerances_factor * (180. / np.pi)


def configure_projector_geometry(config, proj_geom, brightness=10.0, pixel_gap=0.0, **kw):
    if type(proj_geom) is str:
        proj_geom = load_calibration(proj_geom)
    assert type(proj_geom) is dict

    w, h = proj_geom["image_width, pixels"], proj_geom["image_height, pixels"]
    config["pro_image_width"], config["pro_image_height"] = w, h
    mtx = proj_geom["new_mtx"]
    config["pro_scale_x"], config["pro_scale_y"] = mtx[0, 0], mtx[1, 1]
    config["pro_offset_x"], config["pro_offset_y"] = mtx[0, 2], mtx[1, 2]

    # Projector brightness
    config["pro_intensity"] = brightness
    # Simulate gaps between micro-mirrors in DMD projectors
    config["pro_pixel_gap"] = pixel_gap
    
    # Projector extrinsics
    transform = np.zeros((4,4))
    transform[:3, 3] = proj_geom["origin"] * SCALE_FACTOR
    transform[:3, :3] = proj_geom["basis"].T
    transform[3, 3] = 1.0
    config["pro_transform"] = transform2string(transform)


def configure_projector_focus(config, proj_focus, diff_limit=None, **kw):
    if type(proj_focus) is str:
        proj_focus = load_calibration(proj_focus)
    assert type(proj_focus) is dict

    # Convert aperture diameter (focus_calib convention) to aperture radius (mitsuba convention)
    config["pro_aperture_radius"] = proj_focus["aperture, mm"] / 2. * SCALE_FACTOR
    # Convert values to mitsuba scene units (meters)
    config["pro_focus_distance"] = proj_focus["focus, mm"] * SCALE_FACTOR

    # Diffraction limit on projector resolution
    if diff_limit is None:
        # Camera focus must be already configured prior to automatic diff_limit estimation for the projector
        diff_limit = config["cam_diff_limit"] * config["cam_aperture_radius"] / config["pro_aperture_radius"]

    config["pro_diff_limit"] = diff_limit


def configure_camera(cam_geom="camera/camera_geometry.json", cam_focus="camera/camera_focus.json", calib_path=None, **kw):
    if calib_path is not None:
        cam_geom = os.path.join(calib_path, cam_geom) if type(cam_geom) is str else cam_geom
        cam_focus = os.path.join(calib_path, cam_focus) if type(cam_focus) is str else cam_focus

    config = {}
    configure_camera_geometry(config, cam_geom, **kw)
    configure_camera_focus(config, cam_focus, **kw)

    return config


def configure_projector(config, pro_geom="projector/projector_geometry.json", pro_focus="projector/projector_focus.json", calib_path=None, **kw):
    if calib_path is not None:
        pro_geom = os.path.join(calib_path, pro_geom) if type(pro_geom) is str else pro_geom
        pro_focus = os.path.join(calib_path, pro_focus) if type(pro_focus) is str else pro_focus

    configure_projector_geometry(config, pro_geom, **kw)
    configure_projector_focus(config, pro_focus, **kw)


def configure_camera_and_projector(**kw):
    config = configure_camera(**kw)
    configure_projector(config, **kw)
    return config


def configure_object_geometry(config, object_geom, stage_geom=None, rotation=None, **kw):
    pass


def configure_object_material(config, object_mat, **kw):
    if type(object_mat) is str:
        obj_material = ET.parse(object_mat).getroot()

    config["obj_material"] = obj_material


def configure_object(config, stage_geom="stage/stage_geometry.json", obj_mat="object/rough_plastic_material.xml", calib_path=None, **kw):
    if calib_path is not None:
        stage_geom = os.path.join(calib_path, stage_geom) if type(stage_geom) is str else stage_geom
        obj_mat = os.path.join(calib_path, obj_mat) if type(obj_mat) is str else obj_mat

    configure_object_geometry(config, stage_geom, **kw)
    configure_object_material(config, obj_mat, **kw)


def configure_all(**kw):
    config = configure_camera(**kw)
    configure_projector(config, **kw)
    configure_object(config, **kw)
    return config


def prepare_patterns(config, pattern_path, pro_calib_path, vignetting_img_file,
                     predistort=True, numbered=True, scale4x=True,
                     vignetting=True, response=True, colored=False, flip_up_down=False, overwrite=False):
    pro_calib = load_calibration(pro_calib_path)
    calib_path = os.path.join(pattern_path, "calibrated")
    os.makedirs(calib_path, exist_ok=True)
    
    if numbered:
        pattern_images = sorted(glob.glob(pattern_path + "/*[0-9].png"))
    else:
        pattern_images = sorted(glob.glob(pattern_path + "/*.png"))
    
    render_patterns = []
    
    if vignetting:
        sf = 1
        if scale4x:
            sf = 4
        vignetting_img = Image.open(vignetting_img_file).convert('L')
        vignetting_img = vignetting_img.resize((vignetting_img.size[0]*sf,vignetting_img.size[1]*sf), Image.NEAREST)
        vignetting_img = np.array(vignetting_img)/255.0
        
    for pi in pattern_images:
        assert os.path.isfile(pi)
        stem = pi.split("/")[-1]
        
        tex = Image.open(pi)
        
        # Convert grayscale patterns
        if not colored:
            tex = tex.convert('L')

        assert config["pro_image_width"] == tex.size[0]
        assert config["pro_image_height"] == tex.size[1]
        
        # Upscale to 4x resolution
        if scale4x:
            tex = tex.resize((tex.size[0]*4, tex.size[1]*4), Image.NEAREST)
        
        # Convert to numpy
        tex = np.array(tex)
            
        # Predistort patterns 
        if predistort:
            # TODO account for 4x scaling
            tex = cv2.undistort(tex, pro_calib["mtx"], pro_calib["dist"])
        
        # Apply projector response function
        if response:
            # TODO: apply projector response
            #tex = 0.0003266935427124817 * tex**2 + 0.00013533312472154238 * tex + 0.1251008755294262
            tex = tex / 255
        
        # Apply vignetting to pattern
        if vignetting:
            tex = tex * vignetting_img * 255
            tex = tex.astype("uint8")    
        
        # Flip patterns horizontally
        if flip_up_down:
            tex = np.flipud(tex)
            
        # Write out pattern file    
        if overwrite or not os.path.isfile(calib_path + "/" + stem.replace(".png", "_calib.png")):
            img = Image.fromarray(tex)
            img.save(calib_path + "/" + stem.replace(".png", "_calib.png"))
                
        render_patterns.append(calib_path + "/" + stem.replace(".png", "_calib.png"))

    assert len(render_patterns) == len(pattern_images)
    return render_patterns
