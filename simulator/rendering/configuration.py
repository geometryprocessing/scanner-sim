from utils import *
import re
import cv2
from PIL import Image

SCALE_FACTOR = 1 / 1000.0 # configuration is [mm], mitsuba uses [m]


def configure_camera_geometry(config, cam_geom):
    if type(cam_geom) is str:
        cam_geom = load_calibration(cam_geom)
    assert type(cam_geom) is dict

    w, h = cam_geom["image_width, pixels"], cam_geom["image_height, pixels"]
    config["cam_image_width"], config["cam_image_height"] = w, h
    config["cam_fov_y"] = 2 * np.arctan((h / 2.) / cam_geom["new_mtx"][1, 1]) * 180. / np.pi
    config["cam_pixel_aspect"] = cam_geom["new_mtx"][0, 0] / cam_geom["new_mtx"][1, 1]
    # TODO: make use of cam_geom["new_mtx"][:2, 2] and cropping feature in mitsuba to simulate optical axis offset
    
    return cam_geom


def configure_camera_focus(config, cam_focus, tolerances_factor=1.07, wavelength_nm=550):
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


def configure_projector_geometry(config, proj_geom, brightness=10.0, gap_size=0.0):
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
    config["pro_gap_size"] = gap_size
    
    # Projector extrinsics
    transform = np.zeros((4,4))
    transform[:3, 3] = proj_geom["origin"] * SCALE_FACTOR
    transform[:3, :3] = proj_geom["basis"]
    transform[3, 3] = 1.0
    config["pro_transform"] = transform2string(transform)
    
    # Projector pattern settings
    config["pro_offset_x"], config["pro_offset_y"] = w/2, h
    
    return proj_geom

def transform2string(transform):   
    transform = transform.T.flatten()
    transform = np.array2string(transform.T.flatten(), prefix="", suffix="", max_line_width=1000)
    transform = transform.replace("[ ", "").replace(" ]", "")
    transform = re.sub(' +', ' ', transform)
    return transform

def configure_projector_focus(config, proj_focus, diffLimit=0):
    if type(proj_focus) is str:
        proj_focus = load_calibration(proj_focus)
    assert type(proj_focus) is dict

    # Convert values to mitsuba scene units (meters)
    config["pro_focus_distance"] = proj_focus["focus, mm"] * SCALE_FACTOR
    # Convert aperture diameter (focus_calib convention) to aperture radius (mitsuba convention)
    config["pro_aperture_radius"] = proj_focus["aperture, mm"] / 2. * SCALE_FACTOR
    # Diffraction limit on projector resolution
    config["pro_diff_limit"] = diffLimit
    

def configure_projector_and_camera(data_path, cam_geo_file="camera_geometry.json", cam_focus_file="camera_focus.json", pro_geo_file="projector_geometry.json", pro_focus_file="projector_focus.json"):
    config = {}
    cam_geom = configure_camera_geometry(config, os.path.join(data_path, "calibrations", cam_geo_file))
    configure_camera_focus(config, os.path.join(data_path, "calibrations", cam_focus_file))    
    pro_geom = configure_projector_geometry(config, os.path.join(data_path, "calibrations", pro_geo_file))
    configure_projector_focus(config, os.path.join(data_path, "calibrations", pro_focus_file))
    
    # TODO: Is this needed?
    config["pro_diff_limit"] = config["cam_diff_limit"] * config["cam_aperture_radius"] / config["pro_aperture_radius"]
    return config, cam_geom, pro_geom



def configure_object_geometry(config, stage_geom):
    pass


def process_images(config, cam_geom, cam_vignetting, wb):
    pass


def generate_patterns(config, pattern_path, pro_calib, vignetting_img_file, predistort=True, numbered=True, scale4x=True, vignetting=True, response=True, colored=False, flip_up_down=False, overwrite=False):
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
