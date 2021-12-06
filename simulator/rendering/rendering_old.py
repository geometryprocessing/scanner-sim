import mitsuba
import numpy as np
import os
import multiprocessing
import json
import cv2
import glob
from PIL import Image
from mitsuba.core import * 
from mitsuba.render import Scene, Texture
from mitsuba.render import RenderQueue,RenderJob
from mitsuba.render import SceneHandler
import argparse
from tqdm.auto import tqdm
from utils import load_projector_calibration


# Render one configuration (one object, one patternset, multiple rotations)
def render_object(config_path):
    assert os.path.isfile(config_path)
    
    with open(config_path, "r") as fi:
        parameters = json.load(fi)
    
    patterns = prepare_patterns(parameters)
    rotations = prepare_rotations(parameters)
    translations = prepare_translations(parameters)
    #print(len(patterns), len(rotations), rotations)
    
    render_all_patterns_and_rotations(parameters, patterns, rotations, translations)


# Prepare the list of rotations for rendering
def prepare_rotations(parameters):
    if parameters["rot_type"] == "Turntable":
        rotations = [] 
        rots = np.linspace(parameters["rot_range"][0], parameters["rot_range"][1], parameters["rot_range"][2], endpoint=False)
        for i in range(len(rots)):
            rotations.append({"name": "rot_%03i"%i, "rot_x": parameters["rot_x"], "rot_y": rots[i], "rot_z": parameters["rot_z"]})
    elif parameters["rot_type"] == "Random":
        rotations = []
        for i in range(parameters["rot_range"][3]):
            rot_x = np.random.randint(parameters["rot_range"][0][0], parameters["rot_range"][0][1])
            rot_y = np.random.randint(parameters["rot_range"][1][0], parameters["rot_range"][1][1])
            rot_z = np.random.randint(parameters["rot_range"][2][0], parameters["rot_range"][2][1])
            rotations.append({"name": "rot_%03i"%i, "rot_x": rot_x, "rot_y": rot_y, "rot_z": rot_z})
    elif parameters["rot_type"] == "Fixed":
        rotations = [{"name": "rot_fixed", "rot_x": parameters["rot_x"], "rot_y": parameters["rot_y"], "rot_z": parameters["rot_z"]}]
    return rotations

# Prepare the list of translations for rendering
def prepare_translations(parameters):
    if parameters["trans_type"] == "Random":
        translations = []
        for i in range(parameters["trans_range"][3]):
            trans_x = np.random.rand() * (parameters["trans_range"][0][1] - parameters["trans_range"][0][0]) + parameters["trans_range"][0][0]
            trans_y = np.random.rand() * (parameters["trans_range"][1][1] - parameters["trans_range"][1][0]) + parameters["trans_range"][1][0]
            trans_z = np.random.rand() * (parameters["trans_range"][2][1] - parameters["trans_range"][2][0]) + parameters["trans_range"][2][0]
            translations.append({"name": "trans_%03i"%i, "trans_x": trans_x, "trans_y": trans_y, "trans_z": trans_z})
    elif parameters["trans_type"] == "Fixed":
        translations = [{"name": "trans_fixed", "trans_x": parameters["trans_x"], "trans_y": parameters["trans_y"], "trans_z": parameters["trans_z"]}]
    return translations

# Render all patterns and rotations with the given parameters
def render_all_patterns_and_rotations(parameters, patterns, rotations, translations):
    scheduler = Scheduler.getInstance()
    scheduler.stop()
    pars = parameters

    if pars["cpu_count"] == -1:
        pars["cpu_count"] = multiprocessing.cpu_count()

    for i in range(0, pars["cpu_count"]):
        scheduler.registerWorker(LocalWorker(i,'wrk%i'%i))

    scheduler.start()
    queue = RenderQueue()
    
    # Iterate over all rotations
    total_renderings = len(rotations) * len(translations) * (len(patterns) + 1)
    pbar = tqdm(total=total_renderings)
    for r_cnt, rotation in enumerate(rotations):
        if len(rotations) > 1:
            rot_path = os.path.join(pars["root_path"], pars["result_path"], rotation["name"])
        else: 
            rot_path = os.path.join(pars["root_path"], pars["result_path"])
        os.makedirs(rot_path, exist_ok=parameters["result_overwrite"])
        
        # Iterate over all translations
        for t_cnt, translation in enumerate(translations):
            if len(translations) > 1:
                res_path = os.path.join(rot_path, translation["name"])
            else:
                res_path = rot_path
            os.makedirs(res_path, exist_ok=parameters["result_overwrite"])

            # Iterate over all patterns
            for p_cnt, pattern in enumerate(patterns):
                param_map = map_render_parameters(pars, rotation, pattern, translation)
                param_map["result_name"] = res_path + "/img_%03i"%p_cnt
                
                p_dict = {}
                for k in param_map:
                    sk = str(k)
                    key = sk.split(", ")[0][1:]
                    tm = sk.split(", ")
                    if len(tm) == 2:
                        data = sk.split(", ")[1][:-1]
                    else: 
                        data = ",".join(sk.split(", ")[1:])[:-1]
                    #print(dir(k), key, data)
                    p_dict[key] = data
                    
                with open(res_path+"/pars.json", "w") as fi:
                    json.dump(p_dict, fi, indent=2, sort_keys=True)

                img = render_scene(scheduler, queue, pars, param_map)
                img = img[pars["cam_crop_offset_y"]:pars["cam_crop_offset_y"]+pars["cam_crop_height"], pars["cam_crop_offset_x"]:pars["cam_crop_offset_x"]+pars["cam_crop_width"]]
                if not parameters["pattern_colored"]:
                    img = rgb2gray(img)

                #print(img.dtype, img.shape)
                img = Image.fromarray(img.astype(np.uint8))
                img.save(param_map["result_name"] + ".png")
                
                pbar.update(1)

            # Render with ambient illumination
            if pars["render_ambient"] and len(patterns) > 1:
                param_map = map_render_parameters(pars, rotation, patterns[1], translation) # TODO proper selection of black pattern
                param_map["result_name"] = res_path + "/img_amb"
                param_map["const_radiance"] = "0.25"

                img = render_scene(scheduler, queue, pars, param_map)
                img = img[pars["cam_crop_offset_y"]:pars["cam_crop_offset_y"]+pars["cam_crop_height"], pars["cam_crop_offset_x"]:pars["cam_crop_offset_x"]+pars["cam_crop_width"]]
                if not parameters["pattern_colored"]:
                    img = rgb2gray(img)

                img = Image.fromarray(img.astype(np.uint8))
                img.save(param_map["result_name"] + ".png")
                
                pbar.update(1)
    pbar.close()

# Map the render configuration to mitsuba friendly string map
def map_render_parameters(pars, rotation, pattern, translation):
    # TODO cleanup
    paramMap = StringMap()
    paramMap['pattern_name'] = pattern
    paramMap['object_name'] = os.path.join(pars["root_path"], pars["object_path"])
    
    for p in pars:
        if p == "focal_length":
            paramMap[p] = str(pars[p]) + "mm"
        if p == "obj_rx" or p == "obj_ry" or p == "obj_rz":
            a = np.array([[0, pars["obj_rx"], pars["obj_ry"]],
                          [-pars["obj_rx"], 0, pars["obj_rz"]],
                          [-pars["obj_ry"], -pars["obj_rz"], 0]])
            cayley = (np.eye(3) + a).dot(np.linalg.inv(np.eye(3) - a))
            t = np.vstack([cayley, np.zeros(3)])
            x = np.array([[0, 0, 0, 1]])
            trans = np.hstack([t, x.T]).reshape(-1)
            #print(trans)
            tstr = ""
            for t in trans:
                tstr += "%0.6f "%t
            paramMap["obj_transform2"] = tstr
        if p == "proj_rx" or p == "proj_ry" or p == "proj_rz":
            a = np.array([[0, pars["proj_rx"], pars["proj_ry"]],
                          [-pars["proj_rx"], 0, pars["proj_rz"]],
                          [-pars["proj_ry"], -pars["proj_rz"], 0]])
            cayley = (np.eye(3) + a).dot(np.linalg.inv(np.eye(3) - a))
            t = np.vstack([cayley, np.zeros(3)])
            x = np.array([[0, 0, 0, 1]])
            trans = np.hstack([t, x.T]).reshape(-1)
            #print(trans)
            tstr = ""
            for t in trans:
                tstr += "%0.6f "%t
            paramMap["proj_transform2"] = tstr
        if p == "obj_trans":
            tstr = ""
            trans = pars[p]
            for t in trans:
                tstr += "%0.6f "%t
            paramMap["obj_transform"] = tstr
        if p == "proj_trans":
            tstr = ""
            trans = pars[p]
            for t in trans:
                tstr += "%0.6f "%t
            paramMap["proj_transform"] = tstr
        paramMap[p] = str(pars[p])
        
    paramMap['rot_x'] = str(rotation["rot_x"])
    paramMap['rot_y'] = str(rotation["rot_y"])
    paramMap['rot_z'] = str(rotation["rot_z"])
    paramMap['trans_x'] = str(translation["trans_x"])
    paramMap['trans_y'] = str(translation["trans_y"])
    paramMap['trans_z'] = str(translation["trans_z"])

    return paramMap


# Render one specific scene
def render_scene(scheduler, queue, pars, param_map):
    fileResolver = Thread.getThread().getFileResolver()
    fileResolver.appendPath('rdata')

    scene_path = os.path.join(pars["root_path"], pars["scene_path"])
    scene = SceneHandler.loadScene(fileResolver.resolve(scene_path), param_map)
    scene.setDestinationFile(param_map["result_name"])
    job=RenderJob('myRenderJob', scene, queue)
    job.start()

    queue.waitLeft(0)
    film=scene.getFilm()
    size=film.getSize()
    bitmap=Bitmap(Bitmap.ERGB, Bitmap.EUInt8, size)
    film.develop(Point2i(0,0), size, Point2i(0,0), bitmap)
    res = np.array(bitmap.buffer())
    return res

# Color to gray conversion
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# Prepare the patterns for rendering
def prepare_patterns(parameters):
    pattern_path = os.path.join(parameters["root_path"], parameters["pattern_path"])
    calib_path = os.path.join(parameters["root_path"], parameters["pattern_path"], "calibrated")
    if parameters["pattern_numbered"]:
        pattern_images = sorted(glob.glob(pattern_path + "/*[0-9].png"))
    else:
        pattern_images = sorted(glob.glob(pattern_path + "/*.png"))
    
    os.makedirs(calib_path, exist_ok=True)
    
    render_patterns = []
    
    proj_calib = load_projector_calibration("data/calibrations/projector_calibration_new.json")[2]
    
    for pi in pattern_images:
        assert os.path.isfile(pi)
        stem = pi.split("/")[-1]
        
        tex = Image.open(pi)
        if not parameters["pattern_colored"]:
            tex = tex.convert('L')
        
        tex = np.array(tex)
        tex_height, tex_width = tex.shape[0], tex.shape[1]
        if len(tex.shape) == 3:
            channels = tex.shape[2]
        else:
            channels = 1
        
        # TODO allow for larger patterns as well
        assert tex_height == parameters["pattern_height"] and tex_width == parameters["pattern_width"]
        
        if parameters["pattern_calibrate"]:
            # Predistort patterns
            tex = cv2.undistort(tex, proj_calib["mtx"], proj_calib["dist"], newCameraMatrix=proj_calib["new_mtx"])
            
        if parameters["pattern_flip_ud"]:
            tex = np.flipud(tex)
            
        if parameters["pattern_quadrify"]:
#             if tex_width > tex_height:
#                 if channels > 1:
#                     z = np.zeros((tex_width-tex_height, tex_width, channels), dtype="uint8")
#                 else:
#                     z = np.zeros((tex_width-tex_height, tex_width), dtype="uint8")                    
#                 if parameters["pattern_pad_below"]:
#                     tex = np.concatenate((tex, z), axis=0)
#                 else:
#                     tex = np.concatenate((z, tex), axis=0)
            z = np.zeros((tex_height, 138), dtype="uint8") # TODO fix magic numbers
            z2 = np.zeros((tex_height, 134), dtype="uint8")
            tex = np.concatenate((z, tex, z2), axis=1)
            z = np.zeros((1112, 2192), dtype="uint8")
            tex = np.concatenate((tex, z), axis=0)

        if parameters["pattern_overwrite"] or not os.path.isfile(calib_path + "/" + stem.replace(".png", "_calib.png")):
            tex = Image.fromarray(tex)
            tex.save(calib_path + "/" + stem.replace(".png", "_calib.png"))
                
        render_patterns.append(calib_path + "/" + stem.replace(".png", "_calib.png"))

    assert len(render_patterns) == len(pattern_images)
    return render_patterns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate SL scanning.')
    parser.add_argument('-c', '--config', dest='config', type=str, default="data/configs/parameters.json", help='Path to the rendering configuration.')
    a = parser.parse_args()
    
    render_object(config_path=a.config)
