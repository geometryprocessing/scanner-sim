import os
import glob
import subprocess
import lxml.etree as ET
from utils import string2transform, transform2string

def write_scene_file(config, scene_filename, template_filename, warn_missing_config=False):
    template = ET.parse(template_filename)
    valid = True
    for e in template.findall("default"):
        n = e.get("name")
        if n in config:
            e.set("value", str(config[n]))
        else:
            if warn_missing_config:
                print("Configuration value missing: %s. Using default: %s"%(n, e.get("value")))
    #print(len(template.findall("default")))
    # Object material settings need to be filled in directly (if present)
    if "obj_material" in config:
        om = template.xpath("shape")
        assert len(om) == 1
        # print(om, dir(om), config["obj_material"])
        om[0].insert(3, config["obj_material"])
        ET.indent(om[0], space="    ", level=1)
    # ET.dump(template)
    template.write(scene_filename)
    

def write_scene_files(config, patterns, rotations, results_path, scene_path):
    org_transform = string2transform(config["obj_transform"])
    for rot_i, rotation in enumerate(rotations):
        #obj_transform = rotation @ org_transform
        #print(obj_transform)
        print("O", org_transform)
        print("R", rotation)
        config["rot_transform"] = transform2string(rotation)
        r_path = os.path.join(results_path, "rotation_%03i"%rot_i)
        os.makedirs(r_path, exist_ok=True)
        for cnt, pattern in enumerate(patterns):
            config["pro_pattern_file"] = pattern
            p_path = os.path.join(r_path, "pattern_%03i.xml"%cnt)
            write_scene_file(config, p_path, scene_path)
    

def source(script, update=True):
    pipe = subprocess.Popen("source %s; env" % script, stdout=subprocess.PIPE,
                            shell=True, executable="/bin/bash")
    data = pipe.communicate()[0].decode("utf-8")

    env = dict([line.split("=", 1) for line in data.splitlines()])

    if update:
        print("Updating environment:", env)
        os.environ.update(env)

    return env


def render_scenes(filenames_template, output_folder=None, verbose=True):
    scenes = glob.glob(filenames_template)
    print(scenes)
    if output_folder is not None:
        output = ["-o", output_folder]
    else:
        output = []

    for i, scene in enumerate(scenes):
        print("\nScene %d of %d:" % (i+1, len(scenes)), scene)

        proc = subprocess.Popen(["mitsuba", *output, scene], stdout=subprocess.PIPE, universal_newlines=True)

        for line in proc.stdout:
            if verbose:
                print(line[:-1])
