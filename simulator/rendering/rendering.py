import xml.etree.ElementTree
import os
import glob
import subprocess


def write_scene_file(config, scene_filename, template_filename):
    template = xml.etree.ElementTree.parse(template_filename)
    valid = True
    for e in template.findall("default"):
        n = e.get("name")
        if n in config:
            e.set("value", str(config[n]))
        
        if e.get("value") == "N/A":
            print("Configuration value missing for parameter: %s."%n)
            valid = False
    
    if not valid:
        print("Missing parameters, invalid config is not written to file.")
        return

    template.write(scene_filename)
    

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
    if output_folder is not None:
        output = ["-o", output_folder]
    else:
        output = []

    for i, scene in enumerate(sorted(scenes)):
        print("\nScene %d of %d:" % (i+1, len(scenes)), scene)

        proc = subprocess.Popen(["mitsuba", *output, scene], stdout=subprocess.PIPE, universal_newlines=True)

        for line in proc.stdout:
            if verbose:
                print(line[:-1])
