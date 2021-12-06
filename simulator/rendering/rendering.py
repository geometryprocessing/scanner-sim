

def load_template(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    header, body = [], []
    for i, line in enumerate(lines):
        if not line.startswith("<scene"):
            header.append(line)
        else:
            header.append(line)
            body = lines[i+1:]
            break

    return header, body


def generate_scene(header, body, config, filename):
    with open(filename, "w") as f:
        f.writelines(header)

        for key, value in config.items():
            f.write("\t<default name=\"%s\" value=\"%s\"/>\n" % (key, str(value)))

        f.write("\n")
        f.writelines(body)


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
