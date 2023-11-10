import os
import glob
import time
import json
import queue
import numpy as np
import matplotlib.pyplot as plt
# from termcolor import colored
# plt.switch_backend('agg')
from scanner.capture.display import *
from scanner.capture import *
from scanner.capture.stage import *


def gen_calibration_script(filename, step=5, stops=10, delay=2.0, color=False, dots=False, right=None, left=None, name=None):
    with open("scripts/" + filename, "w") as f:
        name = name or "calib%s%s%s" % ("_color" if color else "", "_dots" if dots else "", "_sides" if right or left else "")
        f.write("prefix %s/\n" % name)
        f.write("home\n")

        for i in range(stops):
            if i > 0:
                f.write("move %d\n" % step)
                f.write("delay %f\n" % delay)

            f.write("color 0\n")
            f.write("hdr blank_%d\n" % i)

            f.write("color 255\n")
            f.write("hdr white_%d\n" % i)

            if color:
                if dots:
                    f.write("dots\n")
                    f.write("hdr dots_%d\n" % i)
                else:
                    f.write("checker r d\n")
                    f.write("hdr checker_r_d_%d\n" % i)
                    f.write("checker g d\n")
                    f.write("hdr checker_g_d_%d\n" % i)
                    f.write("checker b d\n")
                    f.write("hdr checker_b_d_%d\n" % i)
            else:
                right = right or 0
                left = left or stops
                side = ""
                if i < right:
                    side = " right"
                if i > left:
                    side = " left"

                f.write("checker%s\n" % side)
                f.write("hdr checker_%d\n" % i)

        f.write("status\n")
        # f.write("exit\n")


def gen_default_scan_script(name):
    with open("scripts/%s.script" % name, "w") as f:
        f.write("prefix %s/\n" % name)
        f.write("color 0\n")
        f.write("hdr blank\n")
        f.write("color 255\n")
        f.write("hdr white\n")
        f.write("checker\n")
        f.write("hdr checker\n")
        f.write("color 255 0 0\n")
        f.write("hdr red\n")
        f.write("color 0 255 0\n")
        f.write("hdr green\n")
        f.write("color 0 0 255\n")
        f.write("hdr blue\n")

        for i in range(11):
            f.write("gray %d\n" % i)
            f.write("hdr horizontal_%d\n" % i)
            f.write("gray %d i\n" % i)
            f.write("hdr horizontal_%d_inv\n" % i)

        for i in range(11):
            f.write("gray %d v\n" % i)
            f.write("hdr vertical_%d\n" % i)
            f.write("gray %d v i\n" % i)
            f.write("hdr vertical_%d_inv\n" % i)

        f.write("status\n")
        # f.write("exit\n")


def gen_response_script(name, step=5):
    with open("scripts/%s.script" % name, "w") as f:
        f.write("prefix %s/\n" % name)

        for i in range(0, 256, step):
            f.write("color %d c\n" % i)
            f.write("hdr gray_%d\n" % i)

            f.write("color %d 0 0 c\n" % i)
            f.write("hdr red_%d\n" % i)

            f.write("color 0 %d 0 c\n" % i)
            f.write("hdr green_%d\n" % i)

            f.write("color 0 0 %d c\n" % i)
            f.write("hdr blue_%d\n" % i)

        f.write("status\n")
        # f.write("exit\n")


def gen_patterns_script(name, patterns_folder="unknown"):
    filenames = glob.glob("patterns/" + patterns_folder + "/*.png")
    # print("Found patterns:", filenames)

    with open("scripts/%s.script" % name, "w") as f:
        f.write("suffix %s\n" % name)
        for file in filenames:
            file = file.replace("\\", "/")
            f.write("load %s\n" % file)
            f.write("hdr %s\n" % os.path.basename(file)[:-4])
        f.write("suffix /\n")
        f.write("status\n")


def gen_multiscan_script(name, step=10, stops=10, delay=2.0, subscript=None):
    with open("scripts/%s.script" % name, "w") as f:
        f.write("prefix %s/\n" % name)

        for i in range(stops):
            if i > 0:
                f.write("move %d\n" % step)
                f.write("delay %f\n" % delay)

            prefix = "%s/position_%d/" % (name, i*step)
            f.write("prefix %s\n" % prefix)

            if subscript:
                f.write("subscript %s\n" % subscript)
            else:
                f.write("subscript color\n")
                f.write("subscript gray\n")
                if i == stops-1:
                    f.write("subscript mps_32\n")
                    f.write("subscript ulp_10\n")

            f.write("status\n")


def safe_int(x, default):
    try:
        return int(x)
    except ValueError:
        print((Red + "%s is not an integer value. Using default value %d instead" + Reset) % (x, default))
        return default


def safe_float(x, default):
    try:
        return float(x)
    except ValueError:
        print((Red + "%s cannot be converted to float. Using default value %f instead" + Reset) % (x, default))
        return default


if __name__ == "__main__":
    # gen_calibration_script("calib.script", step=100, stops=4, delay=2.0, color=False, dots=False)
    # gen_calibration_script("calib.script", step=10, stops=33, delay=2.0, color=False, dots=False)
    gen_calibration_script("calib.script", step=5, stops=65, color=False, dots=False)
    gen_calibration_script("projector_calib.script", step=5, stops=65, color=False, dots=False, right=22, left=50, name="projector_calib")
    gen_calibration_script("calib_color.script", step=5, stops=65, color=True, dots=False)
    gen_calibration_script("calib_color_dots.script", step=5, stops=65, color=True, dots=True)

    gen_response_script("response", step=5)

    # gen_patterns_script("color", patterns_folder="color")
    # gen_patterns_script("gray", patterns_folder="gray")
    # gen_patterns_script("mps_16", patterns_folder="mps/16-15")
    # gen_patterns_script("mps_32", patterns_folder="mps/32-08")
    # gen_patterns_script("mps_64", patterns_folder="mps/64-05")
    # gen_patterns_script("ulp_5", patterns_folder="ulp/0.05_40")
    # gen_patterns_script("ulp_10", patterns_folder="ulp/0.10_40")
    # gen_patterns_script("ulp_15", patterns_folder="ulp/0.15_46")

    gen_default_scan_script("default_scan")

    step = 45
    gen_multiscan_script("default_multiscan", step=step, stops=360//step, delay=2.0, subscript="default_scan")

    step = 30
    for name in ["complete_multiscan", "pawn", "dodo", "shapes", "rook"]:
        gen_multiscan_script(name, step=step, stops=360//step)

    gen_multiscan_script("stage_calib", step=5, stops=23, subscript="checker_center")
    gen_multiscan_script("stage_calib_dense", step=2, stops=76, subscript="checker_center")
    gen_multiscan_script("stage_calib_dense_visible", step=2, stops=56, subscript="checker_center")
    # gen_multiscan_script("material_calib", step=2, stops=76, subscript="material_scan")
    # exit()

    stage, camera = None, None

    stage = RotatingStage(port="COM3")
    # stage = LinearStage(port="COM4")
    # stage.home(speed_divider=12)  # 1 in/sec translation speed

    camera = Camera()
    camera.open()
    camera.init(None, "Mono12")

    supported_commands = ["blank", "stripes", "patterns", "dots", "checker", "color", "gray", "plot",
                          "move", "home", "load", "save", "ldr", "hdr", "ldr_count", "hdr_count", "skip", "exposures",
                          "prefix", "suffix", "delay", "script", "subscript", "dump", "status", "exit"]
    hdr_exposures = default_exposures
    hdr_exposures = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.0167, 0.025, 0.0333, 0.05, 0.1,
                     0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0]
    # hdr_exposures = [0.0167, 0.0333, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    # hdr_exposures = [0.0167, 0.05, 0.1, 0.25, 0.75, 1.5]
    # hdr_exposures = [0.0167, 0.0333, 0.05, 0.1, 0.25, 0.75]
    hdr_exposures = [0.0167, 0.0333, 0.05, 0.1, 0.25, 0.75, 1.5]
    ldr_exposure = 0.5

    ldr, hdr, ldr_name, hdr_name = None, None, None, None
    ldr_count, hdr_count = 0, 0
    data_path = "D:/scanner_sim/captures/stage_batch_3/"
    # data_path = "D:/scanner_sim/calibration/accuracy_test/"
    # data_path = "D:/scanner_sim/calibration/projector_response/"
    prefix = "default/"
    suffix = "/"

    try:
        projector = Projector()
        if camera:
            camera.start_stream()

        print("Supported commands:", supported_commands)
        print("Supported exposures:", default_exposures, "\n")

        input_queue, thr = parallel_input()
        history = []
        timestamp = time.time()
        delay = 0

        input_queue.put("checker")
        input_queue.put("status")
        # input_queue.put("scan")
        # input_queue.put("script scan")

        while not projector.should_close():
            if hdr:
                if hdr.done():
                    hdr_name = hdr_name or str(hdr_count)
                    print("Captured HDR \"%s\"" % (prefix + suffix + hdr_name), hdr.result().shape)
                    # camera.plot_timeline()
                    # camera.plot_hdr(save_preview=False)
                    # plt.pause(0.001)

                    if not os.path.exists(data_path + prefix):
                        os.makedirs(data_path + prefix, exist_ok=True)
                    if not os.path.exists(data_path + prefix + suffix):
                        os.makedirs(data_path + prefix + suffix, exist_ok=True)

                    save_openexr(data_path + prefix + suffix + hdr_name + ".exr", hdr.result())
                    hdr_count += 1
                    hdr = None
                else:
                    # camera.plot_timeline()
                    # plt.pause(0.001)
                    pass

            if ldr:
                if ldr.done():
                    ldr_name = ldr_name or str(ldr_count)
                    print("Captured LDR \"%s\"" % (prefix + suffix + ldr_name), ldr.result()[1].shape)
                    # print("Captured LDR:", ldr.result()[1].shape)
                    camera.plot_ldr(save_preview=False)
                    plt.pause(0.001)

                    if not os.path.exists(data_path + prefix):
                        os.makedirs(data_path + prefix, exist_ok=True)
                    if not os.path.exists(data_path + prefix + suffix):
                        os.makedirs(data_path + prefix + suffix, exist_ok=True)

                    np.save(data_path + prefix + suffix + ldr_name, ldr.result()[1])
                    ldr_count += 1
                    ldr = None

            new_pattern = None

            if not input_queue.empty() and not hdr and not ldr:
                cmd = input_queue.get()
                history.append(cmd)
                print("Got:", cmd)
                cmd = cmd.split(" ")
                cmd, p = cmd[0], cmd[1:]

                if cmd not in supported_commands:
                    print(Red + "Unrecognized command:", cmd, Reset)
                    continue

                if cmd == 'plot':
                    plt.figure("Pattern", (12, 7), clear=True)
                    plt.imshow(projector.get_pattern())
                    plt.title("Pattern")
                    plt.tight_layout()
                    plt.pause(0.001)

                if cmd == 'load' and len(p) > 0:
                    new_pattern = imageio.imread(p[0])
                    if len(new_pattern.shape) == 2:
                        new_pattern = np.repeat(new_pattern[:, :, None], 3, axis=2)

                if cmd == 'save' and len(p) > 0:
                    imageio.imwrite(p[0], projector.get_pattern())

                if cmd == 'patterns' and len(p) > 0:
                    folder = p[0]

                    filenames = glob.glob("patterns/" + folder + "/*.png")

                    if len(filenames) > 0:
                        print("Found patterns:", filenames)
                        input_queue.put("suffix " + folder)
                        for file in filenames:
                            input_queue.put("load " + file)
                            input_queue.put("hdr " + os.path.basename(file)[:-4])
                        input_queue.put("suffix /")
                    else:
                        print(Magenta + "No pattern found in " + folder + Reset)

                if cmd == 'move' and len(p) > 0:
                    if stage:
                        dist = safe_int(p[0], 0)
                        if dist:
                            stage.move(dist)
                        else:
                            print("Invalid distance")
                    else:
                        print("No stage available")

                if cmd == 'home':
                    if stage:
                        stage.home()
                    else:
                        print("No stage available")

                if cmd == "color":
                    if len(p) > 0:
                        if len(p) >= 1:
                            r = g = b = safe_int(p[0], 0)
                        if len(p) >= 3:
                            r, g, b = safe_int(p[0], 0), safe_int(p[1], 0), safe_int(p[2], 0)
                        full_pattern = gen_color((H, W), (r, g, b))
                        new_pattern = full_pattern
                        if len(p) == 2 or len(p) == 4:
                            if p[-1] == "c":
                                new_pattern = np.zeros_like(full_pattern)
                                r, c, s = 540, 960, 100
                                new_pattern[r-s:r+s, c-s:c+s, :] = full_pattern[r-s:r+s, c-s:c+s, :]
                    else:
                        print("Specify the color!")

                if cmd == "gray":
                    if len(p) > 0:
                        i = safe_int(p[0], 0)
                        R, C = gen_gray((H, W), color=[255, 255, 255], invert=("i" in p))
                        if "v" in p:
                            new_pattern = C[i, :, :, :]
                        else:
                            new_pattern = R[i, :, :, :]

                if cmd == "stripes":
                    axis = 1 if 'v' in p else 0
                    stride = safe_int(p[0], 10) if len(p) > 0 else 10
                    new_pattern = gen_stripes((H, W), stride, axis)

                if cmd == 'dots':
                    new_pattern = gen_dots((H, W), (90 + 50, 60 + 50), (100, 100), (9 - 1, 18 - 1))

                if cmd == "checker":
                    new_pattern = gen_checker((H, W), (90, 60), 100, (9, 18))

                    if "right" in p:
                        new_pattern[:, :W//2-100, :] = 0
                        new_pattern[:, W//2-160:W//2-100, :] = 255
                    if "left" in p:
                        new_pattern[:, W//2+100:, :] = 0
                        new_pattern[:, W//2+100:W//2+160, :] = 255
                    if "center" in p:
                        l, r = W//2-500, W//2 + 500
                        new_pattern[:, :l, :] = 0
                        new_pattern[:, l-60:l, :] = 255
                        new_pattern[:, r:, :] = 0
                        new_pattern[:, r:r+60, :] = 255
                        if "s" in p:
                            t = 90 + 200
                            new_pattern[:t, :, :] = 0
                            new_pattern[t-60:t, l-60:r+60, :] = 255
                    if "d" in p:
                        for i in range(9):
                            for j in range(18):
                                new_pattern[90+50 + 100*i, 60+50 + 100*j, :] = 255
                    if "r" in p:
                        new_pattern *= np.array([1, 0, 0], dtype=np.uint8)
                    if "g" in p:
                        new_pattern *= np.array([0, 1, 0], dtype=np.uint8)
                    if "b" in p:
                        new_pattern *= np.array([0, 0, 1], dtype=np.uint8)

                    new_pattern[:30, :, :] = 0
                    new_pattern[H-30:, :, :] = 0

                if cmd == "ldr":
                    if len(p) > 0:
                        ldr_exposure = round(safe_float(p[0], ldr_exposure), ndigits=6)
                        print("LDR exposure:", ldr_exposure)
                    if len(p) > 1:
                        ldr_name = p[1]
                    else:
                        ldr_name = None
                    ldr = camera.capture_async(ldr_exposure)

                if cmd == "hdr":
                    # if len(p) > 0:
                    #     hdr_count = safe_int(p[0], 0)
                    if len(p) > 0:
                        hdr_name = p[0]
                    else:
                        hdr_name = None

                    camera.time_zero()
                    hdr = camera.capture_async(hdr_exposures, dark_path="dark_frames/", gamma=default_gamma, plot=False)

                if cmd == "ldr_count":
                    if len(p) > 0:
                        ldr_count = safe_int(p[0], 0)
                        print("ldr_count =", ldr_count)

                if cmd == "hdr_count":
                    if len(p) > 0:
                        hdr_count = safe_int(p[0], 0)
                        print("hdr_count =", hdr_count)

                if cmd == "skip":
                    hdr_count -= 1

                if cmd == "exposures":
                    exposures = []
                    for pi in p:
                        exp = round(safe_float(pi, 0), ndigits=6)
                        if str(exp) not in [str(exp) for exp in default_exposures]:
                            print(Yellow + "Unsupported exposure:", pi, Reset)
                        else:
                            exposures.append(exp)
                    if len(exposures) > 2:
                        hdr_exposures = exposures
                    else:
                        print(Magenta + "At least two valid exposures needed for HDR", Reset)

                    print("HDR exposures:", hdr_exposures)

                if cmd == 'prefix':
                    if len(p) < 1:
                        print("Define prefix")
                        continue
                    prefix = p[0]
                    if prefix[-1] != "/":
                        prefix += "/"
                    print("Save to:", prefix)

                if cmd == 'suffix':
                    if len(p) < 1:
                        print("Define suffix")
                        continue
                    suffix = p[0]
                    if suffix[-1] != "/":
                        suffix += "/"
                    print("Save to (subfolder):", suffix)

                if cmd == 'subscript':
                    print("Cannot invoke subscript command directly")
                    continue

                if cmd == 'script':
                    if len(p) < 1:
                        print("Define script name")
                        continue

                    script = "scripts/" + p[0]

                    if not os.path.exists(script + ".script"):
                        print(Magenta + "File " + script + ".script does not exist!" + Reset)
                        continue

                    print("Loading script: \"%s.script\"" % (script))

                    with open(script + ".script", "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            if len(line) < 1 or line[0] == "#":
                                continue
                            if "prefix" in line and "ignore_prefix" in p:
                                continue
                            if "subscript" in line:
                                parts = line.strip().split()
                                if len(parts) < 2:
                                    print(Magenta + "No subscript name!" + Reset)
                                    continue
                                with open("scripts/" + parts[1] + ".script", "r") as f2:
                                    extra_lines = [l for l in f2.readlines() if "prefix" not in l]
                                    for l in extra_lines:
                                        input_queue.put(l[:-1])
                                continue
                            input_queue.put(line[:-1])

                if cmd == 'delay':
                    if len(p) < 1:
                        print("Set delay in seconds")
                        continue
                    delay = safe_float(p[0], 0)
                    time.sleep(delay)
                    # timestamp = time.time()

                if cmd == 'dump':
                    if len(p) < 1:
                        print("Define filename")
                        continue
                    with open(p[0], "w") as f:
                        f.write("\n".join(history))

                if cmd == 'status':
                    print("\nSave to:", data_path + prefix + suffix)
                    print("\tLDR exposure:", ldr_exposure)
                    print("\tldr_count =", ldr_count)
                    print("\tHDR exposures:", hdr_exposures)
                    print("\thdr_count =", hdr_count)

                if cmd == 'exit':
                    break

            projector.update(new_pattern)

            for i in range(40):
                if plt.get_fignums():
                    plt.gcf().canvas.start_event_loop(0.001)
                else:
                    time.sleep(0.001)
    finally:
        if stage:
            stage.close()
        if camera:
            camera.stop_stream()
            camera.close()
        glfw.terminate()

    print('Done')

    running = False
    print('Press Enter to exit...')
    thr.join()
