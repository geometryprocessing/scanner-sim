import os
import time
import json
import queue
import numpy as np
import matplotlib.pyplot as plt
# from termcolor import colored
from display import *
from capture import *


def gen_script(filename):
    with open(filename, "w") as f:
        f.write("prefix shapes/\n")
        f.write("color 0\n")
        f.write("hdr 0\n")
        f.write("color 255\n")
        f.write("hdr 1\n")

        for i in range(11):
            f.write("gray %d\n" % i)
            f.write("hdr %d\n" % (100 + i))
            f.write("gray %d i\n" % i)
            f.write("hdr %d\n" % (200 + i))

        for i in range(11):
            f.write("gray %d v\n" % i)
            f.write("hdr %d\n" % (300 + i))
            f.write("gray %d v i\n" % i)
            f.write("hdr %d\n" % (400 + i))
        # f.write("exit\n")


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
    camera = None

    camera = Camera()
    camera.open()
    camera.init(None, "Mono12")

    supported_commands = ["blank", "stripes", "checker", "color", "gray", "plot", "load", "save", "ldr", "hdr", "ldr_count", "hdr_count", "skip", "exposures", "prefix", "delay", "script", "dump", "status", "exit"]
    hdr_exposures = default_exposures
    hdr_exposures = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75,
                     1.0, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0]
    # hdr_exposures = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    ldr_exposure = 1

    ldr, hdr = None, None
    ldr_count, hdr_count = 0, 0
    prefix = "scan/"

    # gen_script("scan.script")

    try:
        projector = Projector()
        if camera:
            camera.start_stream()

        print("Supported commands:", supported_commands)
        print("Supported exposures:", default_exposures, "\n")

        input_queue = parallel_input()
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
                    print("Captured HDR:", hdr.result().shape)
                    camera.plot_timeline()
                    camera.plot_hdr(save_preview=False)
                    plt.pause(0.001)
                    if not os.path.exists(prefix):
                        os.makedirs(prefix, exist_ok=True)
                    save_openexr(prefix + str(hdr_count) + ".exr", hdr.result())
                    hdr_count += 1
                    hdr = None
                else:
                    camera.plot_timeline()
                    plt.pause(0.001)

            if ldr:
                if ldr.done():
                    print("Captured LDR:", ldr.result()[1].shape)
                    camera.plot_ldr(save_preview=False)
                    plt.pause(0.001)
                    if not os.path.exists(prefix):
                        os.makedirs(prefix, exist_ok=True)
                    np.save(prefix + str(ldr_count), ldr.result()[1])
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

                if cmd == 'save' and len(p) > 0:
                    imageio.imwrite(p[0], projector.get_pattern())

                if cmd == "color":
                    if len(p) > 0:
                        c = safe_int(p[0], 0)
                        new_pattern = gen_color((H, W), (c, c, c))

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

                if cmd == "checker":
                    new_pattern = gen_checker((H, W), (90, 60), 100, (9, 18))

                    if "right" in p:
                        new_pattern[:, :W//2-100, :] = 0
                        new_pattern[:, W//2-160:W//2-100, :] = 255
                    if "left" in p:
                        new_pattern[:, W//2+100:, :] = 0
                        new_pattern[:, W//2+100:W//2+160, :] = 255

                    new_pattern[:30, :, :] = 0
                    new_pattern[H-30:, :, :] = 0

                if cmd == "ldr":
                    if len(p) > 0:
                        ldr_exposure = round(safe_float(p[0], ldr_exposure), ndigits=6)
                        print("LDR exposure:", ldr_exposure)
                    ldr = camera.capture_async(ldr_exposure)

                if cmd == "hdr":
                    if len(p) > 0:
                        hdr_count = safe_int(p[0], 0)
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
                    print("Save to:", prefix)

                if cmd == 'script':
                    if len(p) < 1:
                        print("Define script name")
                        continue

                    script = p[0]

                    if not os.path.exists(script + ".script"):
                        print(Magenta + "File " + script + ".script does not exist!")
                        continue

                    print("Loading script: \"%s.script\"" % (script))

                    with open(script + ".script", "r") as f:
                        for line in f.readlines():
                            if len(line) < 1 or line[0] == "#":
                                continue
                            input_queue.put(line[:-1])

                if cmd == 'delay':
                    if len(p) < 1:
                        print("Set delay in seconds")
                        continue
                    delay = safe_float(p[0], 0)
                    timestamp = time.time()

                if cmd == 'dump':
                    if len(p) < 1:
                        print("Define filename")
                        continue
                    with open(p[0], "w") as f:
                        f.write("\n".join(history))

                if cmd == 'status':
                    print("\nSave to:", prefix)
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

            # if camera and camera.ready():
            #     if current_capture is not None:
            #         scan[current_capture[0]]['images'].append((current_capture[1], camera.result))
            #         print(camera.result)
            #         camera.discard_result()
            #         current_capture = None
            #     if not capture_queue.empty():
            #         current_capture = capture_queue.get()
            #         print("current:", current_capture)
            #         camera.start_capture(current_capture[1])
            #
            # if not input_queue.empty() and current_capture is None:
            #     if timestamp is not None:
            #         if time.time()-timestamp < delay:
            #             continue
            #
            #     if camera and not capture_queue.empty():
            #         continue
            #
            #     cmd = input_queue.get()
            #     print("Got:", cmd)
            #     cmd = cmd.split(" ")
            #     p = cmd[1:]
            #     cmd = cmd[0]
            #
            #     if not cmd in supported_commands and not cmd in props.keys():
            #         print(colored("Unrecognized command or property: \"%s\""%cmd, "magenta"))
            #         continue
            #
            #     if cmd == 'scan':
            #         if len(p) < 1:
            #             print("Define scan name")
            #             continue
            #         scan_name = p[0]
            #         print("Scan: \"%s\"" % (scan_name))
            #
            #     if cmd == 'script':
            #         if len(p) < 1:
            #             print("Define script name")
            #             continue
            #         script = p[0]
            #
            #         if not os.path.exists(script + ".script"):
            #             print(colored("File " + script + ".script does not exist!", "magenta"))
            #             continue
            #
            #         print("Loading script: \"%s.script\"" % (script))
            #
            #         with open(script + ".script", "r") as f:
            #             for line in f.readlines():
            #                 if len(line) < 1 or line[0] == "#":
            #                     continue
            #                 input_queue.put(line[:-1])
            #
            #
            #     if cmd == 'delay':
            #         if len(p) < 1:
            #             print("Set delay in seconds")
            #             continue
            #         delay = safe_float(p[0], 0)
            #         timestamp = time.time()
            #
            #     if cmd == 'property':
            #         if len(p) < 1:
            #             print("Choose property:", props.keys())
            #             continue
            #         if not p[0] in props.keys():
            #             print("Unrecognized property:", p[0])
            #             continue
            #         active_property = p[0]
            #
            #     if cmd in props.keys():
            #         active_property = cmd
            #         if len(p) > 0:
            #             if p[0] == 's' or p[0] == 'w':
            #                 props[active_property] += step if p[0] == 'w' else -step
            #             else:
            #                 if active_property in float_props:
            #                     props[active_property] = safe_float(p[0], props[active_property])
            #                 else:
            #                     props[active_property] = safe_int(p[0], props[active_property])
            #             pattern_image = None
            #
            #     if cmd == 'step':
            #         if len(p) < 1:
            #             print("Define step")
            #             continue
            #         try:
            #             step = safe_int(p[0], step)
            #         except Exception as e:
            #             print(e)
            #
            #     if cmd == 'value':
            #         if len(p) < 1:
            #             print("Define value")
            #             continue
            #         try:
            #             if active_property in float_props:
            #                 props[active_property] = safe_float(p[0], props[active_property])
            #             else:
            #                 props[active_property] = safe_int(p[0], props[active_property])
            #             pattern_image = None
            #         except Exception as e:
            #             print(e)
            #
            #     if cmd == 'w' or cmd == 's':
            #         props[active_property] += step if cmd == 'w' else -step
            #         pattern_image = None
            #
            #     if cmd == 'exposures':
            #         if len(p) < 1:
            #             print("Choose exposures:", Camera.supported_exposures)
            #             continue
            #         exposures = []
            #         for exp in p:
            #             if not exp in Camera.supported_exposures:
            #                 print("Unsupported exposure:", exp)
            #             else:
            #                 exposures.append(exp)
            #         print("Selected exposures:", exposures)
            #
            #     if cmd == 'capture':
            #         if len(p) < 1:
            #             print("Define hdr name")
            #             continue
            #         if len(exposures) == 0:
            #             print("Define exposures first")
            #             continue
            #         hdr_name = p[0]
            #         scan[hdr_name] = {"scan" : scan_name, "pattern" : active_pattern, "properties" : props.copy(), "images" : []}
            #         # save_pattern(proj.get_pattern(), hdr_name+'.png')
            #         for exp in exposures:
            #             capture_queue.put((hdr_name, exp))
            #
            #     if cmd == 'status':
            #         print("\n\tStatus:")
            #         if camera:
            #             print("Camera ready:", camera.ready())
            #         print("Active pattern:", active_pattern)
            #         print("Active property:", active_property)
            #         print("Step =", step)
            #         print("Selected exposures:", exposures)
            #         print(scan)
            #         print(list(capture_queue.queue), "\n")

        # save_scan(scan, scan_name+'.json')

    finally:
        if camera:
            camera.stop_stream()
            camera.close()
        glfw.terminate()

    print('Done')
