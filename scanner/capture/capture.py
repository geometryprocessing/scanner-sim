import os
import queue
import threading
from concurrent import futures
import time
import json
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from arena_api.system import system
from arena_api.callback import callback, callback_function
# from arena_api.buffer import BufferFactory
from hdr import *

Black = '\u001b[30m'
Red = '\u001b[31m'
Green = '\u001b[32m'
Yellow = '\u001b[33m'
Blue = '\u001b[34m'
Magenta = '\u001b[35m'
Cyan = '\u001b[36m'
White = '\u001b[37m'
Reset = '\u001b[39m'


@callback_function.device.on_buffer
def on_buffer(buffer, *args, **kwargs):
    status = Red + "incomplete" + Reset if buffer.is_incomplete else Green + "complete" + Reset
    print("Got %s buffer at %.3f sec" % (status, kwargs['now']()))

    cam = kwargs['camera']
    cam.received_at.append(kwargs['now']())

    # ~ 20-30 ms (depends on pixel format)
    if buffer.is_incomplete:
        cam.img = None
    else:
        p = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint8 if cam.pixel_format == "Mono8" else ctypes.c_uint16))
        cam.img = np.copy(np.ctypeslib.as_array(p, (buffer.height, buffer.width)))

    # ~ 50 ms
    # cam.buf = BufferFactory.copy(buffer)

    cam.was_incomplete.append(buffer.is_incomplete)
    cam.got_buffer = True


class Camera:
    # Arena SDK wrapper
    def __init__(self):
        self.device = None
        self.handle = None
        self.t0 = time.time()

        self.roi = (0, 0, 0, 0)
        self.pixel_format = ""

        self.got_buffer = False
        self.img = None
        # self.buf = None

        self.armed_at, self.triggered_at, self.exposures, self.received_at, self.was_incomplete = [], [], [], [], []
        self.stopped_at = 0

    def time_zero(self):
        self.t0 = time.time()

    def now(self):
        return time.time() - self.t0

    def open(self, device_id=0):
        self.device = None
        self.time_zero()

        devices = system.create_device()
        print(f'\nCreated {len(devices)} device(s)')

        try:
            device = devices[device_id]
        except IndexError as ie:
            if len(devices) == 0:
                print(Red + 'No device found!' + Reset)
            else:
                print((Red + 'Only %d device(s) available!' + Reset) % len(devices))
            raise ie
        print(f'Using device:\n\t{device}')

        device.nodemap['UserSetSelector'].value = 'Default'
        device.nodemap['UserSetLoad'].execute()

        self.device = device
        self.handle = callback.register(self.device, on_buffer, camera=self, now=self.now)
        print("Opened device in %.3f sec" % self.now())

    def close(self):
        if self.handle:
            callback.deregister(self.handle)
            self.handle = None
        system.destroy_device()
        self.device = None
        print("Destroyed devices at %.3f sec" % self.now())

    # roi = (Width, Height, OffsetX, OffsetY)
    def init(self, roi=None, pixel_format="Mono12"):
        if self.device:
            nm = self.device.nodemap

            nm['Width'].value = nm['Width'].max if not roi else roi[0]
            nm['Height'].value = nm['Height'].max if not roi else roi[1]
            nm['OffsetX'].value = nm['OffsetX'].min if not roi else roi[2]
            nm['OffsetY'].value = nm['OffsetY'].min if not roi else roi[3]
            nm['PixelFormat'].value = pixel_format

            self.roi = (nm['Width'].value, nm['Height'].value, nm['OffsetX'].value, nm['OffsetY'].value)
            self.pixel_format = nm['PixelFormat'].value

            print("\nFrame Format:\n\t", self.roi, self.pixel_format)

            nm['TriggerMode'].value = 'On'
            nm['TriggerSource'].value = 'Software'
            nm['TriggerSelector'].value = 'FrameStart'

            nm['AcquisitionMode'].value = 'Continuous'
            nm['ExposureAuto'].value = 'Off'

            # should help with missing image data but doesn't (just increases delay)
            # self.device.tl_stream_nodemap['StreamPacketResendEnable'].value = True
            # self.device.tl_stream_nodemap['StreamMaxNumResendRequestsPerImage'].value = 500
        else:
            raise RuntimeError("No open device!")

    def __enter__(self):
        if self.device:
            self.time_zero()

            self.armed_at, self.triggered_at, self.exposures, self.received_at, self.was_incomplete = [], [], [], [], []
            self.stopped_at = 0

            # bug with start_stream (delay of ~ 1/FrameRate)
            self.device.nodemap['AcquisitionFrameRateEnable'].value = True
            self.device.nodemap['AcquisitionFrameRate'].value = self.device.nodemap['AcquisitionFrameRate'].max

            self.device.start_stream()
            print("\nStarted stream in %.3f sec\n" % self.now())
            return self
        else:
            raise RuntimeError("No open device!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device:
            self.device.stop_stream()
            print("\nStopped stream at %.3f sec" % self.now())
            self.stopped_at = self.now()

    # non-blocking, exposure in seconds
    def start_frame(self, exposure, silent=False):
        self.got_buffer = False
        self.img = None

        if self.device:
            nm = self.device.nodemap
            fps_node, exp_node = nm['AcquisitionFrameRate'], nm['ExposureTime']

            # triggers are ignored unless trigger is armed
            # ~ a few ms delay if you disabled AcquisitionFrameRate right after trigger (~ 1/FrameRate otherwise!)
            while not nm['TriggerArmed'].value:
                time.sleep(0.001)
            if not silent:
                print("\nTrigger armed at %.3f sec" % self.now())
            self.armed_at.append(self.now())

            # bug with ExposureTime limit (1/FrameRate even if AcquisitionFrameRate is disabled)
            nm['AcquisitionFrameRateEnable'].value = True
            fps_node.value = max(fps_node.min, min(1. / (1.1 * exposure), fps_node.max))
            real_fps = fps_node.value

            # configure exposure (in us)
            exp_node.value = max(exp_node.min, min(1.e+6 * exposure, exp_node.max))
            real_exp = exp_node.value
            self.exposures.append(real_exp / 1.e+6)

            # trigger exposure
            nm['TriggerSoftware'].execute()
            # ~ 40-50 ms delay since armed
            if not silent:
                print("Triggered at %.3f sec (%.3f ms exposure @ %.3f fps)" % (self.now(), 0.001 * real_exp, real_fps))
            self.triggered_at.append(self.now())

            # bug with TriggerArmed (delay of ~1/FrameRate if don't disable)
            nm['AcquisitionFrameRateEnable'].value = False

            # the actual exposure is slightly different from requested
            return real_exp / 1.e+6
        else:
            raise RuntimeError("No open device!")

    def frame_ready(self):
        return self.got_buffer

    def retrieve_frame(self):
        return self.img

    # blocking, exposure in seconds
    def capture_ldr(self, exposure, recapture_incomplete=True, silent=False):
        real_exp = self.start_frame(exposure, silent)

        while self.img is None:
            while not self.got_buffer:
                time.sleep(0.001)
            if not recapture_incomplete:
                break
            if self.img is None:
                print(Yellow + "Image is missing data! Recapturing..." + Reset)
                self.start_frame(exposure, silent)

        return real_exp, self.img

    def plot_timeline(self, processing_intervals=None):
        if len(self.received_at) == 0:
            return

        plt.figure("Acquisition Timeline", (16, 8))
        plt.title("Acquisition Timeline")

        plt.plot(self.armed_at, [0] * len(self.armed_at), '.r', label="Trigger Armed")

        for i, (trig, exp, inc) in enumerate(zip(self.triggered_at, self.exposures, self.was_incomplete)):
            plt.plot([trig, trig + exp], [0, 0], '--b' if inc else '-b',
                     label="Exposure" if i == len(self.exposures) - 1 else None)

        plt.plot(self.received_at, [0] * len(self.received_at), '.g', label="Buffer Received")
        x_max = max(self.stopped_at, self.received_at[-1] + 0.1)

        if processing_intervals:
            for i, pi in enumerate(processing_intervals):
                plt.plot(pi[:2], -0.1 * np.ones(2) * (pi[2] + 2), '.-' + ('m' if pi[2] == -1 else "k"),
                         label="Processing" if i == 0 else ("Adding" if i == len(processing_intervals) - 1 else None))
            x_max = max(x_max, max([pi[1] + 0.1 for pi in processing_intervals]))

        plt.xlim([0, x_max])
        plt.ylim([-1, 0.4])
        plt.xlabel("Time, sec")
        plt.legend()
        plt.tight_layout()


def capture_batch(path, exposures, roi=None, pixel_format="Mono12", save_preview=True, plot=True):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    camera = Camera()
    camera.open()
    camera.init(roi, pixel_format)

    target_exposures = exposures
    actual_exposures = []

    print("\nTotal exposure time:\n\t", np.sum(target_exposures), "seconds")
    with camera as cam:
        for i, exp in enumerate(target_exposures):
            real_exp, img = cam.capture_ldr(exp)
            actual_exposures.append(real_exp)

            if img is not None:
                np.save(path + str(i), img)

                if save_preview:
                    plt.figure(str(exp) + " sec", (16, 9))
                    plt.imshow(img, vmin=1, vmax=2**8 if cam.pixel_format == "Mono8" else 2**12)
                    plt.title(str(exp) + " sec")
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(path + str(i) + '.png', dpi=100, bbox_inches='tight')
            else:
                print("Image %d is missing data!" % i)

    with open(path + "exposures.json", "w") as f:
        json.dump(list(zip(target_exposures, actual_exposures)), f)

    if plot:
        camera.plot_timeline()

    camera.close()


def capture_dark_frames(path, exposures, count=3):
    print("Total exposure time:\n\t", np.sum(exposures)*count, "seconds")
    print("Estimated service time:\n\t", 0.1*len(exposures)*count, "seconds")

    for exp in exposures:
        full_path = path + "dark_frame_" + str(exp) + "_sec/"
        capture_batch(full_path, [exp]*count, roi=None, pixel_format="Mono12", save_preview=False, plot=False)

    with open(path + "exposures.json", "w") as f:
        json.dump(exposures, f)


def capture_hdr(exposures, low=0.1, high=0.7, camera=None, dark_path=None, gamma=None, plot=False, save_preview=False):
    if dark_path:
        for exp in exposures:
            name = "dark_frame_" + str(exp) + "_sec.exr"
            if not os.path.exists(dark_path + name):
                raise EnvironmentError("Dark frame %s not found in %s" % (name, dark_path))

    if len(exposures) < 2:
        raise ValueError("At least two different exposures are required to capture HDR")

    camera_was_none = camera is None

    if camera is None:
        camera = Camera()
        camera.open()
        camera.init(None, "Mono12")
    else:
        if camera.pixel_format != "Mono12":
            raise AttributeError("HDR supported in 12 bit mode only")

    target_exposures = sorted(exposures)
    processing_intervals = []

    print("\nCapturing HDR with %d exposures:" % len(exposures), target_exposures)
    print("Total exposure time:\n\t", np.sum(exposures), "seconds")
    print("Estimated service time:\n\t", round(0.2*len(exposures), ndigits=3), "seconds")

    size = (camera.roi[1], camera.roi[0])
    total_light = np.zeros(size, dtype=np.float)
    total_exp = np.zeros(size, dtype=np.float)
    counts = np.zeros(size, dtype=np.int)

    executor = futures.ThreadPoolExecutor(max_workers=8)
    image_queue = queue.Queue()

    def parallel_processing(i, real_exp, ldr, verbose=False):
        t0 = camera.now()
        image = ldr.astype(np.float32)

        if dark_path:
            name = "dark_frame_" + str(target_exposures[i]) + "_sec.exr"
            dark_frame = load_openexr(dark_path + name)
            image -= dark_frame
            r, c = np.nonzero(image < 0)
            image[r, c] = 0
            if verbose:
                print("Subtracted", name)

            r, c = np.nonzero(dark_frame > 2 ** 10)
            # Potential index out of bounds error but not with our dark frames)
            image[r, c] = 0.25 * (image[r, c + 1] + image[r, c - 1] + image[r + 1, c] + image[r - 1, c])
            if verbose:
                print("Replaced %d hot pixel(s) in frame %d" % (r.shape[0], i))

        image /= 2 ** 12 - 3
        if verbose:
            print("Normalized frame", i)

        if gamma:
            image = np.power(image, 1 / gamma)
            if verbose:
                print("Gamma corrected frame", i)

        processing_intervals.append((t0, camera.now(), int(threading.current_thread().name[-1])))
        print("%sProcessed frame %d%s in %.3f sec" % (Magenta, i, Reset, camera.now() - t0))
        image_queue.put((i, real_exp, image))

    def parallel_capture(camera, exposures):
        for i, exp in enumerate(exposures):
            print("Capturing frame %d with %s%s sec%s exposure" % (i, Blue, str(exp), Reset))
            real_exp, ldr = camera.capture_ldr(exp, silent=True)
            executor.submit(lambda p: parallel_processing(*p), [i, real_exp, ldr])
        print(Cyan + "\nDone capturing\n" + Reset)

    with camera as cam:
        capture_thread = threading.Thread(target=parallel_capture, daemon=True,
                                          args=(cam, target_exposures))
        capture_thread.start()

        n = 0
        while n < len(target_exposures):
            while not image_queue.empty():
                i, real_exp, image = image_queue.get()

                t0 = cam.now()
                if i == 0:
                    r, c = np.nonzero(image > low - eps)
                    total_light[r, c] = image[r, c]
                    total_exp[r, c] = real_exp
                    counts[r, c] = 1
                elif i == len(exposures) - 1:
                    r, c = np.nonzero(image < high + eps)
                    total_light[r, c] += image[r, c]
                    total_exp[r, c] += real_exp
                    counts[r, c] += 1
                else:
                    r, c = np.nonzero((low < image) & (image < high))
                    total_light[r, c] += image[r, c]
                    total_exp[r, c] += real_exp
                    counts[r, c] += 1

                processing_intervals.append((t0, cam.now(), -1))
                print("%sAdded frame %d%s in %.3f sec" % (Magenta, i, Reset, cam.now() - t0))
                n += 1

            time.sleep(0.001)

    if camera_was_none:
        camera.close()

    hdr = total_light / total_exp
    print(Cyan + "\nComputed HDR\n" + Reset)

    if plot:
        print("Plotting")

        plt.figure("Histograms", (12, 6))
        plt.subplot(1, 2, 1)
        m = np.max(counts)
        plt.hist(counts.ravel(), bins=m, range=[0.5, m + 0.5])
        plt.title("Counts")
        plt.subplot(1, 2, 2)
        plt.hist(np.log10(hdr.ravel()), bins=500)
        plt.title("Dynamic Range")
        plt.yscale("log")
        plt.tight_layout()

        plt.figure("Counts", (16, 9))
        plt.imshow(counts)
        plt.colorbar()
        plt.tight_layout()

        plt.figure("HDR", (16, 9))
        plt.imshow(hdr)
        plt.colorbar()
        plt.tight_layout()

        if save_preview:
            plt.savefig("hdr.png", dpi=120, bbox_inches='tight')

        camera.plot_timeline(processing_intervals)

    return hdr


if __name__ == "__main__":
    exposures = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75,
                 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10]

    # capture_dark_frames("D:/scanner_sim/dark_frames/", exposures, count=100)
    # exit()

    hdr = capture_hdr(exposures, dark_path="dark_frames/", gamma=default_gamma, plot=True, save_preview=True)
    # hdr = capture_hdr([0.1, 0.5, 1.5], dark_path="dark_frames/", gamma=default_gamma, plot=True, save_preview=True)
    print("Ready")

    # np.save("hdr", hdr.astype(np.float32))
    save_openexr("hdr.exr", hdr)
    print("Saved")

    plt.show()
    print('Done')
    exit()

    # path = "gamma/"
    # path = "hdr/"
    # path = "spurious/"
    # path = "dark/"
    path = "calib/"

    # target_exposures = np.logspace(-4, 1, 300)
    # target_exposures = [0.001, 0.004, 0.01, 0.04, 0.1, 0.4, 1, 2, 4, 8, 10]
    # target_exposures = [0.1, 0.3, 0.6, 1, 2, 3.5, 5, 6.5, 8, 10]
    # target_exposures = [0.01, 0.1, 1, 10]
    target_exposures = exposures

    # roi = (300, 300, 1000, 3000)
    roi = None

    capture_batch(path, target_exposures, roi, pixel_format="Mono12", save_preview=True, plot=True)

    plt.show()
    print('Done')
