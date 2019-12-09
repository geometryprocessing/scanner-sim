import time
import json
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from arena_api.system import system
from arena_api.callback import callback, callback_function
# from arena_api.buffer import BufferFactory

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
        print(f'Created {len(devices)} device(s)')

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
        print("\nDestroyed devices at %.3f sec" % self.now())

    # roi = (Width, Height, OffsetX, OffsetY)
    def init(self, roi=None, pixel_format="Mono8"):
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
            print("\nStarted stream in %.3f sec" % self.now())
            return self
        else:
            raise RuntimeError("No open device!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device:
            self.device.stop_stream()
            print("\nStopped stream at %.3f sec" % self.now())
            self.stopped_at = self.now()

    # non-blocking, exposure in seconds
    def start_frame(self, exposure):
        self.got_buffer = False
        self.img = None

        if self.device:
            nm = self.device.nodemap
            fps_node, exp_node = nm['AcquisitionFrameRate'], nm['ExposureTime']

            # triggers are ignored unless trigger is armed
            # ~ a few ms delay if you disabled AcquisitionFrameRate right after trigger (~ 1/FrameRate otherwise!)
            while not nm['TriggerArmed'].value:
                time.sleep(0.001)
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
    def capture_frame(self, exposure, recapture_incomplete=True):
        real_exp = self.start_frame(exposure)

        while self.img is None:
            while not self.got_buffer:
                time.sleep(0.001)
            if not recapture_incomplete:
                break
            if self.img is None:
                print(Yellow + "Image is missing data! Recapturing..." + Reset)
                self.start_frame(exposure)

        return real_exp, self.img

    def plot_timeline(self):
        if len(self.received_at) == 0:
            return

        plt.figure("Camera Timeline")
        plt.title("Camera Timeline")

        plt.plot(self.armed_at, [0] * len(self.armed_at), '.r', label="Trigger Armed")

        for i, (trig, exp, inc) in enumerate(zip(self.triggered_at, self.exposures, self.was_incomplete)):
            plt.plot([trig, trig + exp], [0, 0], '--b' if inc else '-b', label="Exposure" if i == 0 else "")

        plt.plot(self.received_at, [0] * len(self.received_at), '.g', label="Buffer Received")

        plt.xlim([0, max(self.stopped_at, self.received_at[-1] + 0.1)])
        plt.xlabel("Time, sec")
        plt.legend()


if __name__ == "__main__":
    path = "gamma/"
    # path = "hdr/"
    # path = "spurious/"

    target_exposures = np.logspace(-4, 1, 300)
    # target_exposures = [0.001, 0.004, 0.01, 0.04, 0.1, 0.4, 1, 2, 4, 8, 10]
    # target_exposures = [0.1, 0.3, 0.6, 1, 2, 3.5, 5, 6.5, 8, 10]
    print("Total exposure time:\n\t", np.sum(target_exposures), "seconds")
    actual_exposures = []

    camera = Camera()
    camera.open()
    camera.init((300, 300, 1000, 3000), pixel_format="Mono12")
    # camera.init(None, pixel_format="Mono12")

    with camera as cam:
        for i, exp in enumerate(target_exposures):
            real_exp, img = cam.capture_frame(exp, recapture_incomplete=True)
            actual_exposures.append(real_exp)

            if img is not None:
                np.save(path + str(i), img)

                plt.figure(str(exp) + " sec")
                # plt.get_current_fig_manager().window.state('zoomed')
                plt.imshow(img, vmin=0, vmax=2**8 if cam.pixel_format == "Mono8" else 2**12)
                plt.title(str(exp) + " sec")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(path + str(i) + '.png', dpi=300, bbox_inches='tight')
            else:
                print("Image %d is missing data!" % i)

    with open(path + "exposures.json", "w") as f:
        json.dump(list(zip(target_exposures, actual_exposures)), f)

    camera.plot_timeline()
    camera.close()

    # plt.show()
    print('Done')
