import os
import queue
import threading
import time
from concurrent import futures

import cv2

from utils import *

wb_1 = (1.0, 1.0, 1.0)


def combine_colors(prefix, black, suffixes=("_b", "_g", "_r"), wb=wb_1, sigma=1.0):
    bgr = []

    for c, b in zip(suffixes, wb):
        d = cv2.imread(prefix + c + ".tiff", cv2.IMREAD_UNCHANGED).astype(np.float32)

        if sigma is not None:
            d = gaussian_filter(d, sigma=sigma)

        d = np.maximum(0.0, d - black) #* b

        bgr.append(d.astype(np.uint16))

    cv2.imwrite(prefix + ".tiff", np.stack(bgr, axis=2))


class Corrector:
    def __init__(self):
        self.async_executor = futures.ThreadPoolExecutor(max_workers=8)
        self.async_executor2 = futures.ThreadPoolExecutor(max_workers=2)
        self.nonce, self.tail = 0, 0
        self.queue = queue.Queue()
        self.dark_frames = {}
        self.failed = False
        self.e = None

    def check_errors(self, future):
        if self.e is not None and not self.failed:
            self.failed = True
            raise self.e

    def correct_and_save(self, img, exp, filename):
        if self.failed:
            raise RuntimeError("Cannot proceed after error!")

        self.nonce += 1
        future = self.async_executor.submit(self._correct_and_save, img, exp, filename)
        future.add_done_callback(self.check_errors)

    def _correct_and_save(self, img, exp, filename):
        try:
            dark_frame = self.dark_frames[str(exp)]
            # print(np.min(dark_frame), np.max(dark_frame))
            img = np.maximum(0, img.astype(np.float32) - dark_frame)
            r, c = np.nonzero(dark_frame > 2 ** 8)
            # Potential index out of bounds error but not with our dark frames)
            for i in range(10):
                img[r, c] = 0.25 * (img[r, c + 1] + img[r, c - 1] + img[r + 1, c] + img[r - 1, c])
            # print("Replaced %d hot pixel(s) in" % r.shape[0], filename)
            img /= 2 ** 12 - 3  # Normalize by 4093 (4094 - dark floor)
            assert filename.endswith(".tiff")
            cv2.imwrite(filename, np.round(img * (2 ** 16 - 1)).astype(np.uint16))
            print("Saved", filename)
        except Exception as e:
            print(e)
            self.e = e
        self.tail += 1

    def process(self, path, blank_name):
        if self.failed:
            raise RuntimeError("Cannot proceed after error!")

        future = self.async_executor.submit(self._process, path, blank_name, int(self.nonce))
        future.add_done_callback(self.check_errors)

    def _process(self, path, blank_name, nonce):
        while self.tail < nonce:
            time.sleep(0.1)

        try:
            print("Processing", path)
            blank = cv2.imread(path + blank_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

            # m = np.sort(blank.ravel())[int(0.995 * blank.size)]
            # blank2 = np.power(blank / m, 0.75)
            # save_ldr(path + blank_name[:blank_name.rfind(".")] + ".jpg",
            #          np.minimum(255 * blank2, 255).astype(np.uint8), ensure_rgb=True)

            blank = gaussian_filter(blank, sigma=1.5)
            blank /= (2.5 / 0.25)

            # checker = cv2.imread(path + "checker.tiff", cv2.IMREAD_UNCHANGED).astype(np.float32)
            # uniform = cv2.imread(path + "green.tiff", cv2.IMREAD_UNCHANGED).astype(np.float32)
            # uniform = gaussian_filter(uniform, sigma=1.5)
            #
            # checker = np.maximum(0.0, checker - blank)
            # uniform = np.maximum(0.0, uniform - blank)
            # checker = checker / uniform
            # checker[uniform < 0.01 * np.max(uniform)] = 0
            # checker = np.power(checker / 1.1, 0.75)
            #
            # save_ldr(path + "checker.jpg", np.minimum(255 * checker, 255).astype(np.uint8), ensure_rgb=True)

            jobs = [joblib.delayed(combine_colors)(file[:-7], blank, sigma=1.5) for file in glob.glob(path + "*_r.tiff")]
            joblib.Parallel(verbose=15, n_jobs=len(jobs), batch_size=len(jobs), pre_dispatch="all", backend="threading")(jobs)

            for c in ["r", "g", "b"]:
                [os.remove(file) for file in glob.glob(path + "*_%s.tiff" % c)]

            print("Processed", path)
        except Exception as e:
            print(e)
            self.e = e

