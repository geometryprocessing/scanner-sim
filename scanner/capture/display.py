import os.path
import time
import queue
import threading
# import imageio
import cv2
import glob
import json
import glfw
import numpy as np
import matplotlib.pyplot as plt
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

H, W = 1080, 1920

strVS = """
#version 120

attribute vec3 pos;
attribute vec2 texCoords;

varying vec2 uv;

void main() {
  gl_Position = vec4(pos, 1.0); 
  uv = texCoords;
}
"""

strFS = """
#version 120

varying vec2 uv;

uniform sampler2D tex;

void main() {
     gl_FragColor = texture2D(tex, uv);
}
"""


def gen_color(dim, color=(255, 255, 255)):
    r = np.zeros((*dim, 3), dtype=np.uint8)
    r[...] = color
    return r


def gen_stripes(dim, stride, axis=0, color=(255, 255, 255)):
    img = np.zeros((*dim, 3), dtype=np.uint8)
    for i in range(0, dim[axis], 2 * stride):
        if axis == 0:
            img[i:i + stride, :, :] = color
        if axis == 1:
            img[:, i:i + stride, :] = color
    return img


def gen_dots(dim, offset=(0, 0), step=(100, 100), count=(5, 5)):
    img = np.zeros((*dim, 3), dtype=np.uint8)
    for i in range(count[0]):
        for j in range(count[1]):
            pos = np.array(offset) + np.array(step) * np.array([i, j]) + np.array([step[0] // 4, step[1] // 4])
            img[pos[0], pos[1], :] = [255, 0, 0]
            img[pos[0], pos[1] + step[1] // 2, :] = [0, 255, 0]
            img[pos[0] + step[0] // 2, pos[1], :] = [0, 0, 255]
            img[pos[0] + step[0] // 2, pos[1] + step[1] // 2, :] = [255, 255, 255]
    return img


def gen_gray(dim, color=(255, 255, 255), invert=False):
    r, c = np.indices(dim, dtype=np.uint32)
    r = np.bitwise_xor(r, np.right_shift(r, 1))
    c = np.bitwise_xor(c, np.right_shift(c, 1))

    n = np.ceil(np.log2(dim)).astype(np.int)
    R = np.zeros((n[0], *dim, 3), dtype=np.uint8)
    C = np.zeros((n[1], *dim, 3), dtype=np.uint8)

    for i in range(n[0]):
        idx = np.nonzero(np.bitwise_and(r, 1 << i))
        R[i, idx[0], idx[1], :] = color
        if invert:
            R[i, :, :, :] = color
            R[i, idx[0], idx[1], :] = 0

    for i in range(n[1]):
        idx = np.nonzero(np.bitwise_and(c, 1 << i))
        C[i, idx[0], idx[1], :] = color
        if invert:
            C[i, :, :, :] = color
            C[i, idx[0], idx[1], :] = 0

    return R, C


def gen_checker(dim, pos, size, count, lum=255):
    img = np.ones((*dim, 3), dtype=np.uint8) * int(lum)
    for i in range(count[0]):
        for j in range(count[1]):
            if (i + j) % 2 == 0:
                t = pos[0] + i * size
                l = pos[1] + j * size
                img[t:t+size, l:l+size, :] = 0

    return img


class Pattern:
    def __init__(self):
        # create texture to store pattern
        self.texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self.rgb = None
        self.update(np.zeros((H, W, 3), dtype=np.uint8))

        # compile shader to draw a textured quad
        self.program = shaders.compileProgram(shaders.compileShader(strVS, GL_VERTEX_SHADER),
                                              shaders.compileShader(strFS, GL_FRAGMENT_SHADER))
        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "tex"), 0)

        # a quad defined as triangle strip
        quad = np.array([-1, -1, 0, 0, 0,
                           1, -1, 0, 1, 0,
                          -1,  1, 0, 0, 1,
                           1,  1, 0, 1, 1], np.float32)
        self.vbo = vbo.VBO(quad)
        self.vbo.bind()

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*4, self.vbo)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*4, self.vbo+3*4)

    def update(self, rgb):
        self.rgb = rgb
        # add alpha channel
        new_texture = np.zeros((H, W, 4), dtype=np.uint8)
        new_texture[:, :, :3] = rgb[:, :, :3]

        # upload flipped vertically
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, np.flip(new_texture, axis=0))

    def render(self):
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)


class Projector:
    def __init__(self):
        glfw.init()

        # detect projector by matching resolution
        projector = None
        for monitor in glfw.get_monitors():
            mode = glfw.get_video_mode(monitor)
            if mode.size.width == W and mode.size.height == H:
                projector = monitor

        # windowed full screen
        glfw.window_hint(glfw.AUTO_ICONIFY, False)

        self.window = glfw.create_window(W, H, "Projector", projector, None)
        glfw.make_context_current(self.window)

        glfw.set_key_callback(self.window, self.on_keyboard)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        # glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        self.pattern = Pattern()

    def on_keyboard(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def get_pattern(self):
        return self.pattern.rgb

    def update(self, new_pattern=None):
        if new_pattern is not None:
            self.pattern.update(new_pattern)

        self.pattern.render()
        glfw.swap_buffers(self.window)
        glfw.poll_events()


running = True


def console_input(input_queue):
    last_input = None
    print('Enter Commands:')
    # global running
    while running:
        new_input = input()
        if new_input == "" and last_input is not None:
            new_input = last_input
        else:
            last_input = new_input
        input_queue.put(new_input)


def parallel_input():
    input_queue = queue.Queue()
    input_thread = threading.Thread(target=console_input, daemon=True, args=(input_queue,))
    input_thread.start()
    return input_queue, input_thread


def adjust_patterns(path, calib_file, plot=False):
    calib = json.load(open(calib_file))
    polies = [np.array(calib[c]) for c in ["blue", "green", "red"]]

    for file in glob.glob(path + "_ref/*.png"):
        print(file)
        img = cv2.imread(file)
        # print(img)

        if plot:
            plt.figure("LUTs", (12, 7))

        for i in range(3):
            pix = np.arange(256)
            poly = polies[i]
            a, b, c = poly[0], poly[1], -pix
            lut = poly[0] * pix * pix + poly[1] * pix + poly[2]
            lut = np.round(255 * lut / lut[-1]).astype(np.uint8)
            inv_lut = (-b + np.sqrt(b*b - 4*a*c)) / (2 * a)
            inv_lut = np.round(255 * inv_lut / inv_lut[-1]).astype(np.uint8)

            img[:, :, i] = inv_lut[img[:, :, i]]

            if plot:
                print(i, inv_lut)
                plt.plot(pix, lut, label=str(i))
                plt.plot(pix, inv_lut, ".-", label="inv_"+str(i))

        if plot:
            plt.legend()
            plt.tight_layout()

            plt.show()
            break

        cv2.imwrite(path + "/" + os.path.basename(file), img)


if __name__ == "__main__":
    # adjust_patterns("./patterns/gradient", "../calibration/projector/response/projector_polies.json")
    # exit(0)

    try:
        projector = Projector()
        input_queue, thr = parallel_input()
        input_queue.put("checker")
        i = 0

        while not projector.should_close():
            new_pattern = None
            if not input_queue.empty():
                cmd = input_queue.get()
                print("Got:", cmd)
                if cmd == 'red':
                    patt = np.zeros((H, W, 3), dtype=np.uint8)
                    patt[:, :, 0] = np.random.randint(255, size=(H, W))
                    new_pattern = patt
                if cmd == 'w':
                    new_pattern = gen_color((H, W), color=(255, 255, 255))
                if cmd == 'k':
                    new_pattern = gen_color((H, W), color=(0, 0, 0))
                if cmd == 'r':
                    new_pattern = gen_color((H, W), color=(255, 0, 0))
                if cmd == 'g':
                    new_pattern = gen_color((H, W), color=(0, 255, 0))
                if cmd == 'b':
                    new_pattern = gen_color((H, W), color=(0, 0, 255))
                if cmd == 'color':
                    new_pattern = gen_color((H, W), color=(128, 255, 128))
                if cmd == 'stripes':
                    new_pattern = gen_stripes((H, W), 1, axis=0)
                if cmd == 'gray':
                    R, C = gen_gray((H, W), color=[255, 0, 0])
                    new_pattern = R[i, :, :, :]
                if cmd == 'u':
                    i += 1
                    R, C = gen_gray((H, W), color=[255, 0, 0])
                    new_pattern = R[i, :, :, :]
                if cmd == 'd':
                    i -= 1
                    R, C = gen_gray((H, W), color=[255, 0, 0])
                    new_pattern = R[i, :, :, :]
                if cmd == 'dots':
                    new_pattern = gen_dots((H, W), (90 + 50, 60 + 50), (100, 100), (9 - 1, 18 - 1))
                if cmd == 'checker':
                    new_pattern = gen_checker((H, W), (90, 60), 100, (9, 18))
                if cmd == 'both':
                    new_pattern_1 = gen_checker((H, W), (90, 60), 100, (9, 18))
                    new_pattern_2 = gen_dots((H, W), (90 + 50, 60 + 50), (100, 100), (9 - 1, 18 - 1))
                    new_pattern = np.maximum(new_pattern_1, new_pattern_2)
                if cmd == 'checker r':
                    new_pattern = gen_checker((H, W), (90, 60), 100, (9, 18))
                    new_pattern *= np.array([1, 0, 0], dtype=np.uint8)
                if cmd == 'checker g':
                    new_pattern = gen_checker((H, W), (90, 60), 100, (9, 18))
                    new_pattern *= np.array([0, 1, 0], dtype=np.uint8)
                if cmd == 'checker b':
                    new_pattern = gen_checker((H, W), (90, 60), 100, (9, 18))
                    new_pattern *= np.array([0, 0, 1], dtype=np.uint8)
                if cmd == 'checker t':
                    new_pattern = gen_checker((H, W), (60, 60), 100, (7, 18))
                    new_pattern *= np.array([0, 1, 0], dtype=np.uint8)
                    new_pattern[H-260:, :, :] = 0
                if cmd == 'g t':
                    new_pattern = gen_color((H, W), color=(0, 255, 0))
                    new_pattern[H-260:, :, :] = 0
                if cmd[:4] == 'load':
                    new_pattern = cv2.imread(cmd[5:])[:, :, ::-1]
                if cmd[:4] == 'save':
                    cv2.imwrite(cmd[5:]+".png", projector.get_pattern()[:, :, ::-1])
                if cmd == 'exit':
                    break

            projector.update(new_pattern)
            plt.show(block=False)
            time.sleep(0.02)

    finally:
        glfw.terminate()
    print('Done')

    # running = False
    # print('Press Enter to exit...')
    # thr.join()
