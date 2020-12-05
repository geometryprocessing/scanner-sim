import time
import threading
import queue
import numpy as np

from patterns import *

import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

# H, W = 800, 1280
# H, W = 900, 1600
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
        self.update(gen_blank((H,W)))

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
        new_texture = np.zeros((H,W,4), dtype=np.uint8)
        new_texture[:,:,:3] = rgb[:,:,:3]

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

    def update(self, new_pattern = None):
        if new_pattern is not None:
            self.pattern.update(new_pattern)

        self.pattern.render()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

def console_input(input_queue):
    last_input = None
    print('Enter Commands:')
    while True:
        new_input = input()
        if new_input == "" and last_input is not None:
            new_input = last_input
        else:
            last_input = new_input
        input_queue.put(new_input)
        time.sleep(0.02)

def parallel_input():
    input_queue = queue.Queue()

    input_thread = threading.Thread(target=console_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()
    return input_queue

def main():
    try:
        proj = Projector()
        input_queue = parallel_input()
        input_queue.put("checker")

        while not proj.should_close():
            new_pattern = None
            if not input_queue.empty():
                cmd = input_queue.get()
                print("Got:", cmd)
                if cmd == 'red':
                    patt = np.ones((H,W,3), dtype=np.uint8)
                    patt[:,:,0] = np.random.randint(255, size=(H,W))
                    new_pattern = patt
                if cmd == 'stripes':
                    new_pattern = gen_stripes((H, W), 10, 0, 255)
                if cmd == 'circle':
                    new_pattern = gen_circle((H, W), (300, 300), 150, 150, 255, 0.997)
                if cmd == 'checker':
                    new_pattern = gen_checker((H, W), (300, 300), 150, 2, 2, 255)
                if cmd == 'line':
                    new_pattern = gen_line((H, W), (300, 300), (600, 600), 2, 255)
                if cmd[:4] == 'load':
                    new_pattern = load_pattern(cmd[5:])
                if cmd[:4] == 'save':
                    save_pattern(proj.get_pattern(), cmd[5:])
                if cmd == 'exit':
                    break

            proj.update(new_pattern)
            time.sleep(0.02)

        glfw.terminate()
    except Exception as e:
        glfw.terminate()
        raise e

if __name__ == "__main__":
    main()
    print('Done')