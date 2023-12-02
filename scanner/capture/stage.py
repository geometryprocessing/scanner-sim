import time
import queue
import serial
import threading

class LinearStage:
    steps_per_mm = 80
    delimiter = "\r"

    def __init__(self, port="COM3", baudrate=115200, debug=False):
        self.debug = debug
        self.running, self.homed, self.ready = False, False, False

        self.port, self.baudrate = port, baudrate
        self.ser = serial.Serial(port=port, baudrate=baudrate, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
        self.open = self.ser.isOpen()

        if self.debug:
            print(self.port, "is open:", self.open)

        if self.open:
            self.in_queue = queue.Queue()
            self.reader_thread = threading.Thread(target=self.reader, daemon=True)
            self.reader_thread.start()

            while not self.running:
                time.sleep(0.001)

    def reader(self):
        self.running = True
        received = ''
        print("Linear Stage Running")

        while self.running:
            while self.ser.inWaiting() > 0:
                received += self.ser.read(1).decode("ASCII")

            if self.delimiter in received:
                p = received.find(self.delimiter)
                resp = received[:p].strip()
                received = received[p+1:]

                if self.debug:
                    print(resp)

                if resp == "homing":
                    self.homed = False
                elif resp == "homed":
                    self.homed = True
                elif resp == "moving":
                    self.ready = False
                elif resp == "ready":
                    self.ready = True
                else:
                    self.in_queue.put(resp)

    def send(self, cmd):
        if not self.open:
            raise RuntimeError("Serial port not open")

        self.ser.write((cmd + self.delimiter).encode("ASCII"))

    def close(self):
        if self.running:
            self.running = False
            self.reader_thread.join()

    def home(self, speed_divider=8):
        if not self.running:
            raise RuntimeError("Linear stage not available")

        self.homed = False
        self.send("m.divider %d" % speed_divider)  # 24000 / speed_divider = X steps per second
        self.send("offset %d" % (24000 // (speed_divider * 5)))
        self.send("m.target 0")
        self.send("home")

        while not self.homed:
            time.sleep(0.001)

    def move(self, dist_mm):
        if not self.running:
            raise RuntimeError("Linear stage not available")

        if not self.homed:
            raise RuntimeError("Linear stage not homed")

        self.ready = False
        self.send("move %d" % int(round(dist_mm * self.steps_per_mm)))

        while not self.ready:
            time.sleep(0.001)


class RotatingStage:
    steps_per_deg = 31.205
    delimiter = "\n"

    def __init__(self, port="COM3", baudrate=9600, debug=False):
        self.debug = debug
        self.running, self.ready = False, False

        self.port, self.baudrate = port, baudrate
        self.ser = serial.Serial(port=port, baudrate=baudrate, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
        self.open = self.ser.isOpen()

        if self.debug:
            print(self.port, "is open:", self.open)

        if self.open:
            self.in_queue = queue.Queue()
            self.reader_thread = threading.Thread(target=self.reader, daemon=True)
            self.reader_thread.start()

            while not self.running:
                time.sleep(0.001)

    def reader(self):
        self.running = True
        received = ''
        print("Rotating Stage Running")

        while self.running:
            while self.ser.inWaiting() > 0:
                received += self.ser.read(1).decode("ASCII")

            if self.delimiter in received:
                p = received.find(self.delimiter)
                resp = received[:p].strip()
                received = received[p+1:]

                if self.debug:
                    print(resp)

                if resp == "status running":
                    self.ready = False
                elif resp == "status stopped":
                    self.ready = True
                else:
                    self.in_queue.put(resp)

    def send(self, cmd):
        if not self.open:
            raise RuntimeError("Serial port not open")

        if self.debug:
            print(cmd)

        self.ser.write((cmd + self.delimiter).encode("ASCII"))

    def close(self):
        if self.running:
            self.running = False
            self.reader_thread.join()

    def move(self, dist_deg):
        if not self.running:
            raise RuntimeError("Rotating stage not available")

        self.ready = False
        self.send("move %d" % round(dist_deg * self.steps_per_deg))

        while not self.ready:
            time.sleep(0.001)


if __name__ == "__main__":
    stage = LinearStage(port="COM3", debug=True)
    stage.home()

    stage.move(5)
    # stage.move(-5)

    stage.close()

    # stage = RotatingStage(port="COM3", debug=True)
    #
    # stage.move(30)
    # stage.move(-30)
    #
    # stage.close()
