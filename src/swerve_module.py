from abc import ABC, abstractmethod
from math import cos, sin
import numpy as np
import queue

class heading_controller_base(ABC):
    def __init__(self) -> None:
        self.heading = 0
        self.heading_dot = 0

    @abstractmethod
    def step(self, dt):
        pass

    @abstractmethod
    def set(self, heading, heading_dot = 0):
        pass

    def rot_matrix(self):
        return np.array([[cos(self.heading), -sin(self.heading)],
                        [sin(self.heading),  cos(self.heading)]])

    def get_heading(self):
        return self.heading

    def get_heading_dot(self):
        return self.heading_dot

class heading_controller_instant(heading_controller_base):
    def __init__(self, delay = 0) -> None:
        super().__init__()
        self.heading_queue = queue.Queue()

        for i in range(delay):
            self.heading_queue.put((0, 0))

    def step(self, dt):
        self.heading += self.heading_dot*dt

    def set(self, heading, heading_dot = 0):
        self.heading_queue.put((heading, heading_dot))
        (self.heading, self.heading_dot) = self.heading_queue.get()

class linear_controller_base(ABC):
    def __init__(self) -> None:
        self.velocity = 0
        self.accel = 0

    @abstractmethod
    def step(self, dt):
        pass

    @abstractmethod
    def set(self, velocity, accel = 0):
        pass

    def get_velocity(self):
        return self.velocity

    def get_accel(self):
        return self.accel

class linear_controller_instant(linear_controller_base):
    def __init__(self, delay = 0) -> None:
        super().__init__()
        self.linear_queue = queue.Queue()

        for i in range(delay):
            self.linear_queue.put((0,0))

    def step(self, dt):
        self.velocity += self.accel*dt

    def set(self, velocity, acceleration = 0):
        self.linear_queue.put((velocity, acceleration))
        (self.velocity, self.accel) = self.linear_queue.get()

# Abstract class for swerve modules.
# Everything passed into the swerve module should be robot-oriented or non-directional
class swerve_module_base(ABC):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.lin_vel = 0

        self.linear_controller: linear_controller_base = None
        self.heading_controller: heading_controller_base = None

    @abstractmethod
    def step(self, dt):
        pass

    @abstractmethod
    def set(self, lin_vel, heading, lin_accel = 0, heading_dot = 0):
        pass

    def rot_matrix(self):
        return self.heading_controller.rot_matrix()

    def get_lin_vel(self):
        return self.linear_controller.get_velocity()

    def get_lin_accel(self):
        return self.linear_controller.get_accel()

    def get_heading(self):
        return self.heading_controller.get_heading()

    def get_heading_dot(self):
        return self.heading_controller.get_heading_dot()


# class for swerve module without inertia.
# Can give a delay in number of loops (if set() is called each loop) between when a state is commanded and when it's assigned.
class swerve_module_instant(swerve_module_base):
    def __init__(self, x, y, lin_delay = 0, heading_delay = 0) -> None:
        super().__init__(x,y)
        self.linear_controller = linear_controller_instant(lin_delay)
        self.heading_controller = heading_controller_instant(heading_delay)

    # increments lin_vel and heading according to the last state assigned by set()
    def step(self, dt):
        self.linear_controller.step(dt)
        self.heading_controller.step(dt)

    # set the module state
    def set(self, lin_vel, heading, lin_accel = 0, heading_dot = 0):
        self.linear_controller.set(lin_vel, lin_accel)
        self.heading_controller.set(heading, heading_dot)

class swerve_module_feedforward(swerve_module_base):
    pass