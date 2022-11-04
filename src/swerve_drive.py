from math import atan2
from typing import List, Tuple
import swerve_module as sm
import numpy as np
from numpy import sin, cos

class swerve_drive(object):
    def __init__(self, modules: List[sm.swerve_module_base]) -> None:
        self.modules = modules
        self.nmodules = len(modules)

        self.x = 0
        self.y = 0
        self.theta = 0

        self.vx = 0
        self.vy = 0
        self.w = 0

        self.inv_matrix_1 = np.zeros(shape = (2*self.nmodules, 3))

        self.label = '_default label'

        for i in range(self.nmodules):                          # Construct the first-order matrix
            self.inv_matrix_1[2*i] = [1, 0, -modules[i].y]
            self.inv_matrix_1[2*i+1] = [0, 1, modules[i].x]

        self.fwd_matrix_1 = np.linalg.pinv(self.inv_matrix_1)

    # Get swerve drive rotation matrix (local to global).
    # Can also give a hypothetical theta to get the rotation matrix at that angle
    def rot_matrix(self, theta = None) -> np.array:
        theta = self.theta if theta == None else theta
        return np.array([[cos(theta), -sin(theta)],
                        [sin(theta),  cos(theta)]])

    def get_pos(self) -> np.array:
        return np.array([self.x, self.y])

    def get_pose(self) -> np.array:
        return np.array([self.x, self.y, self.theta])

    # Convert a global-oriented vector to a local-oriented one. If more than 2 elements, only convert the first 2.
    def global_to_local(self, global_lin_vector: np.array) -> np.array:
        if len(global_lin_vector) == 2:
            return np.matmul(np.linalg.inv(self.rot_matrix()), global_lin_vector)
        else:
            lin_g = global_lin_vector[range(2)]
            other = global_lin_vector[range(2, len(global_lin_vector))]
            lin_l = self.global_to_local(lin_g)
            return np.hstack((lin_l, other))
    
    # Convert a local-oriented vector to a global-oriented one.  If more than 2 elements, only convert the first 2.
    def local_to_global(self, local_lin_vector: np.array) -> np.array:
        if len(local_lin_vector) == 2:
            return np.matmul(self.rot_matrix(), local_lin_vector)
        else:
            lin_l = local_lin_vector[range(2)]
            other = local_lin_vector[range(2, len(local_lin_vector))]
            lin_g = self.local_to_global(lin_l)
            return np.hstack((lin_g, other))

    # global_vels: <vx,vy, w> as 1D or column vector
    # output: (vels, headings)
    #         vels: module linear speeds
    #         headings: module angle, relative to the robot
    def inverse(self, global_vels: np.array) -> Tuple[np.array, np.array]:
        local_vels = self.global_to_local(global_vels)                  # convert field-oriented velocities to robot-oriented
        voutxy = np.matmul(self.inv_matrix_1, local_vels)               # Perform first-order IK
        vx = voutxy[list(range(0, 2*self.nmodules, 2))] # vx on even indices
        vy = voutxy[list(range(1, 2*self.nmodules, 2))] # vy on odd indices

        vels = np.sqrt(np.square(vx) + np.square(vy))                   # Find the linear speed
        headings = np.array(list(map(atan2, vy.flatten(), vx.flatten())))   # Find the module heading relative to the robot

        return (vels, headings)

    # output: global velocities <vx, vy, w> relative to the field
    def forward(self):
        module_xyvels = np.zeros(shape = 2*self.nmodules)

        for i in range(self.nmodules):                                  # Get the module x and y velocities
            m = self.modules[i];
            module_xyvels[2*i] = m.get_lin_vel()*cos(m.get_heading())
            module_xyvels[2*i+1] = m.get_lin_vel()*sin(m.get_heading())

        local_vels = np.matmul(self.fwd_matrix_1, module_xyvels)        # Perform first-order FK to get robot-oriented velocities
        global_vels = self.local_to_global(local_vels)                  # Convert robot-oriented velocities to field-oriented
        return global_vels

    # output: get module coordinates in global space [[x1, x2, x3, x4...], [y1, y2, y3, y4...]]
    def get_module_coords(self, pose = None):
        pose = self.get_pose() if (np.array([pose]) == None).any() else pose
        local_module_coords = np.zeros(shape = (2, self.nmodules))

        for i in range(self.nmodules):
            m = self.modules[i];
            local_module_coords[range(2),i] = [m.x, m.y]

        module_coords = np.matmul(self.rot_matrix(pose[2]), local_module_coords) + np.array([pose[range(2)]]).T
        return module_coords

    def get_module_vels(self):
        module_vels = np.zeros(shape = (self.nmodules,4))
        for i in range(self.nmodules):
            m = self.modules[i]
            module_vels[i, range(4)] = [m.get_lin_vel(), m.get_heading(), m.get_lin_accel(), m.get_heading_dot()]

        return module_vels


    # global_vel: <vx,vy, w> 1D or column vector
    # global_accels: ignored
    def set(self, global_vels: np.array, global_accels = None):
        (vels, headings) = self.inverse(global_vels)

        for i in range(self.nmodules):
            self.modules[i].set(vels[i], headings[i])

    # perform a time step
    def step(self, dt):
        for i in range(self.nmodules):
            self.modules[i].step(dt)

        [self.vx, self.vy, self.w] = self.forward().flatten()
        self.x += self.vx*dt
        self.y += self.vy*dt
        self.theta += self.w*dt

    def get_global_vels(self) -> np.array:
        return np.array([self.vx, self.vy, self.w])