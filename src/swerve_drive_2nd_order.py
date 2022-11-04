from typing import List, Tuple
import swerve_module as sm
import numpy as np
from numpy import sin, cos
import swerve_drive

class swerve_drive_2nd_order(swerve_drive.swerve_drive):
    def __init__(self, modules: List[sm.swerve_module_base]) -> None:
        super().__init__(modules)

        self.ax = 0
        self.ay = 0
        self.alpha = 0
        
        self.inv_matrix_2 = np.zeros(shape = (2*self.nmodules, 4))

        for i in range(self.nmodules):                                  # Construct the second order kinematics matrix
            self.inv_matrix_2[2*i] = [1, 0, -modules[i].x, -modules[i].y]
            self.inv_matrix_2[2*i+1] = [0, 1, -modules[i].y, modules[i].x]

        self.fwd_matrix_2 = np.linalg.pinv(self.inv_matrix_2)

    # global_vels: <vx,vy, w> as 1D or column vector
    # output: vels: linear velocity of each module
    #         headings: heading of each module, as measured relative to the robot
    #         accels: linear acceleration of each module
    #         omegas: angular velocity of each module, in addition to the angular velocity of the robot
    def inverse(self, global_vels: np.array, global_accels: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
        (vels, headings) = super().inverse(global_vels)                 # perform first-order IK
        local_accels = self.global_to_local(global_accels)              # reorient linear accelerations to be relative to robot directions
        local_accels = np.insert(global_accels, 2, global_vels[2]**2)   # add the omega^2 term
        
        aoutxy = np.matmul(self.inv_matrix_2, local_accels)             # module x and y accelerations from the second-order IK
        
        ax = aoutxy[list(range(0, 2*self.nmodules, 2))]                 # ax on even indices
        ay = aoutxy[list(range(1, 2*self.nmodules, 2))]                 # ay on odd
        
        accels = cos(headings)*ax + sin(headings)*ay                    # Get the linear wheel acceleration
        omegas = cos(headings)/vels*ay - sin(headings)/vels*ax - self.w # Get the module angular velocity. Note since IK gives angular velocity
                                                                            # relative to field, need to subtract robot's angular velocity

        return (vels, headings, accels, omegas)

    # output: robot velocities and angles relative to the field reference frame
    #         global_vels: <vx, vy, theta> from first-order
    #         global_accels: <ax, ay, omega> from second-order
    def forward(self):
        module_xyaccels = np.zeros(shape = 2*self.nmodules)

        for i in range(self.nmodules):                                  # Get the x and y accelerations of each module relative to the robot axes
            m = self.modules[i];
            m_lin_vel = m.get_lin_vel()
            m_heading = m.get_heading() + self.theta
            m_omega = m.get_heading_dot() + self.w
            M = self.rot_matrix(m_heading)
            module_xyaccels[[2*i, 2*i+1]] = np.matmul(M, np.array([m.get_lin_accel(), m_omega*m_lin_vel]))

        global_vels = super().forward()                                 # Perform first order FK
        [ax, ay, w2, alpha] = np.matmul(self.fwd_matrix_2, module_xyaccels) # Perform second order FK, getting robot accelerations relative to robot axes
        local_accels = np.array([ax, ay, alpha])
        global_accels = self.local_to_global(local_accels)              # Convert robot-oriented accelerations to field-oriented
        return (global_vels, global_accels)

    # Set the target robot velocities and accelerations, measured relative to the field reference frame
    # global_vels: <vx, vy, w> 1D or column vector
    # global_accels: <ax, ay, alpha> 1D or column vector
    def set(self, global_vels: np.array, global_accels: np.array = np.zeros(shape=3)):
        (vels, headings, accels, omegas) = self.inverse(global_vels, global_accels)
        
        for i in range(self.nmodules):
            self.modules[i].set(vels[i], headings[i], accels[i], omegas[i])

    # Perform a time step.
    def step(self, dt):
        (global_vels, global_accels) = self.forward()
        [self.vx, self.vy, self.w] = global_vels.flatten()
        [self.ax, self.ay, self.alpha] = global_accels.flatten()

        for i in range(self.nmodules):
            self.modules[i].step(dt)

        self.x += self.vx*dt + .5*self.ax*dt**2
        self.y += self.vy*dt + .5*self.ay*dt**2
        self.theta += self.w*dt + .5*self.alpha*dt**2