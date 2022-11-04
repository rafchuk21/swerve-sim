import swerve_module as sm
import swerve_drive as sd
import plot_utils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

physics_dt = .0002
loop_dt = .05

no_delay_modules = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1]))
no_delay_drive = sd.swerve_drive(no_delay_modules)

fig, axes = plt.subplots()

kP = np.array([0,0,0])

shape_dict = {}
arrow_dict = {}
axes_dict = {}
text_dict = {}

def parse_label(pose, goal_pose):
    return "At: (%.2f, %.2f, %.2f)\nGoal: (%.2f, %.2f, %.2f)" % tuple(np.concatenate((pose, goal_pose)).flatten())

axes.axis('equal')
axes.grid(True)

fig.suptitle('Simulation Time = %.2fms\n Controller Loop Time = %.2fms\n KP_xy = %.2f, KP_theta = %.2f, KI = KD = 0' % (physics_dt*1000, loop_dt*1000, kP[0], kP[2]))
fig.tight_layout()


plot_utils.set_axes(axes, no_delay_drive.x, no_delay_drive.y, 4)

target_vel = np.array([1,0,1])
txt_y_off = -2.3

def run_sim(duration):
    pose_dict = {}
    vel_dict = {}
    last_commanded = -999;
    ts = np.arange(0, duration, physics_dt)
    for t in ts:
        goal_pose_t =  target_vel*t
        pose_dict[('goal', t)] = goal_pose_t
        if t - last_commanded - loop_dt >= -.00001:
            last_commanded = t
            no_delay_drive.set(target_vel + kP * (goal_pose_t - no_delay_drive.get_pose()))
        
        
        no_delay_drive.step(physics_dt)
        pose_dict[(no_delay_drive, t)] = no_delay_drive.get_pose()
        vel_dict[(no_delay_drive, t)] = no_delay_drive.get_global_vels()

    return (ts, pose_dict, vel_dict)

duration = 5
(ts, pose_dict, vel_dict) = run_sim(duration)

def lookup(drive, t):
    return (pose_dict[(drive, t)], pose_dict[('goal', t)], vel_dict[(drive, t)])

def init():
    [p, g, v] = lookup(no_delay_drive,0)
    shape_dict[(no_delay_drive, 'true')] = plot_utils.plot_positions(no_delay_drive, axis = axes, pose = p)
    shape_dict[(no_delay_drive, 'ideal')] = plot_utils.plot_positions(no_delay_drive, axis = axes, linestyle = "--", pose = g)
    arrow_dict[no_delay_drive] = plot_utils.plot_robot_dir(no_delay_drive, axis = axes, pose = p, vels = v)
    text_dict[no_delay_drive] = axes.text(p[0], p[1]+txt_y_off, parse_label(p, g), horizontalalignment ='center')

    mod_objects = list(shape_dict.values()) + list(arrow_dict.values()) + list(text_dict.values())
    return tuple(mod_objects)

def animate(t):
    [p, g, v] = lookup(no_delay_drive,t)
    shape_dict[(no_delay_drive, 'true')] = plot_utils.plot_positions(no_delay_drive, axis = axes, line = shape_dict[(no_delay_drive, 'true')], pose = p)
    shape_dict[(no_delay_drive, 'ideal')] = plot_utils.plot_positions(no_delay_drive, axis = axes, line = shape_dict[(no_delay_drive, 'ideal')], pose = g)
    arrow_dict[no_delay_drive] = plot_utils.plot_robot_dir(no_delay_drive, axis = axes, line = arrow_dict[no_delay_drive], pose = p, vels = v)
    text_dict[no_delay_drive].set_text(parse_label(p, g))
    text_dict[no_delay_drive].set_x(p[0])
    text_dict[no_delay_drive].set_y(p[1]+txt_y_off)
    plot_utils.set_axes(axes, p[0], p[1], 4)
    
    mod_objects = list(shape_dict.values()) + list(arrow_dict.values()) + list(text_dict.values())
    return tuple(mod_objects)

fps = 30
dt = 1/fps
frame_interval = round(dt/physics_dt)
video_dt = frame_interval*physics_dt
anim = FuncAnimation(fig, animate, init_func=init, frames=ts[::frame_interval], interval=video_dt*1000, blit=False, repeat=False)

plt.show()
#anim.save('swerve_sim.gif', writer='imagemagick')
