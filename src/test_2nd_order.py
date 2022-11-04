import numpy as np
import swerve_module as sm
import swerve_drive as sd1
import swerve_drive_2nd_order as sd2
import plot_utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from math import atan2
from numpy import sin, cos



vel_delay = 0
head_delay = 0

modules = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(vel_delay, 4), np.repeat(head_delay, 4)))
goal = sd2.swerve_drive_2nd_order(list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(vel_delay, 4), np.repeat(head_delay, 4))))
first_order = sd1.swerve_drive(list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(vel_delay, 4), np.repeat(head_delay, 4))))
second_order = sd2.swerve_drive_2nd_order(list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(vel_delay, 4), np.repeat(head_delay, 4))))

first_order.label = 'First Order'
second_order.label = 'Second Order'

drives = [first_order, second_order]

physics_dt = .0002
loop_dt = {}
loop_dt[goal] = physics_dt
loop_dt[first_order] = .05
loop_dt[second_order] = .05

vel = np.array([1,0,2])
accel = np.array([0.2,0,0.5])

fig, axes = plt.subplots()

kP = np.array([0, 0, 0])

shape_dict = {}
arrow_dict = {}
axes_dict = {}
text_dict = {}
trail_dict = {}

color_dict = {}
color_dict[first_order] = 'C1'
color_dict[second_order] = 'C2'

def parse_label(pose, goal_pose):
    return "At: (%.2f, %.2f, %.2f)\nGoal: (%.2f, %.2f, %.2f)" % tuple(np.concatenate((pose, goal_pose)).flatten())

axes.axis('equal')
axes.grid(True)

fig.suptitle('Simulation Time = %.2fms\n Controller Loop Time = %.2fms\n KP_xy = %.2f, KP_theta = %.2f, KI = KD = 0' % (physics_dt*1000, loop_dt[drives[-1]]*1000, kP[0], kP[2]))
fig.tight_layout()

plot_utils.set_axes(axes, drives[0].x, drives[0].y, 4)

txt_y_off = -2.3

def run_sim(duration):
    pose_dict = {}
    vel_dict = {}
    mvel_dict = {}
    last_commanded = {}
    goal_pose = np.array([0,0,0])
    
    for d in drives:
        last_commanded[d] = -999

    ts = np.arange(0, duration, physics_dt)
    for t in ts:
        goal_pose_t =  vel*t + .5*accel*t**2
        pose_dict[('goal', t)] = goal_pose_t
        
        for d in drives:
            if t - last_commanded[d] - loop_dt[d] >= -.00001:
                last_commanded[d] = t
                d.set(vel + t*accel + kP * (goal_pose_t - d.get_pose()), accel)

            d.step(physics_dt)
            pose_dict[(d, t)] = d.get_pose()
            vel_dict[(d, t)] = d.get_global_vels()
            mvel_dict[(d, t)] = d.get_module_vels()


    return (ts, pose_dict, vel_dict, mvel_dict)

duration = 10
(ts, pose_dict, vel_dict, mvel_dict) = run_sim(duration)

def lookup(drive, t):
    return (pose_dict[(drive, t)], pose_dict[('goal', t)], vel_dict[(drive, t)])

def init():
    for d in drives:
        [p, g, v] = lookup(d,0)
        shape_dict[(d, 'true')] = plot_utils.plot_positions(d, axis = axes, pose = p, color = color_dict[d])
        arrow_dict[d] = plot_utils.plot_robot_dir(d, axis = axes, pose = p, vels = v, color = color_dict[d])
        #text_dict[d] = axes.text(p[0], p[1]+txt_y_off, parse_label(p, g), horizontalalignment ='center')
        trail_dict[d], = axes.plot(p[0], p[1], color = color_dict[d])

    [p,g,v] = lookup(drives[0],0)
    shape_dict[(d, 'ideal')] = plot_utils.plot_positions(d, axis = axes, linestyle = "--", pose = g, label = 'Goal')
    mod_objects = list(shape_dict.values()) + list(arrow_dict.values()) + list(text_dict.values())
    axes.legend()
    return tuple(mod_objects)

def animate(t):
    for d in drives:
        [p, g, v] = lookup(d,t)
        ax = axes
        shape_dict[(d, 'true')] = plot_utils.plot_positions(d, axis = ax, line = shape_dict[(d, 'true')], pose = p)
        arrow_dict[d] = plot_utils.plot_robot_dir(d, axis = ax, line = arrow_dict[d], pose = p, vels = v)
        #text_dict[d].set_text(parse_label(p, g))
        #text_dict[d].set_x(p[0])
        #text_dict[d].set_y(p[1]+txt_y_off)
        plot_utils.set_axes(ax, p[0], p[1], 4)
        trail_dict[d].set_xdata(np.append(trail_dict[d].get_xdata(), p[0]))
        trail_dict[d].set_ydata(np.append(trail_dict[d].get_ydata(), p[1]))
    
    [p,g,v] = lookup(drives[0], t)
    shape_dict[(d, 'ideal')] = plot_utils.plot_positions(d, axis = axes, line = shape_dict[(d, 'ideal')], pose = g)

    mod_objects = list(shape_dict.values()) + list(arrow_dict.values()) + list(text_dict.values())
    return tuple(mod_objects)

fps = 30
dt = 1/fps
frame_interval = round(dt/physics_dt)
video_dt = frame_interval*physics_dt
anim = FuncAnimation(fig, animate, init_func=init, frames=ts[::frame_interval], interval=video_dt*1000, blit=False, repeat=False)

#plt.show()
anim.save('swerve_sim_with_accel.gif', writer='imagemagick')