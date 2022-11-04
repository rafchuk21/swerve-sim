import swerve_module as sm
import swerve_drive as sd
import plot_utils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

physics_dt = .01
loop_dt = 5*physics_dt

no_delay_modules = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1]))
no_delay_drive = sd.swerve_drive(no_delay_modules)

vel_del = 1
head_del = 1
delay_vel_modules = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(vel_del, 4), np.repeat(0, 4)))
delay_vel_drive = sd.swerve_drive(delay_vel_modules)

delay_head_modules = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(0, 4), np.repeat(head_del, 4)))
delay_head_drive = sd.swerve_drive(delay_head_modules)

delay_both_modules = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(vel_del, 4), np.repeat(head_del, 4)))
delay_both_drive = sd.swerve_drive(delay_both_modules)

drives = [no_delay_drive, delay_vel_drive, delay_head_drive, delay_both_drive]

fig, axes = plt.subplots(2,2)

kP = np.array([0,0,0])

shape_dict = {}
arrow_dict = {}
axes_dict = {}
text_dict = {}

def parse_label(pose, goal_pose):
    return "At: (%.2f, %.2f, %.2f)\nGoal: (%.2f, %.2f, %.2f)" % tuple(np.concatenate((pose, goal_pose)).flatten())

i = 0
flat_ax = axes.flatten()
for ax in flat_ax:
    axes_dict[drives[i]] = ax
    ax.axis('equal')
    ax.grid(True)
    i += 1

for ax in flat_ax[range(2)]:
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

for ax in flat_ax[[1,3]]:
    ax.yaxis.tick_right()

fig.suptitle('Simulation Time = %dms\n Controller Loop Time = %dms\n KP_xy = %.2f, KP_theta = %.2f, KI = KD = 0' % (physics_dt*1000, loop_dt*1000, kP[0], kP[2]))
axes[0,0].set_xlabel('Velocity Delay = 0ms')
axes[0,0].set_ylabel('Heading Delay = 0ms')
axes[0,1].set_xlabel('Velocity Delay = %dms' % (vel_del*loop_dt*1000))
axes[1,0].set_ylabel('Heading Delay = %dms' % (head_del*loop_dt*1000))
fig.tight_layout()

for d in drives:
    plot_utils.set_axes(axes_dict[d], d.x, d.y, 4)

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
            for d in drives:
                d.set(target_vel + kP * (goal_pose_t - d.get_pose()))
        
        for d in drives:
            d.step(physics_dt)
            pose_dict[(d, t)] = d.get_pose()
            vel_dict[(d, t)] = d.get_global_vels()

    return (ts, pose_dict, vel_dict)

duration = 5
(ts, pose_dict, vel_dict) = run_sim(duration)

def lookup(drive, t):
    return (pose_dict[(drive, t)], pose_dict[('goal', t)], vel_dict[(drive, t)])

def init():
    for d in drives:
        [p, g, v] = lookup(d,0)
        shape_dict[(d, 'true')] = plot_utils.plot_positions(d, axis = axes_dict[d], pose = p)
        shape_dict[(d, 'ideal')] = plot_utils.plot_positions(d, axis = axes_dict[d], linestyle = "--", pose = g)
        arrow_dict[d] = plot_utils.plot_robot_dir(d, axis = axes_dict[d], pose = p, vels = v)
        text_dict[d] = axes_dict[d].text(p[0], p[1]+txt_y_off, parse_label(p, g), horizontalalignment ='center')

    mod_objects = list(shape_dict.values()) + list(arrow_dict.values()) + list(text_dict.values())
    return tuple(mod_objects)

def animate(t):
    for d in drives:
        [p, g, v] = lookup(d,t)
        ax = axes_dict[d]
        shape_dict[(d, 'true')] = plot_utils.plot_positions(d, axis = ax, line = shape_dict[(d, 'true')], pose = p)
        shape_dict[(d, 'ideal')] = plot_utils.plot_positions(d, axis = ax, line = shape_dict[(d, 'ideal')], pose = g)
        arrow_dict[d] = plot_utils.plot_robot_dir(d, axis = ax, line = arrow_dict[d], pose = p, vels = v)
        text_dict[d].set_text(parse_label(p, g))
        text_dict[d].set_x(p[0])
        text_dict[d].set_y(p[1]+txt_y_off)
        plot_utils.set_axes(ax, p[0], p[1], 4)
    
    mod_objects = list(shape_dict.values()) + list(arrow_dict.values()) + list(text_dict.values())
    return tuple(mod_objects)

frame_interval = 5
video_dt = frame_interval*physics_dt
anim = FuncAnimation(fig, animate, init_func=init, frames=ts[::frame_interval], interval=video_dt*1000, blit=False, repeat=False)

plt.show()
#anim.save('swerve_sim.gif', writer='imagemagick')
