import swerve_module as sm
import swerve_drive as sd
import swerve_drive_2nd_order as sd2
import plot_utils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

physics_dt = .002
loop_dt = .05

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

no_delay_modules_ord2 = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1]))
delay_vel_modules_ord2 = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(vel_del, 4), np.repeat(0, 4)))
delay_head_modules_ord2 = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(0, 4), np.repeat(head_del, 4)))
delay_both_modules_ord2 = list(map(sm.swerve_module_instant, [1, 1, -1, -1], [1, -1, -1, 1], np.repeat(vel_del, 4), np.repeat(head_del, 4)))

no_delay_drive_ord2 = sd2.swerve_drive_2nd_order(no_delay_modules_ord2)
delay_vel_drive_ord2 = sd2.swerve_drive_2nd_order(delay_vel_modules_ord2)
delay_head_drive_ord2 = sd2.swerve_drive_2nd_order(delay_head_modules_ord2)
delay_both_drive_ord2 = sd2.swerve_drive_2nd_order(delay_both_modules_ord2)

drives = [no_delay_drive, delay_vel_drive, delay_head_drive, delay_both_drive, no_delay_drive_ord2, delay_vel_drive_ord2, delay_head_drive_ord2, delay_both_drive_ord2]

fig, axes = plt.subplots(2,2)

kP = np.array([5, 5, 5])

shape_dict = {}
arrow_dict = {}
axes_dict = {}
color_dict = {}
trail_dict = {}

for d in drives[0:4]:
    color_dict[d] = 'C0'
    d.label = 'First Order'

for d in drives[4:]:
    color_dict[d] = 'C2'
    d.label = 'Second Order'

def parse_label(pose, goal_pose):
    return "At: (%.2f, %.2f, %.2f)\nGoal: (%.2f, %.2f, %.2f)" % tuple(np.concatenate((pose, goal_pose)).flatten())

i = 0
flat_ax = axes.flatten()
for ax in flat_ax:
    axes_dict[drives[i]] = ax
    axes_dict[drives[i+4]] = ax
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

for d in drives[4:]:
    plot_utils.set_axes(axes_dict[d], 0, 0, 4)

target_vel = np.array([1,0,2])

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

duration = 10
(ts, pose_dict, vel_dict) = run_sim(duration)

def lookup(drive, t):
    return (pose_dict[(drive, t)], pose_dict[('goal', t)], vel_dict[(drive, t)])

def init():
    for d in drives:
        [p, g, v] = lookup(d,0)
        shape_dict[(d, 'true')] = plot_utils.plot_positions(d, axis = axes_dict[d], pose = p, color = color_dict[d])
        arrow_dict[d] = plot_utils.plot_robot_dir(d, axis = axes_dict[d], pose = p, vels = v, color = color_dict[d])
        trail_dict[d], = axes_dict[d].plot(p[0], p[1], color = color_dict[d])
    
    for d in drives[0:4]:
        shape_dict[(d, 'ideal')] = plot_utils.plot_positions(d, axis = axes_dict[d], linestyle = "--", pose = g, color = 'C1', label = 'Goal')

    mod_objects = list(shape_dict.values()) + list(arrow_dict.values())
    axes[1,1].legend(loc = 'lower right', fontsize = 'xx-small')
    return tuple(mod_objects)

def animate(t):
    for d in drives:
        [p, g, v] = lookup(d,t)
        ax = axes_dict[d]
        shape_dict[(d, 'true')] = plot_utils.plot_positions(d, axis = ax, line = shape_dict[(d, 'true')], pose = p)
        arrow_dict[d] = plot_utils.plot_robot_dir(d, axis = ax, line = arrow_dict[d], pose = p, vels = v)
        plot_utils.set_axes(ax, g[0], g[1], 4)
        trail_dict[d].set_xdata(np.append(trail_dict[d].get_xdata(), p[0]))
        trail_dict[d].set_ydata(np.append(trail_dict[d].get_ydata(), p[1]))

    for d in drives[0:4]:
        shape_dict[(d, 'ideal')] = plot_utils.plot_positions(d, axis = ax, line = shape_dict[(d, 'ideal')], pose = g)
    
    mod_objects = list(shape_dict.values()) + list(arrow_dict.values())
    return tuple(mod_objects)

fps = 30
dt = 1/fps
frame_interval = round(dt/physics_dt)
video_dt = frame_interval*physics_dt
anim = FuncAnimation(fig, animate, init_func=init, frames=ts[::frame_interval], interval=video_dt*1000, blit=False, repeat=False)

#plt.show()
anim.save('swerve_sim_var_delays_2nd_order_with_p.gif', writer='imagemagick')
