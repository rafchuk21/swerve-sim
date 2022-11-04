import swerve_drive as sd
import matplotlib.pyplot as plt
import numpy as np

def plot_positions(swerve: sd.swerve_drive, axis, line = None, closed = True, linestyle = None, pose = None, color = None, label = None):
    module_coords = swerve.get_module_coords() if (np.array(pose) == None).any() else swerve.get_module_coords(pose)

    xs = module_coords[0,:]
    ys = module_coords[1,:]

    if closed:
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])

    if line == None:
        line, = axis.plot(xs, ys, linestyle = linestyle, color = color, label = swerve.label if label == None else label)
    else:
        line.set_xdata(xs)
        line.set_ydata(ys)

    return line

def set_axes(ax, x, y, sz):
    x_limits = (x-sz, x+sz)
    y_limits = (y-sz, y+sz)
    x_ticks = np.around(np.linspace(*x_limits, round(2*sz+1)).flatten())
    y_ticks = np.around(np.linspace(*y_limits, round(2*sz+1)).flatten())
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    #ax.set(xlim = x_limits, ylim = y_limits)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

def plot_robot_dir(swerve: sd.swerve_drive, axis, line = None, pose = None, vels = None, color = None):
    pose = swerve.get_pose() if (np.array(pose) == None).any() else pose
    vels = swerve.get_global_vels() if (np.array(vels) == None).any() else vels

    [x, y, theta] = pose.flatten()
    [vx, vy, w] = vels.flatten()
    if line == None:
        line = axis.arrow(x, y, vx, vy, width=.05, length_includes_head = True, color = color)
    else:
        line.set_data(x=x, y=y, dx=vx, dy=vy)

    return line
