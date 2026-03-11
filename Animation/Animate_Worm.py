import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_worm_animation(worm_positions, dt, plane='xz'):
    
    if plane == 'xz':
        i, j = 0, 2
    elif plane == 'xy':
        i, j = 0, 1


    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-', lw=2)

    # Collect axis limits
    all_pts = np.concatenate(worm_positions)
    xmin, xmax = np.min(all_pts[:, i]), np.max(all_pts[:, i])
    ymin, ymax = np.min(all_pts[:, j]), np.max(all_pts[:, j])

    ax.set_xlim(xmin - 0.1, xmax + 0.1)
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_aspect('equal')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        coords = worm_positions[frame]
        x = coords[:, i]
        y = coords[:, j]
        line.set_data(x, y)
        return line,

    anim = animation.FuncAnimation(
        fig, update, frames=len(worm_positions),
        init_func=init, blit=True, interval=dt*1000
    )

    return anim