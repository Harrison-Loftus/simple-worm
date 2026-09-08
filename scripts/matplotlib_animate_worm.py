import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def create_worm_animation(worm_positions, dt):

    # worm_positions has shape (N_frames, N_points, 2)
    N_frames = worm_positions.shape[0]

    fig, ax = plt.subplots()

    # initial worm configuration
    line, = ax.plot(worm_positions[0, :, 0], worm_positions[0, :, 1], "k-")


    ax.set_xlim(worm_positions[:, :, 0].min() - 0.1, worm_positions[:, :, 0].max() + 0.1)

    ax.set_ylim(worm_positions[:, :, 2].min() - 0.1, worm_positions[:, :, 2].max() + 0.1)

    ax.set_aspect("equal")

    def update(frame):
        line.set_data(worm_positions[frame, :, 0], worm_positions[frame, :, 2])
        return (line,)

    anim = animation.FuncAnimation(fig, update, frames=N_frames, interval=dt * 1000, blit=True)

    plt.close(fig)
    return anim
