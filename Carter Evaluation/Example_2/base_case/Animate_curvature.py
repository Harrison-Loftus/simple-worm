import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_curvature_animation(n, L, kappa):
    
  #--------------Animation---------------

    plt.rcParams['animation.embed_limit'] = 100 

    N = int(n)
    l = L / N
    modules = (np.arange(N)) * l / L

    N_frames = kappa.shape[1]

    fig, ax = plt.subplots()
    ax.set_xlim(-0.0 - 1/n, 1 + 1/n)
    ax.set_ylim(kappa.min() - 0.5, kappa.max() + 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel(r'\kappa')

    line, = ax.plot(modules, kappa[:, 0], "-o", markersize=8)

    def update(frame):
        line.set_ydata(kappa[:, frame])
        return line,

    anim = animation.FuncAnimation(
        fig, update, frames=N_frames, interval=30, blit=False
    )

    plt.close(fig)
    return anim