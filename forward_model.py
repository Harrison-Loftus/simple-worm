import numpy as np
from fenics import *

from simple_worm.controls import (
    ControlsFenics,
    ControlsNumpy,
)
from simple_worm.worm import Worm

# Parameters
N = 10  # Number of body points - recommend ~100
T = 1.0  # Final time - recommend several undulations
dt = 0.1  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)


def example1():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """
    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    u = SpatialCoordinate(worm.V)[0]
    control = Function(worm.V)

    # holder for other control directions
    zero = Function(worm.V)
    zero.vector = 0

    # wave parameters
    A = 10.0
    lam = 1.5
    omega = 1.0

    # specific forcing function
    def alpha_forcing(t):
        return A * sin(2.0 * pi * lam * u - 2 * pi * omega * t)

    t = 0.0
    while t < T:
        t += dt

        # update control
        project(alpha_forcing(t), function=control)

        # solve
        ret = worm.update_solution(ControlsFenics(alpha=control, beta=zero, gamma=zero))

        # output variables as 'fenics functions
        # x = ret.x
        # curvature = ret.alpha

        ret_np = ret.to_numpy()
        x_np = ret_np.x
        curvature_np = ret_np.alpha

        print(f"{x_np=}")
        print(f"{curvature_np=}")


def example2():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """
    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    # controls holder but we will only use 7 different values
    N_controls = 7
    control = np.empty(N)

    # holder for other control directions
    zeroN = np.zeros(N)
    zeroNm = np.zeros(N - 1)

    # wave parameters
    A = 10.0
    lam = 1.5
    omega = 1.0

    # specific forcing function
    def alpha_forcing(t, j):
        # j is point in numpy array
        # j_control is the corresponding control point
        j_control = j * N // N_controls
        # u_control is center point of control region
        u_control = (j_control + 0.5) / N_controls
        return A * np.sin(2.0 * np.pi * lam * u_control - 2 * np.pi * omega * t)

    t = 0.0
    while t < T:
        t += dt

        # update control
        control[:] = [alpha_forcing(t, j) for j in range(N)]

        # solve
        C = ControlsNumpy(alpha=control, beta=zeroN, gamma=zeroNm)
        ret = worm.update_solution(C.to_fenics(worm))

        # output variables as 'fenics functions
        # x = ret.x
        # curvature = ret.alpha

        ret_np = ret.to_numpy()
        x_np = ret_np.x
        curvature_np = ret_np.alpha

        print(f"{x_np=}")
        print(f"{curvature_np=}")


if __name__ == "__main__":
    example1()
    example2()
