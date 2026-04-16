import numpy as np

def average_velocity(worm_positions, t_eval):
    time_elapsed = np.abs(t_eval[-1] - t_eval[0])
    distance = np.abs(worm_positions[0,0,0] - worm_positions[-1,0,0]) # (time, body segment, spatial direction)

    velocity_x = distance / time_elapsed # x component velocity (mm/s)

    return velocity_x