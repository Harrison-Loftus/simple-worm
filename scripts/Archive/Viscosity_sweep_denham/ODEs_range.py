import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.Archive.parameters import *


from scripts.Archive.kymograph import *
from scripts.Archive.hilbert_calc import *
from scripts.Archive.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.Archive.curvature_viewer import view_curvature_pyqtgraph


range_percentage_vals = np.linspace(0.0, 1.0, 11)
range_vals = [int(N * range_percentage) for range_percentage in range_percentage_vals]



def proprioception_matrix(n, m):
    W = np.zeros((n, n))
    m = abs(m)

    for i in range(n):
        j_min = max(0, i - m)
        for j in range(j_min, i):
            W[j, i] = 1

    return W

W_p_vals = [proprioception_matrix(N, range_val) for range_val in range_vals]


W_g = np.zeros((N, N), float)
for i in range(N):
    for j in range(N):
        if i == j:
            if i == 0 or i == N-1:
                W_g[i, j] = -1
            else:
                W_g[i, j] = -2
        elif abs(i - j) == 1:
            W_g[i, j] = 1


def sigma(A):
    c_m = 10
    c_s = 1
    a_0 = 0
    return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


def F(V):
    return V - V**3 


def ODEs(t, state, kappa, W_p):
    
    
    A_V = state[0: N_muscular]
    A_D = state[N_muscular: 2*N_muscular]
    V_V = state[2*N_muscular: 2*N_muscular + N_controls]
    V_D = state[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls]
    
    epsilon_g = 0.0
    epsilon_p = 1.0

    P = W_p @ kappa

    epsilon_g * W_g 
    
    P_regions = np.array_split(P, N_controls)
    P_ctrl = np.array([np.mean(region) for region in P_regions])
    
    kappa_regions = np.array_split(kappa, N_controls)
    kappa_ctrl = np.array([np.mean(region) for region in kappa_regions])
    

    repeat_fact = N_muscular // N_controls
    repeat_fact_neural_to_body = N // N_controls
    
    V_V_ctrl = np.repeat(V_V, repeat_fact)
    V_D_ctrl = np.repeat(V_D, repeat_fact)
    V_V_body = np.repeat(V_V, repeat_fact_neural_to_body)
    V_D_body = np.repeat(V_D, repeat_fact_neural_to_body)

    

    dA_Vdt = (1/tau_m) * (-A_V + sigma(V_V_ctrl - V_D_ctrl))
    dA_Ddt = (1/tau_m) * (-A_D + sigma(V_D_ctrl - V_V_ctrl))

    dV_Vdt = (1/tau_n)*(F(V_V) - epsilon_p * P_ctrl) #+ epsilon_g * W_g @ V_V)
    dV_Ddt = (1/tau_n)*(F(V_D) + epsilon_p * P_ctrl)#+ epsilon_g * W_g @ V_D) 
    results = np.concatenate([dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results    

 