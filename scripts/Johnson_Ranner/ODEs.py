import numpy as np


# constants
L = 1.0  # body length mm
tau_m = 100.0e-3 # muscle activation timescale seconds
tau_n = 10.0e-3 # neural activity timescale seconds

N = 96 # points of the worms body dicretised
N_controls = 6 # number of neurons approximated on each side
N_muscular = 48 # number of body wall muscles on each side approx

E = 0.1 # Youngs modulus N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
r_c = 0.5e-3  # cuticle width mm
eta = 0.05 # viscosity of the cuticle N·s/mm^2

C_N = 5.2e-9 # Normal drag coefficient in water N·s/mm²
C_T = 3.3e-9 # Tangential drag coefficient in water N·s/mm²

C_N_agar = 128e-6 # Normal drag coefficient in agar N·s/mm²
C_T_agar = 3.2e-6 # Tangential drag coefficient in agar N·s/mm²

K_water = C_N / C_T
K_agar = C_N_agar / C_T_agar

t_c = 1.0

e = (E * I_c * t_c)/(L**4 * C_T_agar) / 2
eta_tilde = (eta * I_c)/(L**4 * C_T_agar) / 2 
print(e)
print(eta_tilde)

l = L / N # segment length


range_percentage = 0.75

range_val = int(N * range_percentage)


def proprioception_matrix(n, m):
    W = np.zeros((n, n))
    m = abs(m)

    for i in range(n):
        j_min = max(0, i - m)
        for j in range(j_min, i):
            W[j, i] = 1

    return W

W_p = proprioception_matrix(N, range_val) 

W_p = W_p / np.maximum(W_p.sum(axis=1, keepdims=True), 1)


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


def ODEs(t, state, kappa):
    
    
    A_V = state[0: N_muscular]
    A_D = state[N_muscular: 2*N_muscular]
    V_V = state[2*N_muscular: 2*N_muscular + N_controls]
    V_D = state[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls]
    
    epsilon_g = 0.0
    epsilon_p = 1.0 
    
    P = W_p @ kappa
    
    P_regions = np.array_split(P, N_controls)
    P_ctrl = np.array([np.mean(region) for region in P_regions])
    
    repeat_fact = N_muscular // N_controls
    repeat_fact_neural_to_body = N // N_controls
    
    V_V_ctrl = np.repeat(V_V, repeat_fact)
    V_D_ctrl = np.repeat(V_D, repeat_fact)
    V_V_body = np.repeat(V_V, repeat_fact_neural_to_body)
    V_D_body = np.repeat(V_D, repeat_fact_neural_to_body)

    

    dA_Vdt = (1/tau_m) * (-A_V + sigma(V_V_ctrl - V_D_ctrl))
    dA_Ddt = (1/tau_m) * (-A_D + sigma(V_D_ctrl - V_V_ctrl))

    dV_Vdt = (1/tau_n)*(F(V_V) - epsilon_p * P_ctrl) #+ epsilon_g * W_g @ V_V)
    dV_Ddt = (1/tau_n)*(F(V_D) + epsilon_p * P_ctrl) #+ epsilon_g * W_g @ V_D) 
    results = np.concatenate([dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results    

 