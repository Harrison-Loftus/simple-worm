import numpy as np


# constants
L = 1.0  # body length mm
tau_m = 100.0e-3 # muscle activation timescale seconds
tau_n = 10.0e-3 # neural activity timescale seconds

N_vals = [6 * i for i in range(1,17)]

N_controls = 6

E = 1.0 # Youngs modulus N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
r_c = 0.5e-3  # cuticle width mm
eta = 0.05 # viscosity of the cuticle N·s/mm^2

C_N = 5.2e-9 # Normal drag coefficient in water N·s/mm²
C_T = 3.3e-9 # Tangential drag coefficient in water N·s/mm²

C_N_agar = 128e-6 # Normal drag coefficient in agar N·s/mm²
C_T_agar = 3.2e-6 # Tangential drag coefficient in agar N·s/mm²

K_water = C_N / C_T

t_c = 1.0

e = (E * I_c * t_c)/(L**4 * C_T_agar) / 2
eta_tilde = (eta * I_c)/(L**4 * C_T_agar) / 2



def make_matrices(N):
    range_percentage = 0.25

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
    
    return W_p, W_g


def sigma(A):
    c_m = 10
    c_s = 1
    a_0 = 2
    return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


def F(V):
    return V - V**3 


def ODEs(t, state, kappa, N):
    
    W_p, W_g = make_matrices(N)

    A_V = state[0:N]
    A_D = state[N:2*N]
    V_V = state[2*N:3*N]
    V_D = state[3*N:4*N]
    
    epsilon_g = 0.0
    epsilon_p = 1.0
    c_p = 0.0
    

    dA_Vdt = (1/tau_m)*(-A_V + V_V - V_D)
    dA_Ddt = (1/tau_m)*(-A_D + V_D - V_V) 

    dV_Vdt = (1/tau_n)*(F(V_V) + c_p * kappa - epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_V)
    dV_Ddt = (1/tau_n)*(F(V_D) - c_p * kappa + epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_D) 
    results = np.concatenate([dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results    

 