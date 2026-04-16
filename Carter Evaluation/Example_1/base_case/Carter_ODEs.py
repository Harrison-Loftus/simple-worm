import numpy as np
from scipy.integrate import solve_ivp

# constants
L = 1.0  # body length mm
R = 40.0e-3  # average body radius mm
r_c = 0.5e-3  # cuticle width mm
E_kPa = 1000 # young's modulus kPa
E = E_kPa * 1e-3 # convert to N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
k_b = E * I_c # bending viscosity N·mm^2
mu_b = 1.3e-7 # body viscocity N·mm²·s 
mu_f_mPas = 1.0 # fluid viscocity mPa·s
mu_f = mu_f_mPas * 1e-9 # N·s/mm^2
C_N = 3.4 * mu_f # normal drag coefficient N·s/mm^2
tau_b = mu_b / k_b # mechanical timescale seconds
tau_m = 100.0e-3 # muscle activation timescale seconds
tau_n = 10.0e-3 # neural activity timescale seconds
n = 48 # number of body segments
l = L / n # segment length

range_val = n//6 * 4

D_4 = np.zeros((n, n), float)
for i in range(n):
    for j in range(n):
        if i == j:
            if i == 0 or i == n-1:
                D_4[i, j] = 7
            else:
                D_4[i, j] = 6
        elif abs(i - j) == 1:
            if i == 0 or i == n-1:
                D_4[i, j] = -4
            else:
                D_4[i, j] = -4
        elif abs(i - j) == 2:
            if i == 0 or i == n-1:
                D_4[i, j] = 1
            else:
                D_4[i, j] = 1

D_4 = D_4 * (1/(l**4))

def proprioception_matrix(n, m):
    W = np.zeros((n, n))
    m = abs(m)

    for i in range(n):
        j_min = max(0, i - m)
        for j in range(j_min, i):
            W[i, j] = 1

    return W

W_p = proprioception_matrix(n, range_val) * (1.0 / range_val)

W_g = np.zeros((n, n), float)
for i in range(n):
    for j in range(n):
        if i == j:
            if i == 0 or i == n-1:
                W_g[i, j] = -1
            else:
                W_g[i, j] = -2
        elif abs(i - j) == 1:
            W_g[i, j] = 1

I_n = np.eye(n)

Kmat = -k_b * D_4 # precompute once

s = l * np.arange(n)

def sigma(A):
    c_m = 10
    c_s = 1
    a_0 = 2
    return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


def F(V):
    return V - V**3 


def ODEs(t, state):
    
    kappa = state[0:n]

    A_V = state[n:2*n]
    A_D = state[2*n:3*n]
    V_V = state[3*n:4*n]
    V_D = state[4*n:5*n]
    
    epsilon_g = 0.0134
    epsilon_p = 0.1
    c_p = 1.0
    
    M = C_N * I_n + mu_b * D_4
    dkappadt = np.linalg.solve(M, Kmat @ (kappa + (sigma(A_V) - sigma(A_D))*30))
    dA_Vdt = (1/tau_m)*(-A_V + V_V - V_D)
    dA_Ddt = (1/tau_m)*(-A_D + V_D - V_V) 

    dV_Vdt = (1/tau_n)*(F(V_V) + c_p * kappa - epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_V)
    dV_Ddt = (1/tau_n)*(F(V_D) - c_p * kappa + epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_D) 
    results = np.concatenate([dkappadt, dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results



