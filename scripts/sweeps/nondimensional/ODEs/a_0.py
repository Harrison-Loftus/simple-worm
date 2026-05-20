import numpy as np
from scipy.integrate import solve_ivp

#---------parameter sweep preferred curvature amplitude-------------
a_0_vals = np.arange(1, 21, 1)

# constants
L = 1.0  # body length mm
R = 40.0e-3  # average body radius mm
r_c = 0.5e-3  # cuticle width mm
E = 10.0 # Youngs modulus N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
k_b = E * I_c # bending viscosity N·mm^2
mu_b = 1e-6 # body viscocity N·mm²·s 


C_N = 5.2e-9 # Normal drag coefficient in water N·s/mm²
C_T = 3.3e-9 # Tangential drag coefficient in water N·s/mm²
C_N_agar = 128e-6 # Normal drag coefficient in agar N·s/mm²
C_T_agar = 3.2e-6 # Tangential drag coefficient in agar N·s/mm²

eta = 0.05 # viscosity of the cuticle N/mm^2

tau_b = mu_b / k_b # mechanical timescale seconds
tau_m = 100.0e-3 # muscle activation timescale seconds
tau_n = 10.0e-3 # neural activity timescale seconds
print("tau",tau_b)
t_c = tau_m

K_water = C_N / C_T
K_agar = C_N_agar / C_T_agar

e = (E * I_c * t_c)/(L**4 * C_T_agar)
print("e",e)

eta_tilde = (eta * I_c)/(L**4 * C_T_agar)
print("eta_tilde", eta_tilde)

N = 120 # number of body segments

N_controls = 6

l = L / N # segment length

range_val = N//N_controls * 4

D_4 = np.zeros((N, N), float)
for i in range(N):
    for j in range(N):
        if i == j:
            if i == 0 or i == N-1:
                D_4[i, j] = 7
            else:
                D_4[i, j] = 6
        elif abs(i - j) == 1:
            if i == 0 or i == N-1:
                D_4[i, j] = -4
            else:
                D_4[i, j] = -4
        elif abs(i - j) == 2:
            if i == 0 or i == N-1:
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

W_p = proprioception_matrix(N, range_val) * (1.0 / (range_val))

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

I_n = np.eye(N)

Kmat = - D_4 # precompute once

s = l * np.arange(N)

def sigma(A, a_0):
    c_m = 10
    c_s = 1
    return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


def F(V):
    return V - V**3 


def ODEs(t, state, a_0):
    
    kappa = state[0:N]

    A_V = state[N:2*N]
    A_D = state[2*N:3*N]
    V_V = state[3*N:4*N]
    V_D = state[4*N:5*N]
    
    epsilon_g = 0.0134

    c_p = 1.0
    epsilon_p = 0.05

    Amp = 22.0

    M = (C_N/mu_b * I_n + D_4) * (tau_b / t_c)
    dkappadt = np.linalg.solve(M, Kmat @ (kappa + Amp * (sigma(A_V, a_0) - sigma(A_D, a_0)))) 
    dA_Vdt = (t_c / tau_m)*(-A_V + V_V - V_D)
    dA_Ddt = (t_c / tau_m)*(-A_D + V_D - V_V) 

    dV_Vdt = (t_c/tau_n)*(F(V_V) + c_p * kappa - epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_V)
    dV_Ddt = (t_c/tau_n)*(F(V_D) - c_p * kappa + epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_D) 
    results = np.concatenate([dkappadt, dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results



