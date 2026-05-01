import numpy as np
from scipy.integrate import solve_ivp


# constants
L = 1.0  # body length mm
R = 40.0e-3  # average body radius mm
r_c = 0.5e-3  # cuticle width mm
E_kPa = 10000 # young's modulus kPa
E = E_kPa * 1e-3 # convert to N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
k_b = E * I_c # bending viscosity N·mm^2
mu_b = 1e-7 # body viscocity N·mm²·s 
mu_f_mPas = 1.0 # fluid viscocity mPa·s
mu_f = mu_f_mPas * 1e-9 # N·s/mm^2
C_N = 5.2 * mu_f # normal drag coefficient N·s/mm^2 

# carter has K_v at 3.4mPas, Ranner has K_v at 5.2e-3 kg /(ms)

tau_b = mu_b / k_b # mechanical timescale seconds
tau_m = 100.0e-3 # muscle activation timescale seconds
tau_n = 10.0e-3 # neural activity timescale seconds
print("tau",tau_b)
t_c = tau_m

eta = mu_b/I_c
print("eta", eta)

C_N_agar = 2.8e4 * 1e-9
C_T = C_N * 3.3/5.2
C_T_agar = C_N_agar * 1.0 / 40.0

eta_tilde = mu_b / (L**4 * C_T_agar)
e = (k_b * tau_m) / (L**4 * C_T_agar)
print(eta_tilde)
print(e)

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

Kmat = -L**5 * D_4 # precompute once

s = l * np.arange(N)

def sigma(A):
    c_m = 10
    c_s = 1
    a_0 = 2
    return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


def F(V):
    return V - V**3 


def ODEs(t, state):
    
    kappa = state[0:N]

    A_V = state[N:2*N]
    A_D = state[2*N:3*N]
    V_V = state[3*N:4*N]
    V_D = state[4*N:5*N]
    
    epsilon_g = 0.0134
    epsilon_p = 0.05
    c_p = 1.0
    
    A = 40.0 # amplitude

    M = (C_N/mu_b * I_n + D_4) * L**5 * (tau_b / tau_m)
    dkappadt = np.linalg.solve(M, Kmat @ (kappa + ((sigma(A_V) - sigma(A_D)) * A))) 
    dA_Vdt = (-A_V + V_V - V_D)
    dA_Ddt = (-A_D + V_D - V_V) 

    dV_Vdt = (tau_m/tau_n)*(F(V_V) + c_p * kappa - epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_V)
    dV_Ddt = (tau_m/tau_n)*(F(V_D) - c_p * kappa + epsilon_p * W_p @ kappa + epsilon_g * W_g @ V_D) 
    results = np.concatenate([dkappadt, dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results



