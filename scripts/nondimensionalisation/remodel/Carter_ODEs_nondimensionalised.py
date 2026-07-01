import numpy as np
from scipy.integrate import solve_ivp


# constants
L = 1.0  # body length mm
R = 40.0e-3  # average body radius mm
r_c = 0.5e-3  # cuticle width mm
E = 10.0 # Youngs modulus N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
k_b = E * I_c # bending modulus N·mm^2
mu_b = 1e-6 # body viscocity N·mm²·s 


C_N = 5.2e-9 # Normal drag coefficient in water N·s/mm²
C_T = 3.3e-9 # Tangential drag coefficient in water N·s/mm²
C_N_agar = 128e-6 # Normal drag coefficient in agar N·s/mm²
C_T_agar = 3.2e-6 # Tangential drag coefficient in agar N·s/mm²

eta = 0.05 # viscosity of the cuticle N·s/mm^2
print("eta: ", eta)

tau_b = mu_b / k_b # mechanical timescale seconds
tau_m = 100.0e-3 # muscle activation timescale seconds
tau_n = 10.0e-3 # neural activity timescale seconds
print("tau",tau_b)
t_c = 1.0

K_water = C_N / C_T
K_agar = C_N_agar / C_T_agar

e = (E * I_c * t_c)/(L**4 * C_T_agar) / 2
print("e",e)

eta_tilde = (eta * I_c)/(L**4 * C_T_agar) / 2
print("eta_tilde", eta_tilde)

N = 120 # number of body segments

N_controls = 6

l = L / N # segment length

range_val = N//N_controls * 4

def controls_to_body(A_V, A_D):
    kappa_control = sigma(A_V) - sigma(A_D)
    kappa_body = np.zeros(N)

    for j in range(N):
        j_control = (j * N_controls) // N
        kappa_body[j] = kappa_control[j_control]

    return kappa_body

def body_to_controls(kappa):
    kappa_coarse = np.zeros(N_controls)

    for i in range(N_controls):
        start = int(i * N / N_controls)
        end = int((i + 1) * N / N_controls)
        kappa_coarse[i] = np.mean(kappa[start:end])

    return kappa_coarse


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


W_p_controls = np.zeros((N_controls, N))

for i in range(N_controls):
    start = int(max(0, (i - 4) * N / N_controls))
    end = int(i * N / N_controls)

    if end > start:
        W_p_controls[i, start:end] = 1.0 / (end - start)


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

W_g_controls = np.zeros((N_controls, N_controls))

for i in range(N_controls):
    if i > 0:
        W_g_controls[i, i-1] = 1
    if i < N_controls - 1:
        W_g_controls[i, i+1] = 1
    W_g_controls[i, i] = -2

I_n = np.eye(N)

Kmat = - D_4 # precompute once

s = l * np.arange(N)

def sigma(A):
    c_m = 10.0 
    c_s = 1.0
    a_0 = 2.0
    return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


def F(V):
    return V - V**3 


def ODEs(t, state):
    
    
    kappa = state[0:N]

    A_V = state[N:N+N_controls]
    A_D = state[N+N_controls:N+2*N_controls]
    V_V = state[N+2*N_controls:N+3*N_controls]
    V_D = state[N+3*N_controls:N+4*N_controls]


    a = K_water 
    b = np.sqrt(a / K_water)

    epsilon_g = 0.0134
    epsilon_p = 0.05 #* b
    c_p = 1.0
    
    Amp = 50.0 # amplitude


    M = (C_N/mu_b * I_n + D_4) * (tau_b / t_c)
    
    kappa_force = controls_to_body(A_V, A_D)
    kappa_coarse = body_to_controls(kappa)

    dkappadt = np.linalg.solve(M, Kmat @ (kappa + Amp * kappa_force))

    dA_Vdt = (t_c / (5*b*tau_m))*(-A_V + V_V - V_D)
    dA_Ddt = (t_c / (5*b*tau_m))*(-A_D + V_D - V_V) 

    dV_Vdt = (t_c/(b*tau_n))*(F(V_V) + c_p * kappa_coarse - epsilon_p * W_p_controls @ kappa + epsilon_g * W_g_controls @ V_V)
    dV_Ddt = (t_c/(b*tau_n))*(F(V_D) - c_p * kappa_coarse + epsilon_p * W_p_controls @ kappa + epsilon_g * W_g_controls @ V_D) 
    results = np.concatenate([dkappadt, dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results



