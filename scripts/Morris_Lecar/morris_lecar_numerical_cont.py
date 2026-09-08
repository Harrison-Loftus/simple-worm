from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
import numpy as np

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

#------Parameters------
C    = 20.0
V_1  = -1.2
V_2  = 18.0
V_3  = 12.0
V_4  = 17.4

g_Ca = 4.0
g_K  = 8.0
g_L  = 2.0

V_Ca = 120.0
V_K  = -84.0
V_L  = -60.0

phi  = 1 / 15


#------Functions------
def m_inf(V):
    return 0.5 * (1 + np.tanh((V - V_1)/(V_2)))

def w_inf(V):
    return 0.5 * (1 + np.tanh((V - V_3)/(V_4)))

def tau_w(V):
    return 1/(np.cosh((V - V_3) / (2 * V_4)))

#------ODE------
def morris_lecar(t, state, I):
    V = state[0]
    w = state[1]

    dV_dt = (- g_Ca * m_inf(V) * (V - V_Ca) - g_K * w * (V - V_K) - g_L * (V - V_L) + I ) / C
    dw_dt = phi * (w_inf(V) - w) / (tau_w(V))

    return np.array([dV_dt, dw_dt])


def residual(state, I):
    return morris_lecar(0.0, state, I)

I_vals = np.linspace(0.0, 200.0, 100)

max_eig_vals = []
max_imag_vals = []

x_prev = np.array([0.0,0.2])
equilibrias = []

for i, I in enumerate(I_vals):
    print("I: ", np.round(I, 2))
    initial_state = [0.0, 0.2]
    sol = root(residual, x_prev, args=(I), method='hybr')
    
    x_prev = sol.x
    equilibrias.append(sol.x[0])

    state_eq = sol.x
    J = approx_derivative(lambda x: residual(x, I), state_eq)

    eigvals = np.linalg.eigvals(J)
    max_eig_vals.append(np.max(np.real(eigvals)))
    print("Max real part of eigenvalues: ", np.max(np.real(eigvals)))
    
    max_imag_vals.append(np.max(np.imag(eigvals)))
    print("Max imaginary part of eigenvalues: ", np.max(np.imag(eigvals)))

plt.figure(figsize=(10, 6))
plt.scatter(I_vals, max_eig_vals, marker='o', color='black')
plt.xlabel(r'$I$')
plt.ylabel(r'Max real part of eigenvalues')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(I_vals, max_imag_vals, marker='o', color='black')
plt.xlabel(r'$I$')
plt.ylabel(r'Max imaginary part of eigenvalues')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(I_vals, equilibrias, marker='o', color='black')
plt.xlabel(r'$I$')
plt.ylabel(r'Equilibrium voltage $V$')
plt.title(r'Equilibrium voltage $V$ vs $I$')
plt.ylim(ymin=-60, ymax=40)

plt.show()

