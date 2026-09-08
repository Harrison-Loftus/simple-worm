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


x_guess_vals = [[i, j] for i in np.linspace(-60.0, 40.0, 100) for j in np.linspace(0.0, 1.0, 10)]

I_vals = np.linspace(-10.0, 200.0, 100)

max_eig_vals = np.full([len(x_guess_vals), len(I_vals)], np.nan)
max_imag_vals = np.full([len(x_guess_vals), len(I_vals)], np.nan)
equilibrias = np.full([len(x_guess_vals), len(I_vals)], np.nan)


for i, x_guess in enumerate(x_guess_vals):

    for j, I in enumerate(I_vals):

        sol = root(residual, x_guess, args=(I,), method='lm')

        res = residual(sol.x, I)

        if np.linalg.norm(res) > 1e-8:
            continue

        if not sol.success:
           continue
        
        state_eq = sol.x

        J = approx_derivative(lambda x: residual(x, I), state_eq)

        eigvals = np.linalg.eigvals(J)

        max_eig_vals[i, j] = np.max(np.real(eigvals))
        
        max_imag_vals[i, j] = np.max(np.imag(eigvals))

        equilibrias[i, j] = state_eq[0]
        
plt.figure(figsize=(10, 6))
for i, x_guess in enumerate(x_guess_vals):
    plt.scatter(I_vals, max_eig_vals[i, :], marker='o', color='black', alpha=0.1)
plt.xlabel(r'$I$')
plt.ylabel(r'Max real part of eigenvalues')
plt.show()

plt.figure(figsize=(10, 6))
for i, x_guess in enumerate(x_guess_vals):
    plt.scatter(I_vals, max_imag_vals[i, :], marker='o', color='black', alpha=0.1)
plt.xlabel(r'$I$')
plt.ylabel(r'Max imaginary part of eigenvalues')
plt.show()

plt.figure(figsize=(10, 6))

for i, x_guess in enumerate(x_guess_vals):

    stable = max_eig_vals[i, :] < 0
    unstable = max_eig_vals[i, :] > 0

    plt.scatter(
        I_vals[stable],
        equilibrias[i, stable],
        color='blue',
        alpha=0.1,
        s=10
    )

    plt.scatter(
        I_vals[unstable],
        equilibrias[i, unstable],
        color='red',
        alpha=0.1,
        s=10
    )

plt.xlabel(r'$I$')
plt.ylabel(r'Equilibrium voltage $V$')
plt.title(r'Equilibrium voltage $V$ vs $I$')
plt.ylim([-60, 40])
plt.show()