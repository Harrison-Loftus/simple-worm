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
V_3  = 2.0
V_4  = 30.0

g_Ca = 4.4
g_K  = 8.0
g_L  = 2.0

V_Ca = 120.0
V_K  = -84.0
V_L  = -60.0

phi  = 0.04


#------Free Parameter------
I = 225

#------Functions------
def m_inf(V):
    return 0.5 * (1 + np.tanh((V - V_1)/(V_2)))

def w_inf(V):
    return 0.5 * (1 + np.tanh((V - V_3)/(V_4)))

def tau_w(V):
    return 1/(np.cosh((V - V_3) / (2 * V_4)))

#------ODE------
def morris_lecar(t, state):
    V = state[0]
    w = state[1]

    dV_dt = (- g_Ca * m_inf(V) * (V - V_Ca) - g_K * w * (V - V_K) - g_L * (V - V_L) + I) / C
    dw_dt = phi * (w_inf(V) - w) / (tau_w(V))

    return np.array([dV_dt, dw_dt])

dt = 0.01
t_eval = np.arange(0, 400 + dt, dt)

initial_state = [-20.0, 0.0]

sols = solve_ivp(morris_lecar, (t_eval[0], t_eval[-1]),  initial_state, method='RK23', t_eval=t_eval, max_step=dt)

V = sols.y[0, :]
w = sols.y[1, :]


plt.figure()
plt.plot(t_eval, V)
plt.xlabel(r'Time (ms)', size=12)
plt.ylabel("V", size=12)
plt.title(r'Time series plot of V for Morris-Lecar', size=16)
plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - V time series.png",
        dpi=300,
        bbox_inches="tight"
)
plt.close()


plt.figure()
plt.plot(t_eval, w)
plt.xlabel(r'Time (ms)', size=12)
plt.ylabel("w", size=12)
plt.title(r'Time series plot of w for Morris-Lecar', size=16)
plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - w time series.png",
        dpi=300,
        bbox_inches="tight"
)
plt.close()


plt.figure()
plt.plot(V, w)
plt.xlabel(r'V', size=12)
plt.ylabel("w", size=12)
plt.title(r'Phase portrait for Morris-Lecar', size=16)
plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - phase portrait.png",
        dpi=300,
        bbox_inches="tight"
)
plt.close()