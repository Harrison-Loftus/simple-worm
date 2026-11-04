import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.colors import TwoSlopeNorm


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.optimize import root

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

amp_dir = OUTPUT_DIR / "amplitudes"
amp_dir.mkdir(exist_ok=True)


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

phi_vals  = np.arange(0.0, 0.3, 0.005)
# phi_vals = np.array([1/15])

#------Free Parameter------
I_vals = np.arange(0,200, 1.0)

#------Functions------
def m_inf(V):
    return 0.5 * (1 + np.tanh((V - V_1)/(V_2)))

def w_inf(V):
    return 0.5 * (1 + np.tanh((V - V_3)/(V_4)))

def tau_w(V):
    return 1/(np.cosh((V - V_3) / (2 * V_4)))

#------ODE------
def morris_lecar(t, state, phi, I):
    V = state[0]
    w = state[1]

    dV_dt = (- g_Ca * m_inf(V) * (V - V_Ca) - g_K * w * (V - V_K) - g_L * (V - V_L) + I ) / C
    dw_dt = phi * (w_inf(V) - w) / (tau_w(V))

    return np.array([dV_dt, dw_dt])



def periodic_orbit_finder(phi, I, T_eval, sol):
    
    V = sol.y[0]

    tail = len(V)//4

    V_tail = V[-tail:]
   
    # V_peaks_idx, _ = find_peaks(V, prominence=0.1)
    # V_peaks = V[V_peaks_idx]

    amp = np.max(V_tail) - np.min(V_tail)

    if amp <1.0:
        return np.nan
    else:
        return amp

periods = np.full((len(phi_vals), len(I_vals)), np.nan, dtype=float)
max_Vs = np.full((len(phi_vals), len(I_vals)), np.nan, dtype=float)
amps = np.full((len(phi_vals), len(I_vals)), np.nan, dtype=float)

dt = 0.01
t_eval = np.arange(0, 2000 + dt, dt)

initial_state = [0.0, 0.0]

for i, phi in enumerate(phi_vals):
    for j, I in enumerate(I_vals):
        sol = solve_ivp(morris_lecar, [t_eval[0], t_eval[-1]], initial_state, args=(phi, I), t_eval=t_eval)
        amp = periodic_orbit_finder(phi,I, t_eval, sol)
        print("I: ", I, "amp: ", amp)
        amps[i,j] = amp

with open(OUTPUT_DIR / "amps.csv", "w") as f:
    # header row
    f.write("I," + ",".join(f"{phi:.4f}" for phi in phi_vals) + "\n")

    # data rows
    for i_idx, I in enumerate(I_vals):
        row = [f"{I:.4f}"]

        for phi_idx in range(len(phi_vals)):
            row.append(str(amps[phi_idx, i_idx]))

        f.write(",".join(row) + "\n")



plt.imshow(
    amps,
    aspect='auto',
    extent=[I_vals[0], I_vals[-1], phi_vals[0], phi_vals[-1]],
    origin='lower',
    cmap='bwr'
)

plt.xlabel(r'$I$', size=12)
plt.ylabel(r'$\phi$', size=12)
cbar = plt.colorbar()
cbar.set_label('voltage amplitude', fontsize=12)
plt.show()


