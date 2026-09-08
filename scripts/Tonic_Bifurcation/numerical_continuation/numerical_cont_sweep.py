from numpy.linalg import norm
from dolfinx import fem
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import ufl

from simple_worm.controls import (
    ControlsFenics,
    ControlsNumpy,
    ControlSequenceFenics,
)
from simple_worm.material_parameters import MaterialParameters, MaterialParametersFenics
from simple_worm.worm import Worm
from simple_worm.util import f2n, v2f



import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.kymograph import *
from scripts.hilbert_calc import *
from scripts.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.curvature_viewer import view_curvature_pyqtgraph


from pathlib import Path
import time

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

BASE_DIR = OUTPUT_DIR / SCRIPT_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)

from scripts.Tonic_Bifurcation.numerical_continuation.test import *


from scipy.optimize import root 
from scipy.optimize._numdiff import approx_derivative


def residual(state, I, epsilon_p):
    return ODEs(0.0, state, I, epsilon_p)


amp_vals = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)
max_eig_vals = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)
max_imag_vals = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)

for i, eps in enumerate(epsilon_p_vals):
    print("epsilon_p: ", np.round(eps, 2))
    for j, I in enumerate(I_vals):
        print("I: ", np.round(I, 2))
        n_state = 108
        x0 = np.zeros(n_state)

        #x0[2*N_muscular: 2*N_muscular + N_controls] = 0.5
        #x0[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls] = -0.5
        

        sol = root(residual, x0, args=(I, eps),method="hybr")

        x_eq = sol.x

        J = approx_derivative(lambda x: residual(x, I, eps), x_eq)
        eigs = np.linalg.eigvals(J)
        max_real_part = np.max(np.real(eigs))
        print("Max real part of eigenvalues: ", max_real_part)

        max_eig_vals[i, j] = max_real_part

        max_imag_part = np.max(np.imag(eigs))
        print("Max imaginary part of eigenvalues: ", max_imag_part)

        max_imag_vals[i, j] = max_imag_part

        idx = np.argsort(np.real(eigs))[::-1]
        print(x_eq[96:102])   # ventral neurons
        print(x_eq[102:108])  # dorsal neurons

        kappa = x_eq[0:48] - x_eq[48:96]

        amp = np.max(kappa) - np.min(kappa)
        amp_vals[i, j] = amp

        print("amp: ", amp)

plt.figure(figsize=(10, 6))
plt.imshow(amp_vals, extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0], epsilon_p_vals[-1]], origin='lower', aspect='auto', cmap='bwr')
cbar = plt.colorbar(label=r'Amplitude of Kappa')
cbar.ax.yaxis.label.set_size(12)
plt.xlabel("I")
plt.ylabel("Amplitude of Kappa")
plt.title("Amplitude of Kappa vs I for epsilon_p = {}".format(eps))
plt.show()


plt.figure(figsize=(10, 6))
plt.imshow(max_eig_vals, extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0], epsilon_p_vals[-1]], origin='lower', aspect='auto', cmap='bwr')
cbar = plt.colorbar(label=r'Max Real Part of Eigenvalues')
cbar.ax.yaxis.label.set_size(12)
plt.xlabel("I")
plt.ylabel("Max Real Part of Eigenvalues")
plt.title("Max Real Part of Eigenvalues vs I for epsilon_p = {}".format(eps))
plt.axhline(0, color='red', linestyle='--', label='Re(eigenvalue) = 0')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(max_imag_vals, extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0], epsilon_p_vals[-1]], origin='lower', aspect='auto', cmap='bwr')
cbar = plt.colorbar(label=r'Max Imaginary Part of Eigenvalues')
cbar.ax.yaxis.label.set_size(12)
plt.xlabel("I")
plt.ylabel("Max Imaginary Part of Eigenvalues")
plt.title("Max Imaginary Part of Eigenvalues vs I for epsilon_p = {}".format(eps))
plt.show()