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

from scipy.optimize import KrylovJacobian

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.Archive.kymograph import *
from scripts.Archive.hilbert_calc import *
from scripts.Archive.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.Archive.curvature_viewer import view_curvature_pyqtgraph

from scipy.sparse.linalg import eigs

from pathlib import Path
import time
tick = time.time()
            
# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

BASE_DIR = OUTPUT_DIR / SCRIPT_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)

from scripts.Archive.Tonic_Bifurcation.numerical_continuation.test import *


from scipy.optimize import root 
from scipy.optimize._numdiff import approx_derivative


def residual(state, I, epsilon_p):
    return ODEs(0.0, state, I, epsilon_p)

n_states = 108
n_guesses = 10

a = -5
b = 5

x_guesses = a + (b - a) * np.random.rand(n_guesses, n_states)

equilibrias = np.full(([len(x_guesses), len(I_vals), len(epsilon_p_vals), n_states]), np.nan, dtype=float)
amp_vals = np.full(([len(x_guesses), len(I_vals), len(epsilon_p_vals)]), np.nan, dtype=float)
max_eig_vals = np.full(([len(x_guesses), len(I_vals), len(epsilon_p_vals)]), np.nan, dtype=float)
max_imag_vals = np.full(([len(x_guesses), len(I_vals), len(epsilon_p_vals)]), np.nan, dtype=float)

for i, x_guess in enumerate(x_guesses):
    print(i)

    for k, eps in enumerate(epsilon_p_vals):
            
        x = x_guess

        for j, I in enumerate(I_vals):
            sol = root(residual, x, args=(I,eps,), method='hybr')
            if not sol.success:
                continue

            res = residual(sol.x,I,eps)

            if np.linalg.norm(res) > 1e-8:
                continue

            x = sol.x
            
            state_eq = sol.x
                        
            J = approx_derivative(lambda x: residual(x, I, eps), state_eq)
            
                                    
            eigvals = np.linalg.eigvals(J)
            
            max_eig_vals[i,j,k] = np.max(np.real(eigvals))
            max_imag_vals[i,j,k] = np.max(np.imag(eigvals))
    
            equilibrias[i, j, k, :] = state_eq

print(time.time() - tick)


plt.figure(figsize=(10, 6))
for i, x_guess in enumerate(x_guesses):
    plt.scatter(I_vals, max_eig_vals[i, :], marker='o', color='black', alpha=0.1)
plt.xlabel(r'$I$')
plt.ylabel(r'Max real part of eigenvalues')
plt.show()

plt.figure(figsize=(10, 6))
for i, x_guess in enumerate(x_guesses):
    plt.scatter(I_vals, max_imag_vals[i, :], marker='o', color='black', alpha=0.1)
plt.xlabel(r'$I$')
plt.ylabel(r'Max imaginary part of eigenvalues')
plt.show()

plt.figure(figsize=(10, 6))

for i, x_guess in enumerate(x_guesses):

    eps_idx = 0

    stable = max_eig_vals[i, :, eps_idx] < 0
    unstable = max_eig_vals[i, :, eps_idx] > 0

    plt.scatter(
        I_vals[stable],
        equilibrias[i, :, eps_idx, 0][stable],
        color="blue",
        alpha=0.1,
        s=10,
)

    plt.scatter(
        I_vals[unstable],
        equilibrias[i, :, eps_idx, 0][unstable],
        color='red',
        alpha=0.1,
        s=10
    )

plt.xlabel(r'$I$')
plt.ylabel(r'Equilibrium voltage $V$')
plt.title(r'Equilibrium voltage $V$ vs $I$')
plt.show()