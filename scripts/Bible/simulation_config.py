
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class SimulationConfig:
    # physical properties
    L: float = 1.0
    E: float = 0.1
    I_c: float = 2.0e-7
    r_c: float = 0.5e-3
    eta: float = 0.05
    t_c: float = 1.0

    # neural and muscular timescales
    tau_m: float = 100.0e-3
    tau_n: float = 10.0e-3

    # drag coefficients
    C_N: float = 5.2e-9
    C_T: float = 3.3e-9
    C_N_agar: float = 128e-6
    C_T_agar: float = 3.2e-6

    # discretisation
    N: int = 96
    N_controls: int = 6
    N_muscular: int = 48

    # neural coupling
    epsilon_p: float = 1.0
    epsilon_g: float = 0.0
    range_percentage: float = 0.5

    # sigma(A) and F(V)
    c_m: float = 10.0
    c_s: float = 1.0
    a_0: float = 0.0

    I: float = 0.0


    # numerical simulation
    T: float = 10.0
    dt: float = 0.01
    max_time: float = 100.0

    # convergence
    convergence_tol: float = 0.01
    consecutive_required: int = 3
    extra_cycles: int = 3

    K_override: float | None = None

    @property
    def K_water(self):
        return self.C_N / self.C_T

    @property
    def K_agar(self):
        return self.C_N_agar / self.C_T_agar

    @property
    def K(self):
        if self.K_override is not None:
            return self.K_override
        return self.K_agar

    @property
    def e(self):
        return (
            self.E * self.I_c * self.t_c
            / (2 * self.L**4 * self.C_T_agar)
        )

    @property
    def eta_tilde(self):
        return (
            self.eta * self.I_c
            / (2 * self.L**4 * self.C_T_agar)
        )

    @property
    def l(self):
        return self.L / self.N

