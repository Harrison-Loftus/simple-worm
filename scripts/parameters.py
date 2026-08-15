L = 1.0
E = 0.1 # Youngs modulus N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
r_c = 0.5e-3  # cuticle width mm
eta = 0.05 # viscosity of the cuticle N·s/mm^2

tau_m = 100.0e-3 # muscle activation timescale seconds
tau_n = 10.0e-3 # neural activity timescale seconds


C_N = 5.2e-9 # Normal drag coefficient in water N·s/mm²
C_T = 3.3e-9 # Tangential drag coefficient in water N·s/mm²

C_N_agar = 128e-6 # Normal drag coefficient in agar N·s/mm²
C_T_agar = 3.2e-6 # Tangential drag coefficient in agar N·s/mm²

K_water = C_N / C_T
K_agar = C_N_agar / C_T_agar

N = 96 # points of the worms body dicretised
N_controls = 6 # number of neurons approximated on each side
N_muscular = 48 # number of body wall muscles on each side approx


t_c = 1.0
e = (E * I_c * t_c)/(L**4 * C_T_agar) / 2
eta_tilde = (eta * I_c)/(L**4 * C_T_agar) / 2


l = L / N # segment length