from forward_model import *

if __name__ == "__main__":

    worm_positions, curvatures = example2()
    t_eval = np.linspace(0, T, n_timesteps + 1) 

    print(curvatures.shape)

    state = np.zeros(5*N)
    state[3*N] = 1.0
    state[4*N] = -1.0
    sols = []

    sol = solve_ivp(Example.ODEs, [0, T], state, t_eval=t_eval, method='RK45', atol=1e-6, rtol=1e-9, max_step=dt)
    sols.append(sol)

    kappa = sol.y[0:N, :]
    print(kappa.shape)

    #-------------wavelength comparison---------------------
    wavelength_simple_worm, frequency_simple_worm = module.Hilbert_Transform(curvatures, N, N_controls,t_eval)
    print("Wavelength from simple worm: ", np.round(wavelength_simple_worm, 2))
    print("Frequency from simple worm: ", np.round(frequency_simple_worm, 2))

    wavelength_preliminary, frequency_preliminary = module.Hilbert_Transform(kappa, N, N_controls, t_eval)
    print("Wavelength from preliminary work: ", np.round(wavelength_preliminary, 2))
    print("Frequency from preliminary work: ", np.round(frequency_preliminary, 2))
    
    
    #-------------curvature comparison plots----------------
    plt.figure()
    plt.plot(t_eval, kappa[N//2,:], label='preliminary values')
    plt.plot(t_eval, curvatures[N//2,:], label='simple worm values')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Curvature')
    plt.show()

    #-------------curvature difference plot----------------
    plt.figure()
    plt.plot(t_eval, np.abs(kappa[N//2,:] - curvatures[N//2,:]), label='difference in vals')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Curvature difference')
    plt.show()


    print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
