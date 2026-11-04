import numpy as np



def ODEs(t, state, kappa, config):

    def proprioception_matrix(config):
        n = config.N
        W = np.zeros((config.N, config.N))
        range_percentage = config.range_percentage
        range_val = int(n * range_percentage)
        m = abs(range_val)

        for i in range(n):
            j_min = max(0, i - m)
            for j in range(j_min, i):
                W[j, i] = 1

        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)

        return W


    def gap_junction_matrix(config):
        W_g = np.zeros((config.N, config.N), float)
        for i in range(config.N):
            for j in range(config.N):
                if i == j:
                    if i == 0 or i == config.N-1:
                        W_g[i, j] = -1
                    else:
                        W_g[i, j] = -2
                elif abs(i - j) == 1:
                    W_g[i, j] = 1
        return W_g


    def sigma(A, config):
        c_m = config.c_m
        c_s = config.c_s
        a_0 = config.a_0
        return 0.5 * c_m * (np.tanh((A - a_0)*c_s) + 1)


    def F(V, config):
        I = config.I
        return V - V**3 + I



    N = config.N
    N_controls = config.N_controls
    N_muscular = config.N_muscular

    tau_m = config.tau_m
    tau_n = config.tau_n

    epsilon_p = config.epsilon_p
    epsilon_g = config.epsilon_g
    
    A_V = state[0: N_muscular]
    A_D = state[N_muscular: 2*N_muscular]
    V_V = state[2*N_muscular: 2*N_muscular + N_controls]
    V_D = state[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls]
    
    P = W_p @ kappa

    
    P_regions = np.array_split(P, N_controls)
    P_ctrl = np.array([np.mean(region) for region in P_regions])
    
    
    repeat_fact = N_muscular // N_controls
    
    V_V_ctrl = np.repeat(V_V, repeat_fact)
    V_D_ctrl = np.repeat(V_D, repeat_fact)
    

    dA_Vdt = (1/tau_m) * (-A_V + sigma(V_V_ctrl - V_D_ctrl, config))
    dA_Ddt = (1/tau_m) * (-A_D + sigma(V_D_ctrl - V_V_ctrl, config))

    dV_Vdt = (1/tau_n)*(F(V_V, config) - epsilon_p * P_ctrl) #+ epsilon_g * W_g @ V_V)
    dV_Ddt = (1/tau_n)*(F(V_D, config) + epsilon_p * P_ctrl)#+ epsilon_g * W_g @ V_D) 
    results = np.concatenate([dA_Vdt, dA_Ddt, dV_Vdt, dV_Ddt])

    return results    

 