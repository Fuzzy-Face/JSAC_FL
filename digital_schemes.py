import numpy as np
import tensorflow as tf
from utils import dig_comp_level

def vanila_DSGD(var_lists_by_Devices, W, zeta): # vanila DSGD 
    theta_next_by_devices = [[(1-zeta) * weights[i] + zeta * tf.add_n([theta * w_i_j for theta, w_i_j in zip(weights, W[i])]) 
                                                        for i in range(len(var_lists_by_Devices))] for weights in zip(*var_lists_by_Devices)]
    theta_next_by_devices = list(zip(*theta_next_by_devices))
    return theta_next_by_devices



def proposed_DSGD(G, flattened_theta_by_devices, flattened_hat_theta_by_devices, W, zeta, CH, N, Chi, H, barP, d = 7850, tilde_d = 2 ** 13):
    K = len(G.nodes())
    CG = np.abs(CH) ** 2
    # A list (device_i's) of #rows for the RLC matrix supported by the channels of each device to its neighbors
    m_array= dig_comp_level(G, CG, N, Chi, barP)

    ########## Random Linear Coding (RLC) ############
    r = np.random.binomial(1, .5, (tilde_d,))
    r[r == 0] = -1
    for i in range(K):
        temp = [H[i,:] * r for i in range(m_array[i])]
        A = (1 / np.sqrt(m_array[i])) * np.array(temp) # A^{(t)} is of shape (m, d)
        # A list (device_i's) of model differences that device i prepares to broadcast to its neighbours
        model_diff = flattened_theta_by_devices[i] - flattened_hat_theta_by_devices[i]
        # A list (device_i's) of compressed signal that device i prepares to broadcast to its neighbours 
        u = np.concatenate((model_diff, np.zeros( (tilde_d - d,) )), axis = 0) 
        flattened_hat_theta_by_devices[i] += ( m_array[i]/d * A.T @ (A @ u) )[:d]

    # flattened_theta_next_by_devices turns out to be a list (device_i's) of aggregate theta_i's after the consensus update
    flattened_theta_next_by_devices = [ flattened_theta_by_devices[i] + 
                                                zeta * (sum([ W[i,j] * flattened_hat_theta_by_devices[j] for j in G.nodes() ]) -
                                                        flattened_hat_theta_by_devices[i]) 
                                                    for i in range(K) ]

    # Unflatten the theta_next's in a list (device_i's) of [theta_{i}^{W}, theta_{i}^{b}]
    theta_next_by_devices = [[flattened_theta_next_by_devices[i][:7840].reshape((784,10)),  
                                flattened_theta_next_by_devices[i][7840:]] for i in range(K)]

    return theta_next_by_devices, flattened_hat_theta_by_devices
