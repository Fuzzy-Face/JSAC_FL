import numpy as np
import tensorflow as tf
from utils import comp_quant_encoding, dig_sparse_level, MyNeighbour

def vanila_DSGD(var_lists_by_Devices, W, zeta): # vanila DSGD 
    theta_next_by_devices = [[(1-zeta) * weights[i] + zeta * tf.add_n([theta * w_i_j for theta, w_i_j in zip(weights, W[i])]) 
                                                        for i in range(len(var_lists_by_Devices))] for weights in zip(*var_lists_by_Devices)]
    theta_next_by_devices = list(zip(*theta_next_by_devices))
    return theta_next_by_devices



def ub_DSGD(G, flattened_theta_by_devices, flattened_hat_theta_by_devices, W, zeta, d = 7850, b = 64):
    K = len(G.nodes())
    q_array = [ d ] * K
    digits_array = [b] * K
    # Quantize each entry of theta_i's in float-point precision of float64 
    flattened_hat_theta_by_devices = comp_quant_encoding(flattened_theta_by_devices, flattened_hat_theta_by_devices, q_array, digits_array)
    # flattened_theta_next_by_devices turns out to be a list (device_i's) of aggregate theta_i's after the consensus update)
    flattened_theta_next_by_devices = [ flattened_theta_by_devices[i] + 
                                                zeta * (sum([ W[i,j] * flattened_hat_theta_by_devices[j] for j in G.nodes() ]) -
                                                        flattened_hat_theta_by_devices[i]) 
                                                    for i in range(K) ]
    # Unflatten the theta_next's in a list (device_i's) of [theta_{i}^{W}, theta_{i}^{b}]
    theta_next_by_devices = [[flattened_theta_next_by_devices[i][:7840].reshape((784,10)),  
                                flattened_theta_next_by_devices[i][7840:]] for i in range(K)]

    return theta_next_by_devices, flattened_hat_theta_by_devices



def proposed_DSGD(G, flattened_theta_by_devices, flattened_hat_theta_by_devices, W, zeta, CH, N, Chi, N0, log2_comb_list):
    K = len(G.nodes())
    CG = np.abs(CH) ** 2
    q_array, digits_array = dig_sparse_level(G, CG, N, Chi, N0, log2_comb_list)
    flattened_hat_theta_by_devices = comp_quant_encoding(flattened_theta_by_devices, flattened_hat_theta_by_devices, q_array, digits_array)

    # flattened_theta_next_by_devices turns out to be a list (device_i's) of aggregate theta_i's after the consensus update)
    flattened_theta_next_by_devices = [ flattened_theta_by_devices[i] + 
                                                zeta * (sum([ W[i,j] * flattened_hat_theta_by_devices[j] for j in G.nodes() ]) -
                                                        flattened_hat_theta_by_devices[i]) 
                                                    for i in range(K) ]
    # Unflatten the theta_next's in a list (device_i's) of [theta_{i}^{W}, theta_{i}^{b}]
    theta_next_by_devices = [[flattened_theta_next_by_devices[i][:7840].reshape((784,10)),  
                                flattened_theta_next_by_devices[i][7840:]] for i in range(K)]

    return theta_next_by_devices, flattened_hat_theta_by_devices



def TDMA_DSGD(G, flattened_theta_by_devices, flattened_hat_theta_by_devices, W, zeta, CG, N, N0, log2_comb_list):
    K = len(G.nodes())
    q_array, digits_array = dig_sparse_level(G, CG, N, K, N0, log2_comb_list)
    flattened_hat_theta_by_devices = comp_quant_encoding(flattened_theta_by_devices, flattened_hat_theta_by_devices, q_array, digits_array)

    # flattened_theta_next_by_devices turns out to be a list (device_i's) of aggregate theta_i's after the consensus update)
    flattened_theta_next_by_devices = [ flattened_theta_by_devices[i] + 
                                                zeta * (sum([ W[i,j] * flattened_hat_theta_by_devices[j] for j in G.nodes() ]) -
                                                        flattened_hat_theta_by_devices[i]) 
                                                    for i in range(K) ]
    # Unflatten the theta_next's in a list (device_i's) of [theta_{i}^{W}, theta_{i}^{b}]
    theta_next_by_devices = [[flattened_theta_next_by_devices[i][:7840].reshape((784,10)),  
                                flattened_theta_next_by_devices[i][7840:]] for i in range(K)]

    return theta_next_by_devices, flattened_hat_theta_by_devices
    