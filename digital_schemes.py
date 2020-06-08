import numpy as np
import tensorflow as tf
from utils import comp_quant_encoding, dig_sparse_level, MyNeighbour

def vanila_DSGD(var_lists_by_Devices, W): # vanila DSGD 
    tilde_theta_by_devices = [[tf.add_n([theta * w_i_j for theta, w_i_j in zip(weights, w_i)]) 
                                                        for w_i in W] for weights in zip(*var_lists_by_Devices)]
    tilde_theta_by_devices = list(zip(*tilde_theta_by_devices))
    return tilde_theta_by_devices



def ub_DSGD(G, W, flattened_theta_by_devices, acc_error, K = 8, d = 7850, b = 32):
    E = [(i+1, j+1) for i, j in G.edges()]
    q_array = [ d ] * len(G.nodes()) 
    digits_array = {d : b}
    # Quantize each entry of theta_i's in float-point precision of float32 
    flattened_quantized_theta_by_devices, acc_error = comp_quant_encoding(flattened_theta_by_devices, acc_error, q_array, digits_array)
    # flattened_tilde_theta_by_devices turns out to be a list (device_i's) of weighted theta_i's after the consensus step)
    flattened_tilde_theta_by_devices = np.array([ W[i,i] * flattened_theta_by_devices[i] + sum([ W[i,j-1] * flattened_quantized_theta_by_devices[j-1] 
                                            for j in MyNeighbour(E,i+1) ]) 
                                                    for i in range(K) ])
    # Unflatten the weighted theta_i's in a list (device_i's) of [theta_{i}^W, theta_{i}^b]
    tilde_theta_by_devices = [[flattened_tilde_theta_by_devices[i][:7840].reshape((784,10)),  
                                flattened_tilde_theta_by_devices[i][7840:]] for i in range(K)]

    return tilde_theta_by_devices, acc_error



def proposed_DSGD(G, CG,  W, flattened_theta_by_devices, N, Chi, N0, log2_comb_list, acc_error, K = 8):
    acc_error_pre = acc_error
    E = [(i+1, j+1) for i, j in G.edges()]
    q_array, digits_array = dig_sparse_level(E, CG, N, Chi, N0, log2_comb_list)
    flattened_quantized_theta_by_devices, acc_error = comp_quant_encoding(flattened_theta_by_devices, acc_error_pre, q_array, digits_array)

    # flattened_tilde_theta_by_devices turns out to be a list (device_i's) of weighted theta_i's after the consensus step)
    flattened_tilde_theta_by_devices = np.array([ W[i,i] * flattened_theta_by_devices[i] + sum([ W[i,j-1] * flattened_quantized_theta_by_devices[j-1] 
                                            for j in MyNeighbour(E,i+1) ]) 
                                                    for i in range(K) ])
    # Unflatten the weighted theta_i's in a list (device_i's) of [theta_{i}^W, theta_{i}^b]
    tilde_theta_by_devices = [[flattened_tilde_theta_by_devices[i][:7840].reshape((784,10)),  
                                flattened_tilde_theta_by_devices[i][7840:]] for i in range(K)]

    return tilde_theta_by_devices, acc_error



def TDMA_DSGD(G, CG, W, flattened_theta_by_devices, N, N0, log2_comb_list, acc_error, K = 8):
    acc_error_pre = acc_error
    E = [(i+1, j+1) for i, j in G.edges()]
    Chi = K
    q_array, digits_array = dig_sparse_level(E, CG, N, Chi, N0, log2_comb_list)
    flattened_quantized_theta_by_devices, acc_error = comp_quant_encoding(flattened_theta_by_devices, acc_error_pre, q_array, digits_array)

    # flattened_tilde_theta_by_devices turns out to be a list (device_i's) of weighted theta_i's after the consensus step)
    flattened_tilde_theta_by_devices = np.array([ W[i,i] * flattened_theta_by_devices[i] + sum([ W[i,j-1] * flattened_quantized_theta_by_devices[j-1] 
                                            for j in MyNeighbour(E,i+1) ]) 
                                                    for i in range(K) ])
    # Unflatten the weighted theta_i's in a list (device_i's) of [theta_{i}^W, theta_{i}^b]
    tilde_theta_by_devices = [[flattened_tilde_theta_by_devices[i][:7840].reshape((784,10)),  
                                flattened_tilde_theta_by_devices[i][7840:]] for i in range(K)]

    return tilde_theta_by_devices, acc_error
