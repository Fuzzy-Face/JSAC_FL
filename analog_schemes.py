import numpy as np
import tensorflow as tf
from utils import MyNeighbour
#from sklearn.linear_model import Lasso, MultiTaskLasso, OrthogonalMatchingPursuit
import gc

def proposed_DSGD(G, flattened_theta_by_devices, flattened_theta_half_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices, 
                    W, zeta, CH, N, schedule_list, Tx_times, H_par, P, flag = False, N0 = 10 ** (-169/10) * 1e-3, d = 7850, tilde_d = 2 ** 13):
    K = len(G.nodes())
    CG = np.abs(CH) ** 2

    m = H_par.shape[0]
    r = np.random.binomial(1, .5, (tilde_d,))
    r[r == 0] = -1
    temp = [H_par[i,:] * r for i in range(m)]
    A = (1 / np.sqrt(m)) * np.array(temp) # A^{(t)} is of shape (m, d)

    ########## for flat matrix U (RLC solution) ############
    # A list (device_i's) of model differences that device i prepares to send to its neighbours
    model_diff = [flattened_theta_half_by_devices[i] - flattened_hat_theta_by_devices[i]
                            for i in range(K)] 
    # A list (device_i's) of compressed signal that device i prepares to send to its neighbours before (transmitting) power scaling
    phi = [ A @  np.concatenate((model_diff[i], np.zeros( (tilde_d- d,) )), axis = 0) for i in range(K)] 
    barP = P / N
    
    # Calculate the transmit power sacling factor gamma_i for device i's scheduled neighbors and the transmit power scaling factor alpha_i for device i itself (as the BC node) when device i is scheduled as the star center 
    gamma = {}
    alpha = {}
    for i in range(len(schedule_list)):
        for star in schedule_list[i]:
            gamma[star] = min( barP * N /sum(Tx_times[j]) * CG[j,star] / ( (np.abs(W[star,j])**2) * (np.linalg.norm(phi[j],2)**2) ) for j in schedule_list[i][star] )
            alpha[star] = barP * N /sum(Tx_times[star]) / np.linalg.norm(phi[star],2)**2

    # Calculate the #times for which device i receives, i.e., # of added AWGN
    Rx_times ={node:sum(Tx_times[node]) for node in G.nodes()}
    # Generate random noise a number of receiving times for device i
    # A list (device_i's) of iterators that include a number of receiving times of received AWGN with shape (m,)
    noise =  {node: iter(np.sqrt(N0 / 2) * np.random.randn(Rx_times[node],m) + 1j * np.sqrt(N0 / 2) * np.random.randn(Rx_times[node],m))
                    for node in G.nodes()}

    # (1st step) Process the received signal, by, e.g., re-scaling for decoding
    post_y_by_devices = []
    for node in range(K):
        post_y = sum( W[node,j] * phi[j] for j in G[node] ).astype(complex)
        if node in gamma:
            post_y += next(noise[node]) / np.sqrt(gamma[node])
            for schedule in schedule_list:
                AirCompSetof_node = schedule.get(node)
                if AirCompSetof_node:
                    break
        else:
            AirCompSetof_node = {}
        for i in G[node]:
            if i not in AirCompSetof_node:
                post_y += W[node,i] * next(noise[node]) / ( CH[i,node] * np.sqrt(alpha[i]) )
        post_y_by_devices.append(post_y)

    # (2nd step) Process the rescaled signal by m/d * A.T @ real(post_y)            
    for i in range(K):
        hat_y_by_devices[i] += ( m/d * A.T @ np.real(post_y_by_devices[i]) )[:d]
        flattened_hat_theta_by_devices[i] += ( m/d * A.T @ phi[i] )[:d]

    # RLC_error = [ np.linalg.norm((m/d * A.T @ np.real(post_y_by_devices[i]))[:d] - sum( W[i,j] * model_diff[j] for j in G[i] ), 2)**2 / np. linalg.norm(sum( W[i,j] * model_diff[j] for j in G[i] ), 2)**2 
    #                 for i in range(K) ]
    # print("Normalized MSE(dB):", "|".join("{:.2f}".format(10 * np.log10(RLC_error[i])) for i in range(K)))

    if flag:
        flattened_avg_theta = sum(flattened_theta_by_devices) / K
        cons_error = sum([ np.linalg.norm(flattened_avg_theta - flattened_theta_by_devices[i], 2)**2 for i in range(K) ]) / K
        comp_error = sum([ np.linalg.norm(flattened_hat_theta_by_devices[i]- flattened_theta_by_devices[i], 2)**2 for i in range(K) ]) / K
    
    # flattened_theta_next_by_devices turns out to be a list (device_i's) of aggregate theta_i's after the consensus update)
    flattened_theta_next_by_devices = [ flattened_theta_half_by_devices[i] + 
                                                zeta * ( W[i,i]*flattened_hat_theta_by_devices[i] + hat_y_by_devices[i] -
                                                        flattened_hat_theta_by_devices[i] ) 
                                                    for i in range(K) ]
    # Unflatten the theta_next's in a list (device_i's) of [theta_{i}^{W}, theta_{i}^{b}]
    theta_next_by_devices = [ [ flattened_theta_next_by_devices[i][:7840].reshape((784,10)),  
                                flattened_theta_next_by_devices[i][7840:] ] for i in range(K) ] 
    if flag:
        return theta_next_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices, cons_error, comp_error
    else:
        return theta_next_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices



def TDMA_DSGD(G, flattened_theta_half_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices, W, zeta, CH, N, H_par, P, N0 = 10 ** (-169/10) * 1e-3, d = 7850, tilde_d = 2 ** 13):
    K = len(G.nodes())
    CG = np.abs(CH) ** 2

    m = H_par.shape[0]
    r = np.random.binomial(1, .5, (tilde_d,))
    r[r == 0] = -1
    temp = [H_par[i,:] * r for i in range(m)]
    A = (1 / np.sqrt(m)) * np.array(temp) # A^{(t)} is of shape (m, d)

    ########## for flat matrix U (RLC solution) ############
    # A list (device_i's) of model differences that device i prepares to send to its neighbours
    model_diff = [flattened_theta_half_by_devices[i] - flattened_hat_theta_by_devices[i]
                            for i in range(K)] 
    # A list (device_i's) of compressed signal that device i prepares to send to its neighbours before (transmitting) power scaling
    phi = [ A @  np.concatenate((model_diff[i], np.zeros( (tilde_d- d,) )), axis = 0) for i in range(K)] 
    barP = P / N
    
    # Calculate the transmit power sacling factor gamma_i for device i's scheduled neighbors and the transmit power scaling factor alpha_i for device i itself (as the BC node) when device i is scheduled as the star center 
    gamma = [ min( barP * N /len(G[j]) * CG[j,i] / ( (np.abs(W[i,j])**2) * (np.linalg.norm(phi[j],2)**2) ) for j in G[i] )
                    for i in range(K)]

    # Generate random noise for device i
    noise =  [np.sqrt(N0 / 2) * np.random.randn(m,) + 1j * np.sqrt(N0 / 2) * np.random.randn(m,)
                    for i in range(K)]

    # (1st step) Process the received signal, by, e.g., re-scaling for decoding
    post_y_by_devices = [sum( W[i,j] * phi[j] for j in G[i] ) + noise[i] / np.sqrt(gamma[i]) 
                    for i in range(K)]
    # (2nd step) Process the rescaled signal by m/d * A.T @ real(post_y)            
    for i in range(K):
        hat_y_by_devices[i] += ( m/d * A.T @ np.real(post_y_by_devices[i]) )[:d]
        flattened_hat_theta_by_devices[i] += ( m/d * A.T @ phi[i] )[:d]

    # RLC_error = [ np.linalg.norm((m/d * A.T @ np.real(post_y_by_devices[i]))[:d] - sum( W[i,j] * model_diff[j] for j in G[i] ), 2)**2 / np. linalg.norm(sum( W[i,j] * model_diff[j] for j in G[i] ), 2)**2 
    #                 for i in range(K) ]
    
    # print("Normalized MSE(dB):", "|".join("{:.2f}".format(10 * np.log10(RLC_error[i])) for i in range(K)))

    # flattened_theta_next_by_devices turns out to be a list (device_i's) of aggregate theta_i's after the consensus update)
    flattened_theta_next_by_devices = [ flattened_theta_half_by_devices[i] + 
                                                zeta * ( W[i,i]*flattened_hat_theta_by_devices[i] + hat_y_by_devices[i] -
                                                        flattened_hat_theta_by_devices[i] ) 
                                                    for i in range(K) ]
    # Unflatten the theta_next's in a list (device_i's) of [theta_{i}^{W}, theta_{i}^{b}]
    theta_next_by_devices = [[flattened_theta_next_by_devices[i][:7840].reshape((784,10)),  
                                flattened_theta_next_by_devices[i][7840:]] for i in range(K)] 

    return theta_next_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices 
