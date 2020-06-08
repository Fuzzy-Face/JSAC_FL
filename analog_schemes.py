import numpy as np
import tensorflow as tf
from utils import MyNeighbour
from sklearn.linear_model import Lasso, MultiTaskLasso, OrthogonalMatchingPursuit
import gc

def Rx_DSGD(G, CH, W, flattened_theta_by_devices, N, A, A_dag, N0, noise, acc_error, sr, estimator = 'Lasso', sigma_square = .01, lamda = 1e-6, K = 8, d = 7850):

    barP = .001
    EC_flattened_theta_by_devices = [flattened_theta_by_devices[i] + acc_error[i] for i in range(K)]
    ######### for skinny matrix U (least-square solution) ############
    if A.shape[0] >= A.shape[1]:
        gamma = min([barP * N / (np.linalg.norm(A @ EC_flattened_theta_by_devices[j],2) ** 2 * sum(1/np.abs(CH[j][i]) ** 2 for i in G[j])) 
                        for j in range(K)])
        y = [ np.sqrt(gamma) * A @ sum (EC_flattened_theta_by_devices[j] for j in G[i]) + noise[i] for i in range(K)]

    # U = [LMMSE_Rx(i, G, CH, A, alpha, Rsc[i], Rcc[i]) for i in range(K)] # a list (device_i's) of detection matrix U_i at the Rx of device i
        hat_y = [np.real(y[i]) / np.sqrt(gamma) for i in range(K)] # a list (device_i's) of estimated received signal at device i, i.e., A\sum_{j\in\mathcal{N}_i}\theta_j
        hat_sum_theta =[A_dag @ hat_y[i] 
                            for i in range(K)] # a list (device_i's) of estimated sum of parameters from the adjacent nodes of i, i.e., \sum_{j\in\mathcal{N}_i}\theta_j
    else:
    ########## for flat matrix U (sparse-recovery solution) ############
        k = int((1- sr ** (1 / K)) * d )
        sparse_EC_flattened_theta_by_devices = np.zeros(flattened_theta_by_devices.shape) # A list (device_i's) of quantized theta_{i}'s that device i prepares to send to its neighbours
        for i in range(K):
            idx = np.argsort(np.abs(EC_flattened_theta_by_devices[i]))[d-k:]  # Index for the Top-q entries
            sparse_EC_flattened_theta_by_devices[i][idx] = EC_flattened_theta_by_devices[i][idx]

        gamma = min([barP * N / (np.linalg.norm(A @ sparse_EC_flattened_theta_by_devices[j],2) ** 2 * sum(1 / np.abs(CH[j][i]) ** 2 for i in G[j])) 
                        for j in range(K)])

        y = [ np.sqrt(gamma) * A @ sum (sparse_EC_flattened_theta_by_devices[j] for j in G[i]) + noise[i] 
                for i in range(K)]       
        # U = [LMMSE_Rx(i, G, CH, A, gamma, Rsc[i], Rcc[i]) for i in range(K)] # a list (device_i's) of detection matrix U_i at the Rx of device i
        hat_y = [np.real(y[i]) / np.sqrt(gamma) for i in range(K)] # a list (device_i's) of estimated received signal at device i, i.e., A\sum_{j\in\mathcal{N}_i}\theta_j

        if estimator =='OMP':
            # n_nonzeros = [len(np.nonzero(sum(sparse_EC_flattened_theta_by_devices[j] for j in G[i]))[0]) for i in range(K)] #a list (device_i's) of number_of_nonzeros in the target signal
            # hat_sum_theta = [ OrthogonalMatchingPursuit(n_nonzeros[i]).fit( A, hat_y[i] ).coef_ for i in range(K) ]
            hat_sum_theta = [ OrthogonalMatchingPursuit(int(min(A.shape[0], 0.1 * d))).fit( A, hat_y[i] ).coef_ for i in range(K) ] # construct a list (device_i's) of estimated sum of parameters from the adjacent nodes of i, i.e., \sum_{j\in\mathcal{N}_i}\theta_j
            OMP_error = [ np.mean(np.abs(hat_sum_theta[i] - sum(sparse_EC_flattened_theta_by_devices[j] for j in G[i])) ** 2) for i in range(K) ]
            OMP_rel_error = [OMP_error[i] / np.mean(np.abs(sum(sparse_EC_flattened_theta_by_devices[j] for j in G[i])) ** 2) for i in range(K) ]
            print("Normalized MSE(dB):", "|".join("{:.2f}".format(10 * np.log10(OMP_rel_error[i])) for i in range(K)))
        else:
            hat_sum_theta = [ Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[i] ).coef_ for i in range(K) ] # construct a list (device_i's) of estimated sum of parameters from the adjacent nodes of i, i.e., \sum_{j\in\mathcal{N}_i}\theta_j
            # hat_sum_theta = MultiTaskLasso(alpha = lamda, fit_intercept = False).fit( A, list(zip(*hat_y)) ).coef_ # construct a list (device_i's) of estimated sum of parameters from the adjacent nodes of i, i.e., \sum_{j\in\mathcal{N}_i}\theta_j
            Lasso_error = [ np.mean(np.abs(hat_sum_theta[i] - sum(sparse_EC_flattened_theta_by_devices[j] for j in G[i])) ** 2) for i in range(K) ]
            print("Normalized MSE(dB):", "|".join("{:.2f}".format(10 * np.log10(Lasso_error[i])) for i in range(K)))
            # hat_sum_theta = [ AMP( A, hat_y[i], N0, mu = 0, sigma_square = sigma_square, shape = (d,) ) for i in range(K) ] # construct a list (device_i's) of estimated sum of parameters from the adjacent nodes of i, i.e., \sum_{j\in\mathcal{N}_i}\theta_j
            # AMP_error = [ np.mean(np.abs(hat_sum_theta[i] - sum(sparse_EC_flattened_theta_by_devices[j] for j in G[i])) ** 2) for i in range(K) ]
            # AMP_rel_error = [AMP_error[i] / np.mean(np.abs(sum(sparse_EC_flattened_theta_by_devices[j] for j in G[i])) ** 2) for i in range(K) ]
            # print("Normalized MSE(dB):", "|".join("{:.2f}".format(10 * np.log10(AMP_rel_error[i])) for i in range(K)))



    # flattened_tilde_theta_by_devices turns out to be a list (device_i's) of combined theta_i's used for the consensus step)
    alpha = W[0][[j for j in G[0]][0]]
    flattened_tilde_theta_by_devices = np.array([(1-G.degree[i] * alpha) * flattened_theta_by_devices[i] + alpha * hat_sum_theta[i]
                                                    for i in range(K)])  

    # Unflatten the combined theta_i's in a list (device_i's) of [theta_{i}^W, theta_{i}^b]
    tilde_theta_by_devices = [[flattened_tilde_theta_by_devices[i][:7840].reshape((784, 10)),
                               flattened_tilde_theta_by_devices[i][7840:]] for i in range(K)]
    # Update the estimated self and cross co-variance matrices using the exponential weighted moving average (EWMA)
    # for i in range(K):
    #     # Rsc[i] = (1 - beta_sc) * (flattened_theta_by_devices[i].reshape(d,1) @ flattened_theta_by_devices[i].reshape(d,1).T) + beta_sc * Rsc[i]
    #     # Rcc[i] = (1 - beta_cc) * ((flattened_theta_by_devices[i].reshape(d,1) @ hat_sum_theta[i].reshape(d,1).T) / G.degree[i]) + beta_cc * Rcc[i]
    #     Rsc[i] = (1 - beta_sc) * flattened_theta_by_devices[i] ** 2 + beta_sc * Rsc[i]
    #     Rcc[i] = (1 - beta_cc) * flattened_theta_by_devices[i] * hat_sum_theta[i] / G.degree[i] + beta_cc * Rcc[i]

    if A.shape[0] >= A.shape[1]:
        for i in range(K):
            acc_error[i] += (flattened_theta_by_devices[i] - EC_flattened_theta_by_devices[i])
    else:
        for i in range(K):
            acc_error[i] += (flattened_theta_by_devices[i] - sparse_EC_flattened_theta_by_devices[i])

    return tilde_theta_by_devices, acc_error



def sp_case1(G, CH, W, flattened_theta_by_devices, N, A, N0, noise, acc_error, sr, estimator = 'Lasso', sigma_square = .01, lamda = 1e-6, K = 8, d = 7850):
    barP = .001
    EC_flattened_theta_by_devices = [flattened_theta_by_devices[i] + acc_error[i] for i in range(K)]
    ########## for flat matrix U (sparse-recovery solution) ############
    k = int((1- sr ** (1 / K)) * d )
    sparse_EC_flattened_theta_by_devices = np.zeros(flattened_theta_by_devices.shape) # A list (device_i's) of quantized theta_{i}'s that device i prepares to send to its neighbours
    for i in range(K):
        idx = np.argsort(np.abs(EC_flattened_theta_by_devices[i]))[d-k:]  # Index for the Top-q entries
        sparse_EC_flattened_theta_by_devices[i][idx] = EC_flattened_theta_by_devices[i][idx]

    ########## Calculate gamma and alpha ############
    num_transmit_ota = [0,1,1,1,1,1,1,1] # the number of times that each device transmits to a star center
    num_transmit_bc = [1,0,0,0,0,0,0,0] # the number of times that each device transmits as a star center
    # Define N_j^o as a subset of each device's neighboring nodes who are scheduled (prior to it) as the star center
    neighbor_o = [(), (0,), (0,), (0,), (0,), (0,), (0,), (0,)]
    # transmit power scaling factor gamma taking into account all the slots in which each device transmits to a "star" center
    gamma = min( 1 / (np.linalg.norm(A @ sparse_EC_flattened_theta_by_devices[j],2) ** 2 * sum( 1 / np.abs(CH[i][j]) ** 2 for i in neighbor_o[j])) 
                    * barP * N * (num_transmit_ota[j] / (num_transmit_ota[j] + num_transmit_bc[j])) for j in range(K) if len(neighbor_o[j]) > 0)
    # transmit power scaling factor alpha taking into account the slot (maximum 1) that each device transmits as a "star" center
    alpha = [ 1 / np.linalg.norm(A @ sparse_EC_flattened_theta_by_devices[j],2) ** 2 
                    * barP * N * (num_transmit_bc[j] / (num_transmit_ota[j] + num_transmit_bc[j])) for j in range(K)]
    
    ########## Combine received signals from multiple slots at each device ############
    # Intialize the received signal y, the noisy observation hat_y (A @ sum() + N_0), and the recovered sum of signals hat_sum_theta (sum())
    y = np.zeros((K, A.shape[0]), dtype = np.complex128)
    hat_y = np.zeros((K, A.shape[0]))
    hat_sum_theta = np.zeros((K, A.shape[1]))
    # Node 0 as Rx: from node 1-7
    y[0] = np.sqrt(gamma) * A @ sum(sparse_EC_flattened_theta_by_devices[j] for j in G[0]) + noise[0] 
    hat_y[0] = np.real(y[0]) / np.sqrt(gamma)
    hat_sum_theta[0] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[0] ).coef_
    # Node 1 as Rx: from node 0
    y[1] = np.sqrt(alpha[0]) * A @ (sparse_EC_flattened_theta_by_devices[0]) * CH[0,1] + noise[1] 
    hat_y[1] = np.real(y[1] / (np.sqrt(alpha[0]) * CH[0,1]))
    hat_sum_theta[1] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[1] ).coef_
    # Node 2 as Rx: from node 0
    y[2] = np.sqrt(alpha[0]) * A @ (sparse_EC_flattened_theta_by_devices[0]) * CH[0,2] + noise[2] 
    hat_y[2] = np.real(y[2] / (np.sqrt(alpha[0]) * CH[0,2]))
    hat_sum_theta[2] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[2] ).coef_
    # Node 3 as Rx: from node 0
    y[3] = np.sqrt(alpha[0]) * A @ (sparse_EC_flattened_theta_by_devices[0]) * CH[0,3] + noise[3] 
    hat_y[3] = np.real(y[3] / (np.sqrt(alpha[0]) * CH[0,3]))
    hat_sum_theta[3] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[3] ).coef_
    # Node 4 as Rx: from node 0
    y[4] = np.sqrt(alpha[0]) * A @ (sparse_EC_flattened_theta_by_devices[0]) * CH[0,4] + noise[4] 
    hat_y[4] = np.real(y[4] / (np.sqrt(alpha[0]) * CH[0,4]))
    hat_sum_theta[4] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[4] ).coef_
    # Node 5 as Rx: from node 0
    y[5] = np.sqrt(alpha[0]) * A @ (sparse_EC_flattened_theta_by_devices[0]) * CH[0,5] + noise[5] 
    hat_y[5] = np.real(y[5] / (np.sqrt(alpha[0]) * CH[0,5]))
    hat_sum_theta[5] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[5] ).coef_
    # Node 6 as Rx: from node 0
    y[6] = np.sqrt(alpha[0]) * A @ (sparse_EC_flattened_theta_by_devices[0]) * CH[0,6] + noise[6] 
    hat_y[6] = np.real(y[6] / (np.sqrt(alpha[0]) * CH[0,6]))
    hat_sum_theta[6] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[6] ).coef_
    # Node 7 as Rx: from node 0 
    y[7] = np.sqrt(alpha[0]) * A @ (sparse_EC_flattened_theta_by_devices[0]) * CH[0,7] + noise[7] 
    hat_y[7] = np.real(y[7] / (np.sqrt(alpha[0]) * CH[0,7]))
    hat_sum_theta[7] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[7] ).coef_

    Lasso_error = [ np.mean(np.abs(hat_sum_theta[i] - sum(sparse_EC_flattened_theta_by_devices[j] for j in G[i])) ** 2) for i in range(K) ]
    print("Normalized MSE(dB):", "|".join("{:.2f}".format(10 * np.log10(Lasso_error[i])) for i in range(K)))

    # flattened_tilde_theta_by_devices turns out to be a list (device_i's) of combined theta_i's used for the consensus step)
    alpha = W[0][[j for j in G[0]][0]]
    flattened_tilde_theta_by_devices = np.array([(1-G.degree[i] * alpha) * flattened_theta_by_devices[i] + alpha * hat_sum_theta[i]
                                                    for i in range(K)])    

    # Unflatten the combined theta_i's in a list (device_i's) of [theta_{i}^W, theta_{i}^b]
    tilde_theta_by_devices = [[flattened_tilde_theta_by_devices[i][:7840].reshape((784, 10)),
                               flattened_tilde_theta_by_devices[i][7840:]] for i in range(K)]

    for i in range(K):
        acc_error[i] += (flattened_theta_by_devices[i] - sparse_EC_flattened_theta_by_devices[i])

    return tilde_theta_by_devices, acc_error




def sp_case2(G, CH, W, flattened_theta_by_devices, N, A, N0, noise, acc_error, sr, estimator = 'Lasso', sigma_square = .01, lamda = 1e-6, K = 8, d = 7850):
    barP = .001
    EC_flattened_theta_by_devices = [flattened_theta_by_devices[i] + acc_error[i] for i in range(K)]
    ########## for flat matrix U (sparse-recovery solution) ############
    k = int((1- sr ** (1 / K)) * d )
    sparse_EC_flattened_theta_by_devices = np.zeros(flattened_theta_by_devices.shape) # A list (device_i's) of quantized theta_{i}'s that device i prepares to send to its neighbours
    for i in range(K):
        idx = np.argsort(np.abs(EC_flattened_theta_by_devices[i]))[d-k:]  # Index for the Top-q entries
        sparse_EC_flattened_theta_by_devices[i][idx] = EC_flattened_theta_by_devices[i][idx]

    ########## Calculate gamma and alpha ############
    num_transmit_ota = [2,1,0,2,0,2,3,1]
    num_transmit_bc = [0,1,1,1,1,0,0,1]
    # Define N_j^o as a subset of each device's neighboring nodes that are scheduled as the star center during the scheduling algorithm
    neighbor_o = [(4,1), (4,), (), (2,1), (), (3,7), (2,7,3), (4,)]
    # transmit power scaling factor gamma taking into account all the slots that each device transmits to a "star" center
    gamma = min( 1 / (np.linalg.norm(A @ sparse_EC_flattened_theta_by_devices[j],2) ** 2 * sum( 1 / np.abs(CH[i][j]) ** 2 for i in neighbor_o[j])) 
                    * barP * N * (num_transmit_ota[j] / (num_transmit_ota[j] + num_transmit_bc[j])) for j in range(K) if len(neighbor_o[j]) > 0)
    # transmit power scaling factor alpha taking into account the slot (maximum 1) that each device transmits as a "star" center
    alpha = [ 1 / np.linalg.norm(A @ sparse_EC_flattened_theta_by_devices[j],2) ** 2 
                    * barP * N * (num_transmit_bc[j] / (num_transmit_ota[j] + num_transmit_bc[j])) for j in range(K)]
    
    ########## Combine received signals from multiple slots at each device ############
    # Intialize the received signal y, the noisy observation hat_y (A @ sum() + N_0), and the recovered sum of signals hat_sum_theta (sum())
    y = np.zeros((K, A.shape[0]), dtype = np.complex128)
    hat_y = np.zeros((K, A.shape[0]))
    hat_sum_theta = np.zeros((K, A.shape[1]))
    # Node 0 as Rx: from node 4; and node 1
    y[0] = np.sqrt(alpha[4]) * A @ (sparse_EC_flattened_theta_by_devices[4]) * CH[4,0] + noise[0] 
    hat_y[0] = np.real(y[0] / (np.sqrt(alpha[4]) * CH[4,0]))
    hat_sum_theta[0] = Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[0] ).coef_ 

    y[0] = np.sqrt(alpha[1]) * A @ (sparse_EC_flattened_theta_by_devices[1]) * CH[1,0] + noise[0] 
    hat_y[0] = np.real(y[0] / (np.sqrt(alpha[1]) * CH[1,0]))
    hat_sum_theta[0] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[0] ).coef_ 
    # Node 1 as Rx: from node 4; and node 0, 3
    y[1] = np.sqrt(alpha[4]) * A @ (sparse_EC_flattened_theta_by_devices[4]) * CH[4,1] + noise[1] 
    hat_y[1] = np.real(y[1] / (np.sqrt(alpha[4]) * CH[4,1]))
    hat_sum_theta[1] = Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[1] ).coef_

    y[1] = np.sqrt(gamma) * A @ (sparse_EC_flattened_theta_by_devices[0] + sparse_EC_flattened_theta_by_devices[3]) + noise[1] 
    hat_y[1] = np.real(y[1]) / np.sqrt(gamma)
    hat_sum_theta[1] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[1] ).coef_
    # Node 2 as Rx: from node 3, 6
    y[2] = np.sqrt(gamma) * A @ (sparse_EC_flattened_theta_by_devices[3] + sparse_EC_flattened_theta_by_devices[6]) + noise[1] 
    hat_y[2] = np.real(y[2]) / np.sqrt(gamma)
    hat_sum_theta[2] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[2] ).coef_
    # Node 3 as Rx: from node 2; node 1; and node 5, 6
    y[3] = np.sqrt(alpha[2]) * A @ (sparse_EC_flattened_theta_by_devices[2]) * CH[2,3] + noise[3] 
    hat_y[3] = np.real(y[3] / (np.sqrt(alpha[2]) * CH[2,3]))
    hat_sum_theta[3] = Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[3] ).coef_

    y[3] = np.sqrt(alpha[1]) * A @ (sparse_EC_flattened_theta_by_devices[1]) * CH[1,3] + noise[3] 
    hat_y[3] = np.real(y[3] / (np.sqrt(alpha[1]) * CH[1,3]))
    hat_sum_theta[3] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[3] ).coef_

    y[3] = np.sqrt(gamma) * A @ (sparse_EC_flattened_theta_by_devices[5] + sparse_EC_flattened_theta_by_devices[6]) + noise[3] 
    hat_y[3] = np.real(y[3]) / np.sqrt(gamma)
    hat_sum_theta[2] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[3] ).coef_
    # Node 4 as Rx: from node 0, 1, 7
    y[4] = np.sqrt(gamma) * A @ (sparse_EC_flattened_theta_by_devices[0] + sparse_EC_flattened_theta_by_devices[1] + sparse_EC_flattened_theta_by_devices[7]) + noise[4] 
    hat_y[4] = np.real(y[4]) / np.sqrt(gamma)
    hat_sum_theta[4] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[4] ).coef_
    # Node 5 as Rx: from node 7; and node 3
    y[5] = np.sqrt(alpha[7]) * A @ (sparse_EC_flattened_theta_by_devices[7]) * CH[7,5] + noise[5] 
    hat_y[5] = np.real(y[5] / (np.sqrt(alpha[7]) * CH[7,5]))
    hat_sum_theta[5] = Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[5] ).coef_

    y[5] = np.sqrt(alpha[3]) * A @ (sparse_EC_flattened_theta_by_devices[3]) * CH[3,5] + noise[5] 
    hat_y[5] = np.real(y[5] / (np.sqrt(alpha[3]) * CH[3,5]))
    hat_sum_theta[5] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[5] ).coef_
    # Node 6 as Rx: from node 2; node 7; and node 3
    y[6] = np.sqrt(alpha[2]) * A @ (sparse_EC_flattened_theta_by_devices[2]) * CH[2,6] + noise[6] 
    hat_y[6] = np.real(y[6] / (np.sqrt(alpha[2]) * CH[2,6]))
    hat_sum_theta[6] = Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[6] ).coef_

    y[6] = np.sqrt(alpha[7]) * A @ (sparse_EC_flattened_theta_by_devices[7]) * CH[7,6] + noise[6] 
    hat_y[6] = np.real(y[6] / (np.sqrt(alpha[7]) * CH[7,6]))
    hat_sum_theta[6] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[6] ).coef_

    y[6] = np.sqrt(alpha[3]) * A @ (sparse_EC_flattened_theta_by_devices[3]) * CH[3,6] + noise[6] 
    hat_y[6] = np.real(y[6] / (np.sqrt(alpha[3]) * CH[3,6]))
    hat_sum_theta[6] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[6] ).coef_
    # Node 7 as Rx: from node 4; and node 5, 6
    y[7] = np.sqrt(alpha[4]) * A @ (sparse_EC_flattened_theta_by_devices[4]) * CH[4,7] + noise[7] 
    hat_y[7] = np.real(y[7] / (np.sqrt(alpha[4]) * CH[4,7]))
    hat_sum_theta[7] = Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[7] ).coef_

    y[7] = np.sqrt(gamma) * A @ (sparse_EC_flattened_theta_by_devices[5] + sparse_EC_flattened_theta_by_devices[6]) + noise[7] 
    hat_y[7] = np.real(y[7]) / np.sqrt(gamma)
    hat_sum_theta[7] += Lasso(alpha = lamda, fit_intercept = False).fit( A, hat_y[7] ).coef_

    Lasso_error = [ np.mean(np.abs(hat_sum_theta[i] - sum(sparse_EC_flattened_theta_by_devices[j] for j in G[i])) ** 2) for i in range(K) ]
    print("Normalized MSE(dB):", "|".join("{:.2f}".format(10 * np.log10(Lasso_error[i])) for i in range(K)))

    # flattened_tilde_theta_by_devices turns out to be a list (device_i's) of combined theta_i's used for the consensus step)
    alpha = W[0][[j for j in G[0]][0]]
    flattened_tilde_theta_by_devices = np.array([(1-G.degree[i] * alpha) * flattened_theta_by_devices[i] + alpha * hat_sum_theta[i]
                                                    for i in range(K)])    

    # Unflatten the combined theta_i's in a list (device_i's) of [theta_{i}^W, theta_{i}^b]
    tilde_theta_by_devices = [[flattened_tilde_theta_by_devices[i][:7840].reshape((784, 10)),
                               flattened_tilde_theta_by_devices[i][7840:]] for i in range(K)]

    for i in range(K):
        acc_error[i] += (flattened_theta_by_devices[i] - sparse_EC_flattened_theta_by_devices[i])

    return tilde_theta_by_devices, acc_error

    

