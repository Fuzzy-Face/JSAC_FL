import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten #pylint: disable = import-error
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import networkx as net
from utils import solve_graph_weights, get_model, calc_grad, log2_comb, TwoSectionH, solve_num_per_class, seq_scheduling
from scipy.linalg import hadamard
import digital_schemes as ds
import analog_schemes as ans
# from scipy.sparse import identity, csr_matrix


def train( scheme, top, P, N, a, initial_cr, a_prime, nth ):

    (train_images, train_labels), (test_images,
                                test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 5100.0
    test_images = test_images / 5100.0
    
    # Normalize the data sample by the square root of the sum of all element-wise square of the (#sample, 28, 28) matrix
    # train_images = train_images / np.sqrt(np.sum(train_images ** 2))
    # test_images = test_images / np.sqrt(np.sum(train_images ** 2))
    # train_images = train_images / np.linalg.norm(train_images, ord = 'fro', axis = (1,2), keepdims = True)
    # test_images = test_images / np.linalg.norm(test_images, ord = 'fro', axis = (1,2), keepdims = True)

    np.random.seed(2)
    K = 20
    num_lack_max = 4
    X, num_per_class, _ = solve_num_per_class(num_lack_max,K) # X is the 10-by-K indicator matrix indicating if data samples from class n is assigned to device K
    num_per_class = num_per_class.astype(int)

    sample_indices = [ [] for i in range(K) ]
    # num_per_class shows how many samples are there for each class (equal across classes for the same device) on each device
    # num_per_class.dtype returns the current data type of each element of an ndarray
    
    for n in range(10):
        # extract the indices for class n
        indices = np.nonzero( train_labels == n )[0]
        assigned = 0 # mark the position of assigned indices for each class of samples
        for k in range(K):
            if X[n,k]:
                sub_indices = indices[assigned:assigned+num_per_class[k]].tolist()
                assigned += num_per_class[k]
                sample_indices[ k ] += sub_indices
    # Convert to one-hot vector
    # train_labels = tf.squeeze(tf.one_hot(train_labels, 10))
    # test_labels = tf.squeeze(tf.one_hot(test_labels, 10))
    BATCH_SIZE = 64
    datasets = [
        tf.data.Dataset.from_tensor_slices( ( train_images[ sample_indices[i] ], train_labels[ sample_indices[i] ] ) ).shuffle(8000).batch( BATCH_SIZE ).repeat() 
            for i in range(K)
    ]
    zip_ds = tf.data.Dataset.zip( tuple(datasets) )

    com_interval = 1 # Also known as "H" in the SPARQ-SGD paper
    training_times = 10 # There will be training_times of lists each with a different setup of blockages, each of which is of (ComRound, K)
    Tmax = 5000 # Maximum number of commun. rounds during each training session
    # T = 1
    # BW = .5 * 1e4
    # N0 = 10 ** (-169/10) * 1e-3  # power spectral density of the AWGN noise per channel use
    # N0 = 0
    # N = T * BW

    LOCAL = False
    # scheme = 5
    # Generate a learning rate scheduler that returns initial_learning_rate / (1 + decay_rate * t / decay_step)
    decayed_learning = True
    mu = 0.002 # assuming eta = (3.25/mu/a) / (1 + t/a), where mu = 2*lamda
    b = 3.25/mu # 3.25/mu = 1625
    initial_lr = b / a
    decay_steps = a
    decay_rate = 1
    learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(initial_lr,
                                                                    decay_steps, decay_rate)
    decayed_cr = True
    # initial_cr = 0.01
    # a = 5.0
    cs_rate_fn = lambda t: initial_cr / (1 + t/a_prime)

    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    acc_fn = keras.metrics.SparseCategoricalAccuracy()

    seeds = iter(range(1000))
    p = 0.1 # The probability that one edge is included in the connectivity graph as per the Erdos-Renyi (random) graph
    # losseses and accses are of shape (training_times, ComRound, K)
    tr_losseses, tst_accses, grad_normses = [[] for i in range(training_times)] , [[] for i in range(training_times)], [[] for i in range(training_times)]
    # # thetases is of shape (training_times, ComRound, K, d)
    # thetases = [[] for i in range(training_times)]
    if scheme == 6:
        # cons_errors/comp_errors is of shape (training_times, ComRound, 1)
        # noise_errors is of shape (training_times, ComRound, K)
        cons_errors, comp_errors, noise_errors = [[] for i in range(training_times)], [[] for i in range(training_times)], [[] for i in range(training_times)] 

    # for n in range(training_times): # n is the index for the times of training
    print("The {}th time of training:".format(nth))
    # Generate a (random) graph model suffering from blockages under Erdos Renyi Model
    if top == 'star-ER':
        # Generate a star-based ER graph
        ER = net.erdos_renyi_graph(K-1, p, seed = next(seeds))
        G = net.star_graph(reversed(range(K)))
        G.add_edges_from(ER.edges())
        
    elif top == 'star':
        # Generate a star graph
        G = net.star_graph(reversed(range(K)))
    
    elif top == 'ER':
        alg_connect = 0
        while alg_connect < 1e-4:
            # Generate an arbitrary ER graph
            G = net.erdos_renyi_graph(K, p, seed = next(seeds))
            L = np.array(net.laplacian_matrix(G, nodelist = range(K)).todense())
            E, _ = np.linalg.eigh(L) # eigenvalues are assumed given in an ascending order
            alg_connect = E[1] 

    elif top == 'chain-ER':
        # Generate a chain-based ER graph
        ER = net.erdos_renyi_graph(K, p, seed = next(seeds))
        G = net.cycle_graph(K)
        G.remove_edge(0, K-1)
        G.add_edges_from(ER.edges())

    elif top == 'chain':
        # Generate a chain (cycle) graph
        G = net.cycle_graph(K)
        G.remove_edge(0, K-1)

    elif top == 'torus':
        # Generate a 2-D torus (5-by-4)
        G = net.grid_2d_graph(5, 4, periodic=True)
        mapping = { (m,n):4*m+n for m, n in G.nodes()}
        _ = net.relabel_nodes(G, mapping, copy=False)

    elif top == 'CG':
        # Generate a complete graph
        G = net.complete_graph(K)
    # If there is no Eigenvalues for the Laplacian matrix that has been calculated
    #  
    try:
        alpha = 2 / (E[K-1] + E[1])
    except NameError:
        L = np.array(net.laplacian_matrix(G, nodelist = range(K)).todense())
        E, _ = np.linalg.eigh(L) # eigenvalues are assumed given in an ascending order
        alpha = 2 / (E[K-1] + E[1])
    W = np.eye(K) - alpha * L
    # _, Chi = TwoSectionH(G) 

    # Set distances among any pair of nodes
    # d_min = 50
    # d_max = 200
    # rho =  d_min + (d_max - d_min) * np.random.rand(K,1) 
    # theta = 2 * np.pi * np.random.rand(K,1)
    D = 125 * np.ones((K, K)) # mimic the effect of the same pathloss
    # D = np.sqrt(rho ** 2 + rho.T ** 2 - 2 * (rho @ rho.T) * np.cos(theta - theta.T))
    # # Fill in D[i,i] some non-zero value to avoid Inf in PL
    # for i in range(K):
    #     if i:
    #         D[i,i] = D[i,i-1]
    #     else:
    #         D[i,i] = D[i,i+1]
    A0 = 10 ** (-3.35)
    d0 = 1
    gamma = 3.76
    PL = A0 * ((D / d0) ** (-gamma)) 

    # Generate random channels for unblocked edges of the given graph
    np.random.seed(nth)
    CH_gen = np.random.randn(Tmax, K, K)/np.sqrt(2) + 1j * np.random.randn(Tmax, K, K)/np.sqrt(2)
    # with open('./data/simulations/channels_Tmax_{:.0f}_{:.0f}-{:.0f}.pkl'.format(Tmax, training_times, n), 'wb') as channels:
    #     pickle.dump(CH_gen, channels) 
    # with open('./data/simulations/channels_Tmax_{:.0f}_{:.0f}-{:.0f}.pkl'.format(Tmax, training_times, n), 'rb') as channels:
    #     CH_gen = pickle.load(channels) 

    models = [
        get_model('Device_{}'.format(i), (28, 28), lamda = 0.001, flag = False) for i in range(K)
    ]
    d = int(sum(theta.numpy().size for theta in models[0].trainable_weights))


    if scheme != 1 and not LOCAL:
        tilde_d = 2 ** 13
        H = hadamard(tilde_d)
        flattened_hat_theta_by_devices = [np.zeros((d,)) for i in range(K)] 

        if scheme == 4 or scheme == 6:
            schedule_list, Tx_times = seq_scheduling(G.copy())
            M = 2 * len(schedule_list)
            m = int(N / M) 
            H_par = H[:m] 
        elif scheme == 5:
            m = int(N / K) 
            H_par = H[:m]

        if scheme == 4 or scheme == 5 or scheme == 6:
            # Initialize the \hat{y}_i^{(t)}'s as zeros
            hat_y_by_devices = [np.zeros((d,)) for i in range(K)] 
            # Initialize the \theta_i^{(t)}'s as [\theta_{i,W}=np.zeros((784,10)), \theta_{i,b}=np.zeros((10,))]
            # for the purpose of keeping track of the consensus and the compression error
            theta_next_by_devices = [ [np.zeros((784,10)), np.zeros((10,))] for i in range(K)]  

        if scheme == 2:
            _, from_node_to_color_id = TwoSectionH(G)
            Chi = max(from_node_to_color_id.values()) + 1
            # Chi = 20 # only used for comparing performance with different topologies (thus a common M = Chi for fair comparison)

    #     A = np.random.randn(s, d) * np.sqrt(1 / d)
    #     if s >= d:
    #         A_dag = np.linalg.inv(A.T @ A) @ A.T
    #         eta = .01
    #     else:
    #         A_dag = None

    if decayed_learning:
        opts = [tf.keras.optimizers.SGD(learning_rate = learning_rate_fn, momentum = 0.9) for i in range(K)]
    else:
        opts = [tf.keras.optimizers.SGD(learning_rate = 2, momentum = 0.9) for i in range(K)]
    


    for t, batch_data in enumerate( zip_ds ): # t is the index for the SGD iteration    
        if t // com_interval >= Tmax:
            break

        # Extract per-iteration channel generator from the tth row of CH_gen
        CH_gen_current = CH_gen[t] 
        CH = np.ones((K, K), dtype=complex)  # Channel coefficients
        for i in range(K):
            for j in G[i]:
                if j < i:
                    CH[i, j] = CH_gen_current[i,j]
        for i in range(K):
            for j in G[i]:
                if j > i:
                    CH[i, j] = np.conjugate(CH[j, i])
        CH = np.sqrt(PL) * CH 
        
        if scheme == 4 or scheme == 5 or scheme == 6:
        # Keep track of the original theta_i^{(t)}'s at the beginning of the iteration t
            if isinstance(theta_next_by_devices[0][0], tf.Variable):
                flattened_theta_by_devices = [ np.ndarray.flatten(np.concatenate(( theta_next[0].numpy(),
                                                                                    theta_next[1].numpy().reshape(1,10) ), axis = 0))
                                                                            for theta_next in theta_next_by_devices ] 
            else:
                flattened_theta_by_devices = [ np.ndarray.flatten(np.concatenate(( theta_next[0],
                                                                theta_next[1].reshape(1,10) ), axis = 0))
                                                        for theta_next in theta_next_by_devices ] 

        # Keep track of \theta_i^{(t+1/2)}'s after the local model updates
        flattened_theta_half_by_devices = []
        # Keep track of the model parameters in tensors as [\theta_{i,W}, \theta_{i,b}]
        var_lists_by_devices = []
        # keep track of the 2-norm mof the gradients as [np.linalg.norm(flattened_grads,2)]
        grad_norm_by_devices = []
        for i in range(K): # i is the index for the device (node of graph)
            model = models[i]
            # batch_images, batch_labels = sample_data(i, t, train_images, train_labels)
            batch_images, batch_labels = batch_data[ i ]
            # Compute the gradients for a list of variables.
            tr_loss, var_list, grads = calc_grad(model, batch_images, batch_labels)
            # local model updates
            # opt.apply_gradients( [ ( grad0, var0 ), (grad1, var1), ..., (grad_n, var_n) ] )
            clipped_grads, grad_norm = tf.clip_by_global_norm(grads, 0.2)
            opts[i].apply_gradients(zip(clipped_grads, var_list))
            grad_norm_by_devices.append(grad_norm.numpy())

            var_lists_by_devices.append(var_list)
            # FLatten the updated model parameters in the shape of (7850,)
            flattened_theta_half_by_devices.append( np.ndarray.flatten(np.concatenate((var_list[0].numpy(),
                                                                                        var_list[1].numpy().reshape(1,10)), axis = 0)) )

        if not (t % com_interval) and not LOCAL:
            if decayed_cr:
                zeta = cs_rate_fn(t)
            else:
                zeta = initial_cr

            ############ vanila DSGD ################
            if scheme == 1:
                theta_next_by_devices = ds.vanila_DSGD(var_lists_by_devices, W, zeta)
            ############ D-DSGD based on 2-section hypergraph (finite channel use) ################
            elif scheme == 2:
                theta_next_by_devices, flattened_hat_theta_by_devices = ds.proposed_DSGD(G, flattened_theta_half_by_devices, flattened_hat_theta_by_devices, W, zeta, CH, N, Chi, H, P)
            ############ D-DSGD based on TDMA ################
            elif scheme == 3:
                theta_next_by_devices, flattened_hat_theta_by_devices = ds.proposed_DSGD(G, flattened_theta_half_by_devices, flattened_hat_theta_by_devices, W, zeta, CH, N, K, H, P)
            ############ A-DSGD based on dynamic coloring of the 2-section hypergraph ################
            elif scheme == 4:
                # noise =  np.sqrt(N0 / 2) * np.random.randn(K, s) + 1j * np.sqrt(N0 / 2) * np.random.randn(K, s)
                theta_next_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices = ans.proposed_DSGD(G, flattened_theta_by_devices, flattened_theta_half_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices, W, zeta, CH, N, schedule_list, Tx_times, H_par, P)
            ############ A-DSGD based on TDMA ################
            elif scheme == 5:
                theta_next_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices = ans.TDMA_DSGD(G, flattened_theta_half_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices, W, zeta, CH, N, H_par, P)
            ############ A-DSGD keeping track of the errors ################
            elif scheme == 6:
                theta_next_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices, cons_error, comp_error, noise_error_by_devices = ans.proposed_DSGD(G, flattened_theta_by_devices, flattened_theta_half_by_devices, flattened_hat_theta_by_devices, hat_y_by_devices, W, zeta, CH, N, schedule_list, Tx_times, H_par, P, flag = True)
                cons_errors[nth].append(cons_error)
                comp_errors[nth].append(comp_error)
                noise_errors[nth].append(noise_error_by_devices)

                print("Round{}: consensus error {:.2f}".format(t // com_interval, cons_error))
                # print("Round{}: compression error {:.2f}".format(t // com_interval, comp_error))
                df = pd.DataFrame(noise_errors[nth], columns = ['Device_{}'.format(i + 1) for i in range(K)] )
                # !!! Note that one m has already been multiplied by N0 when calculating noise_error_by_devices
                acc_noise_error = (zeta**2) * (m**2/d) * (df.sum().sum())
                print("Round{}: AWGN error {:.2f}".format(t // com_interval, acc_noise_error))
        else: # local training step
            theta_next_by_devices = var_lists_by_devices

        # Update the model parameters as theta_i^{(t+1)}'s at the end of the iteration t
        for var_list, thetas_next in zip(var_lists_by_devices, theta_next_by_devices):
            # Manually update the var_list to the aggregated value after the consensus update
            for theta, theta_next in zip(var_list, thetas_next):
                theta.assign(theta_next)  # theta := tilde_theta
        
        if not (t % com_interval):
            # Evaluate the individual model on the test set every com_interval
            # tr_losses, tst_accs = [], []
            # for i in range(K): # i is the index for the device (node of graph)
            #     model = models[i]
            #     tr_loss = loss_fn(train_labels[ sample_indices[i] ], model(train_images[ sample_indices[i] ]))
            #     tst_acc = acc_fn(test_labels, model(test_images))
            #     acc_fn.reset_states()
            #     tr_losses.append( tr_loss.numpy() ) 
            #     tst_accs.append( tst_acc.numpy() )

            # Evaluate the average model on the test set every com_interval
            avg_model = get_model('Avg_model', (28, 28), lamda = 0.001, flag = False)
            thetas_avg =[sum(weights_by_devices) / K for weights_by_devices in zip(*theta_next_by_devices)]
            var_list = avg_model.trainable_weights
            for theta, theta_avg in zip(var_list, thetas_avg):
                theta.assign(theta_avg)

            tr_losses = []
            for i in range(K):
                # the following training loss is evaluated over individual data sets using the average model
                tr_loss = loss_fn(train_labels [ sample_indices[i] ], avg_model(train_images[ sample_indices[i] ]))
                tr_losses.append( tr_loss.numpy() )
            tst_accs = acc_fn(test_labels, avg_model(test_images)) # the same across all devices over the test data set
            acc_fn.reset_states()

            # keep track of the training losses by devices (K,) per iteration
            tr_losseses[nth].append(tr_losses)
            # keep track of the test accuracy per iteration
            tst_accses[nth].append(tst_accs.numpy())

            # print("Round{}".format(t // com_interval), "|".join("{:.3f}".format(x)
            #                     for x in tr_losses))
            print("Round{}: loss function {:.3f}".format(t // com_interval, sum(tr_losses) / K ))
            # print("Round{}".format(t // com_interval), "|".join("{:.4f}".format(x)
            #                     for x in tst_accs))
            print("Round{}: accuracy level {:.4f}".format(t // com_interval, tst_accs))

            # keep track of the 2-norm gradients by devices (K,) per iteration
            grad_normses[nth].append(grad_norm_by_devices)


    import sys
    if sys.platform == 'win32':
        path = './data/simulations/'
    else:
        path = '/scratch/users/k1818742/data/'

    # with open('{}grad_normses_SCHEME_{}.pkl'.format(path, scheme), 'wb') as grads:
    #     pickle.dump(grad_normses, grads)
    with open('{}losseses_SCHEME_{}_P_{:.6f}mW_N_{:.0f}_a_{:.2f}_zeta0_{:.4f}_a_prime_{:.2f}_{}_equal_n-{:d}.pkl'.format(path, scheme, P*1e3, N, a, initial_cr, a_prime, top, nth), 'wb') as output1:
        pickle.dump(tr_losseses, output1)
    # with open('{}accses_SCHEME_{}_P_{:.6f}mW_N_{:.0f}_a_{:.2f}_zeta0_{:.4f}_a_prime_{:.2f}_{}_equal_n-{:d}.pkl'.format(path, scheme, P*1e3, N, a, initial_cr, a_prime, top, nth), 'wb') as output2:
    #     pickle.dump(tst_accses, output2)
    
    if scheme == 6:
        with open('{}cons_e_SCHEME_{}_P_{:.6f}mW_N_{:.0f}_a_{:.2f}_zeta0_{:.4f}_a_prime_{:.2f}_{}_n-{:d}.pkl'.format(path, scheme, P*1e3, N, a, initial_cr, a_prime, top, nth), 'wb') as output3:
            pickle.dump(cons_errors, output3)
        with open('{}comp_e_SCHEME_{}_P_{:.6f}mW_N_{:.0f}_a_{:.2f}_zeta0_{:.4f}_a_prime_{:.2f}_{}_n-{:d}.pkl'.format(path, scheme, P*1e3, N, a, initial_cr, a_prime, top, nth), 'wb') as output4:
            pickle.dump(comp_errors, output4)
        with open('{}noise_e_SCHEME_{}_P_{:.6f}mW_N_{:.0f}_a_{:.2f}_zeta0_{:.4f}_a_prime_{:.2f}_{}_n-{:d}.pkl'.format(path, scheme, P*1e3, N, a, initial_cr, a_prime, top, nth), 'wb') as output5:
            pickle.dump(noise_errors, output5)

    # scp -r k1818742@login.rosalind.kcl.ac.uk:/scratch/users/k1818742/data/*.pkl /home/Helen/MyDocuments/visiting_research@KCL/D2D_DSGD/repo_jv/data/

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme', type=int, default=5)
    parser.add_argument('--topology', type=str, default='chain')
    parser.add_argument('--P', type=float, default=2e-7)
    parser.add_argument('--N', type=float, default=20000)
    parser.add_argument('--a', type=float, default=1000)
    parser.add_argument('--zeta0', type=float, default=0.1)
    parser.add_argument('--a_prime', type=float, default=500)
    parser.add_argument('--nth', type=int, default=1)

    args = parser.parse_args()

    train( args.scheme, args.topology, args.P, args.N, args.a, args.zeta0, args.a_prime, args.nth )



if __name__ == "__main__":

    print("TensorFlow version: {}".format(tf.__version__)) #pylint: disable = no-member
    # tf.compat.v1.enable_eager_execution()
    # print("Eager execution: {}".format(tf.executing_eagerly()))
    np.set_printoptions(suppress=True)
    tf.keras.backend.set_floatx('float64')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    main()

