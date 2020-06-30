import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten #pylint: disable = import-error
import matplotlib.pyplot as plt
import pickle
import networkx as net
from utils import solve_graph_weights, get_model, calc_grad, log2_comb, TwoSectionH, solve_num_per_class
import digital_schemes as ds
import analog_schemes as ans
from scipy.sparse import identity, csr_matrix



if __name__ == "__main__":

    print("TensorFlow version: {}".format(tf.__version__)) #pylint: disable = no-member
    # tf.compat.v1.enable_eager_execution()
    print("Eager execution: {}".format(tf.executing_eagerly()))
    np.set_printoptions(suppress=True)

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

    tf.keras.backend.set_floatx('float64')

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
    # Convert to one-hot  vector
    # train_labels = tf.squeeze(tf.one_hot(train_labels, 10))
    # test_labels = tf.squeeze(tf.one_hot(test_labels, 10))
    BATCH_SIZE = 64
    datasets = [
        tf.data.Dataset.from_tensor_slices( ( train_images[ sample_indices[i] ], train_labels[ sample_indices[i] ] ) ).shuffle(8000).batch( BATCH_SIZE ).repeat() 
            for i in range(K)
    ]
    zip_ds = tf.data.Dataset.zip( tuple(datasets) )

    # Set distances among any pair of nodes
    d_min = 50
    d_max = 200
    rho =  d_min + (d_max - d_min) * np.random.rand(K,1) 
    theta = 2 * np.pi * np.random.rand(K,1)
    # D = np.ones((K, K))
    D = np.sqrt(rho ** 2 + rho.T ** 2 - 2 * (rho @ rho.T) * np.cos(theta - theta.T))
    # Fill in D[i,i] some non-zero value to avoid Inf in PL
    for i in range(K):
        if i:
            D[i,i] = D[i,i-1]
        else:
            D[i,i] = D[i,i+1]
    A0 = 10 ** (-3.35)
    d0 = 1
    gamma = 3.76
    PL = A0 * ((D / d0) ** (-gamma))

    com_interval = 1 # Also known as "H" in the SPARQ-SGD paper
    training_times = 1 # There will be training_times of lists each with a different setup of blockages, each of which is of (ComRound, K)
    p = 0.2 # The probability that one edge is included in the connectivity graph as per the Erdos-Renyi (random) graph
    T = 1
    BW = 10 * 1e4
    N0 = 10 ** (-169/10) * 1e-3  # power spectral density in W
    # N0 = 0
    N = T * BW

    LOCAL = False
    scheme = 4
    SCHEME = scheme
    # Generate a learning rate scheduler that returns initial_learning_rate / (1 + decay_rate * t / decay_step)
    decayed_learning = True
    mu = 0.002 # assuming eta = (4/mu/a) / (1 + t/a), where mu = 2*lamda
    b = 4/mu # 4/mu
    a = 30
    initial_lr = b/a
    decay_steps = a
    decay_rate = 1
    learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(initial_lr,
                                                                    decay_steps, decay_rate)

    zeta = 1

    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    acc_fn = keras.metrics.SparseCategoricalAccuracy()

    seeds = iter(range(1000))
    # losseses and accses are as if of shape (training_times, ComRound, K)
    tr_losseses, tst_accses = [[] for i in range(training_times)] , [[] for i in range(training_times)]
    # # thetases is of shape (training_times, ComRound, K, d)
    # thetases = [[] for i in range(training_times)]

    for n in range(training_times): # n is the index for the times of training
        # Generate a random graph model suffering from blockages under Erdos Renyi Model
        print("The {}th time of training:".format(n))
        alg_connect = 0
        while alg_connect < 1e-4:
            G = net.erdos_renyi_graph(K, p, seed = next(seeds))
            # # Construct a complete graph with K nodes
            # G = net.complete_graph(K)
            # E = [(i+1, j+1) for i,j in G.edges()]

            # # Construct a ring graph with K nodes
            # E =  [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (1,8)]
            # G = net.Graph()
            # G.add_edges_from(E)

            # # Construct a star graph with K nodes
            # E =  [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7)]
            # G = net.Graph()
            # G.add_edges_from(E)
            # E = [(i+1, j+1) for i, j in G.edges()]
            L = np.array(net.laplacian_matrix(G).todense())
            D, _ = np.linalg.eigh(L) # eigenvalues are assumed given in an ascending order
            alg_connect = D[1] 
        # W, _ = solve_graph_weights(K, E)
        alpha = 2 / (D[K-1] + D[1])
        W = np.eye(K) - alpha * L
        _, Chi = TwoSectionH(G) 



        models = [
            get_model('Device_{}'.format(i), input_shape=(28, 28), lamda = 0.001) for i in range(K)
        ]
        d = int(sum(theta.numpy().size for theta in models[0].trainable_weights))

        log2_comb_list = [np.ceil(log2_comb(d,q)) for q in range(d + 1)]
        
        # Adapt the learning rate to the recoverying algorithm employed by analog implementation (if any)
        if scheme == 5 or scheme == 6:
            s = int (N / Chi) * (scheme == 5) + int(N / K) * (scheme == 6)
            A = np.random.randn(s, d) * np.sqrt(1 / d)
            if s >= d:
                A_dag = np.linalg.inv(A.T @ A) @ A.T
                eta = .01
            else:
                A_dag = None

        if decayed_learning:
            opts = [tf.keras.optimizers.SGD(learning_rate = learning_rate_fn) for i in range(K)]
        else:
            opts = [tf.keras.optimizers.SGD(learning_rate = 5) for i in range(K)]



        for t, batch_data in enumerate( zip_ds ): # t is the index for the SGD iteration 
            
            if t // com_interval >= 3000:
                break
            # Generate per-iteration channels following Rayleigh fading
            CH_gen = iter( np.random.randn(len(G.edges()),)/np.sqrt(2) + 1j * np.random.randn(len(G.edges()),)/np.sqrt(2) )
            CH = np.ones((K, K), dtype=complex)  # Channel coefficients
            for i in range(K):
                for j in G[i]:
                    if j < i:
                        CH[i, j] = next(CH_gen)
            for i in range(K):
                for j in G[i]:
                    if j > i:
                        CH[i, j] = np.conjugate(CH[j, i])
            CG = PL * (np.abs(CH)) ** 2 

            # Compute the gradients for a list of variables.
            var_lists_by_devices = []
            for i in range(K): # i is the index for the device (node of graph)
                model = models[i]
                # batch_images, batch_labels = sample_data(
                #     i, t, train_images, train_labels)
                batch_images, batch_labels = batch_data[ i ]
                tr_loss, var_list, grads = calc_grad(model, batch_images, batch_labels)
                # grads_by_Devices.append(grads)
                # opt.apply_gradients( [ ( grad0, var0 ), (grad1, var1), ..., (grad_n, var_n) ] )
                opts[i].apply_gradients(zip(grads, var_list))
                var_lists_by_devices.append(var_list)



            flattened_theta_by_devices = [ np.ndarray.flatten(np.concatenate((var_list[0].numpy(),var_list[1].numpy().reshape(1,10)), 
                                                                                        axis = 0)) for var_list in var_lists_by_devices ] # A list by devices of flattened parameters with shape of (7850,)
            # flattened_theta_by_devices = np.array(flattened_theta_by_devices) # flattened_theta_by_devices is now an NDarray of shape (8, 7850)
            if SCHEME != 1 and not LOCAL:
                flattened_hat_theta_by_devices = [np.zeros(sum(theta.numpy().size for theta in models[i].trainable_weights)) for i in range(K)] 

            if not (t % com_interval) and not LOCAL:
                    ############ vanila DSGD ################
                if SCHEME == 1:
                    theta_next_by_devices = ds.vanila_DSGD(var_lists_by_devices, W, zeta)
                    ############ D-DSGD upper-bound (indefinite channel use) ################
                elif SCHEME == 2:
                    theta_next_by_devices, flattened_hat_theta_by_devices = ds.ub_DSGD(G, flattened_theta_by_devices, flattened_hat_theta_by_devices, W, zeta)
                    ############ D-DSGD based on 2-section hypergraph (finite channel use) ################
                elif SCHEME == 3:
                    theta_next_by_devices, flattened_hat_theta_by_devices = ds.proposed_DSGD(G, flattened_theta_by_devices, flattened_hat_theta_by_devices, W, zeta, CG, N, Chi, N0, log2_comb_list)
                    ############ D-DSGD based on TDMA ################
                elif SCHEME == 4:
                    theta_next_by_devices, flattened_hat_theta_by_devices = ds.TDMA_DSGD(G, flattened_theta_by_devices, flattened_hat_theta_by_devices, W, zeta, CG, N, N0, log2_comb_list)
                    ############ A-DSGD based on reverse 2-section hypergraph ################
                elif SCHEME == 5:
                    noise =  np.sqrt(N0 / 2) * np.random.randn(K, s) + 1j * np.sqrt(N0 / 2) * np.random.randn(K, s)
                    theta_next_by_devices, flattened_hat_theta_by_devices = ans.Rx_DSGD(G, CH, W, flattened_theta_by_devices, N, A, A_dag, N0, noise, flattened_hat_theta_by_devices, sparse_ratio, estimator = est, sigma_square = sigma_square, lamda = lamda)
                    ############ A-DSGD based on TDMA ################
                elif SCHEME == 6:
                    noise =  np.sqrt(N0 / 2) * np.random.randn(K, s) + 1j * np.sqrt(N0 / 2) * np.random.randn(K, s)
                    theta_next_by_devices, flattened_hat_theta_by_devices = ans.Rx_DSGD(G, CH, W, flattened_theta_by_devices, N, A, A_dag, N0, noise, flattened_hat_theta_by_devices, sigma_square, est)
            else: # local training step
                theta_next_by_devices = var_lists_by_devices

            # Loop over Devices
            for var_list, thetas_next in zip(var_lists_by_devices, theta_next_by_devices):
                # Manually update the var_list to the aggregated value after the consensus update
                for theta, theta_next in zip(var_list, thetas_next):
                    theta.assign(theta_next)  # theta := tilde_theta
            
            # Evaluate the current model on the test set every com_interval
            if not (t % com_interval):
                tr_losses, tst_accs = [], []
                for i in range(K): # i is the index for the device (node of graph)
                    model = models[i]
                    tr_loss = loss_fn(train_labels, model(train_images))
                    tst_acc = acc_fn(test_labels, model(test_images))
                    acc_fn.reset_states()
                    tr_losses.append( tr_loss.numpy() ) 
                    tst_accs.append( tst_acc.numpy() )

                tr_losseses[n].append(tr_losses)
                tst_accses[n].append(tst_accs)
            
                print("Round{}".format(t // com_interval), "|".join("{:.3f}".format(x)
                                    for x in tr_losses))
                print("Round{}:{:.3f}".format(t // com_interval, sum(tr_losses) / K ))

                print("Round{}".format(t // com_interval), "|".join("{:.4f}".format(x)
                                    for x in tst_accs))
                print("Round{}: {:.4f}".format(t // com_interval, sum(tst_accs) / K ))



        with open('./data/losseses_SCHEME_{}_eta0_{:.2f}_zeta_{:.2f}_10-{:d}.pkl'.format(SCHEME, initial_lr, zeta, n), 'wb') as output1:
            pickle.dump(tr_losseses, output1)
        with open('./data/accses_SCHEME_{}_eta0_{:.2f}_zeta_{:.2f}_10-{:d}.pkl'.format(SCHEME, initial_lr, zeta, n), 'wb') as output2:
            pickle.dump(tst_accses, output2)