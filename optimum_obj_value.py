import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten #pylint: disable = import-error
# import matplotlib.pyplot as plt
import pickle
import pandas as pd
import networkx as net
from utils import solve_graph_weights, get_model, calc_grad, log2_comb, TwoSectionH, solve_num_per_class, seq_scheduling
# from scipy.linalg import hadamard
import digital_schemes as ds
# import analog_schemes as ans
# from scipy.sparse import identity, csr_matrix


def train( a, initial_cr ):

    (train_images, train_labels), (test_images,
                                test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 5100.0
    test_images = test_images / 5100.0
    
    np.random.seed(2)
    K = 20
    num_lack_max = 5
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

    # BATCH_SIZE = 64
    # datasets = [
    #     tf.data.Dataset.from_tensor_slices( ( train_images[ sample_indices[i] ], train_labels[ sample_indices[i] ] ) ).shuffle(8000).batch( BATCH_SIZE ).repeat() 
    #         for i in range(K)
    # ]
    # zip_ds = tf.data.Dataset.zip( tuple(datasets) )



    com_interval = 1 # Also known as "H" in the SPARQ-SGD paper
    training_times = 1 # There will be training_times of lists each with a different setup of blockages, each of which is of (ComRound, K)
    Tmax = 5000 # Maximum number of commun. rounds during each training session

    LOCAL = False
    # scheme = 5
    # Generate a learning rate scheduler that returns initial_learning_rate / (1 + decay_rate * t / decay_step)
    decayed_learning = False
    mu = 0.002 # assuming eta = (3.25/mu/a) / (1 + t/a), where mu = 2*lamda
    b = 3.25/mu # 3.25/mu = 1625
    initial_lr = b / a
    decay_steps = a
    decay_rate = 1
    learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(initial_lr,
                                                                    decay_steps, decay_rate)
   
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    acc_fn = keras.metrics.SparseCategoricalAccuracy()

    seeds = iter(range(1000))
    p = 0.2 # The probability that one edge is included in the connectivity graph as per the Erdos-Renyi (random) graph
    # losseses and accses are of shape (training_times, ComRound, K)
    tr_losseses, tst_accses = [[] for i in range(training_times)] , [[] for i in range(training_times)]



    for n in range(training_times): # n is the index for the times of training
        print("The {}th time of training:".format(n))
        # Generate a random graph model suffering from blockages under Erdos Renyi Model
        alg_connect = 0
        while alg_connect < 1e-4:
            # # Generate a star-based ER graph
            # ER = net.erdos_renyi_graph(K-1, p, seed = next(seeds))
            # ER.add_node(K-1)
            # G = net.star_graph(reversed(range(K)))
            # G.add_edges_from(ER.edges())

            # Generate an arbitrary ER graph
            G = net.erdos_renyi_graph(K, p, seed = next(seeds))

            # # Generate a 2-D torus (5-by-4)
            # G = net.grid_2d_graph(5, 4, periodic=True)
            # mapping = { (m,n):4*m+n for m, n in G.nodes()}
            # _ = net.relabel_nodes(G, mapping, copy=False)

            # # Generate a complete graph
            # G = net.complete_graph(K)

            L = np.array(net.laplacian_matrix(G, nodelist = range(K)).todense())
            D, _ = np.linalg.eigh(L) # eigenvalues are assumed given in an ascending order
            alg_connect = D[1] 
        # W, _ = solve_graph_weights(K, E)
        alpha = 2 / (D[K-1] + D[1])
        W = np.eye(K) - alpha * L
        # _, Chi = TwoSectionH(G) 

        models = [
            get_model('Device_{}'.format(i), (28, 28), lamda = 0.001, flag = False) for i in range(K)
        ]

        if decayed_learning:
            opts = [tf.keras.optimizers.SGD(learning_rate = learning_rate_fn, momentum = 0.9) for i in range(K)]
        else:
            opts = [tf.keras.optimizers.SGD(learning_rate = initial_lr, momentum = 0.9) for i in range(K)]
        
        for t in range(Tmax): # t is the index for the SGD iteration    
            # Keep track of the model parameters in tensors as [\theta_{i,W}, \theta_{i,b}]
            var_lists_by_devices = []

            for i in range(K): # i is the index for the device (node of graph)
                model = models[i]
                # batch_images, batch_labels = sample_data(i, t, train_images, train_labels)
                batch_images, batch_labels = train_images[ sample_indices[i] ], train_labels[ sample_indices[i] ]
                # Compute the gradients for a list of variables.
                tr_loss, var_list, grads = calc_grad(model, batch_images, batch_labels)
                # local model updates
                # opt.apply_gradients( [ ( grad0, var0 ), (grad1, var1), ..., (grad_n, var_n) ] )
                clipped_grads, _ = tf.clip_by_global_norm(grads, 0.15)
                opts[i].apply_gradients(zip(clipped_grads, var_list))

                var_lists_by_devices.append(var_list)

            if not (t % com_interval) and not LOCAL:
                ############ vanila DSGD ################
                theta_next_by_devices = ds.vanila_DSGD(var_lists_by_devices, W, initial_cr)
            else: # local training step
                theta_next_by_devices = var_lists_by_devices


            
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
                tr_losseses[n].append(tr_losses)
                # keep track of the test accuracy per iteration
                tst_accses[n].append(tst_accs.numpy())

                # print("Round{}".format(t // com_interval), "|".join("{:.3f}".format(x)
                #                     for x in tr_losses))
                print("Round{}: loss function {:.3f}".format(t // com_interval, sum(tr_losses) / K ))
                # print("Round{}".format(t // com_interval), "|".join("{:.4f}".format(x)
                #                     for x in tst_accs))
                print("Round{}: accuracy level {:.4f}".format(t // com_interval, tst_accs))


 
        import sys
        if sys.platform == 'win32':
            path = './data/simulations/'
        else:
            path = '/scratch/users/k1818742/data/'

        # with open('{}grad_normses_SCHEME_{}.pkl'.format(path, scheme), 'wb') as grads:
        #     pickle.dump(grad_normses, grads)
        with open('{}losseses_SCHEME_{}_rho_a_{:.2f}_zeta0_{:.2f}.pkl'.format(path, 1, a, initial_cr), 'wb') as output1:
            pickle.dump(tr_losseses, output1)
        with open('{}accses_SCHEME_{}_rho_a_{:.2f}_zeta0_{:.2f}.pkl'.format(path, 1, a, initial_cr), 'wb') as output2:
            pickle.dump(tst_accses, output2)

    # scp -r k1818742@login.rosalind.kcl.ac.uk:/scratch/users/k1818742/data/*.pkl /home/Helen/MyDocuments/visiting_research@KCL/D2D_DSGD/repo_jv/data/

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=800)
    parser.add_argument('--zeta0', type=float, default=1)

    args = parser.parse_args()

    train( args.a, args.zeta0 )