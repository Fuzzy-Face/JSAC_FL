import cvxpy as cvx
import numpy as np
import tensorflow as tf
import networkx as net
import itertools as it
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten #pylint: disable = import-error
from scipy.stats import norm
from scipy.special import binom
from scipy.sparse import identity, spdiags
import time
# import vampyre as vp

def solve_graph_weights( K, E = None ):

    if E is None:
        E = []

    W = cvx.Variable( (K, K), symmetric = True )
    s = cvx.Variable( )
    A = np.ones( (K, K) ) / K

    constraints = [
        cvx.sum( W, axis = 1, keepdims = True) == np.ones((K,1)),
    ]

    # constraints += [ W[i,j] == 0. for i in range(K) for j in range(K) if (i+1, j+1) not in E and ( j+1, i+1 ) not in E and i != j ]
    for i in range(K):
        for j in range(K):
            if (i+1,j+1) not in E and (j+1,i+1) not in E and (i+1) != (j+1):
                constraints.append(W[i,j]==0)
            else:
                pass

    constraints += [ 
        W - A << s * np.eye(K),
        W - A >> - s * np.eye(K),
        s >= 0.,
    ]

    obj = cvx.Minimize( s )
    prob = cvx.Problem( obj, constraints )
    prob.solve( verbose = True, solver = cvx.CVXOPT )

    return W.value, prob.value



def get_model( name, input_shape, lamda = 1e-6 ):
    return keras.Sequential( [
        Flatten( input_shape = input_shape ),
        Dense( 10, activation = 'softmax', kernel_regularizer = keras.regularizers.l2( lamda ) ),
        ], name = name,
        )

def log2_comb(n, k):
    combN = range(n, n-k, -1)
    combD = range(k, 0, -1)
    log2_comb = np.sum(np.log2(combN)) - np.sum(np.log2(combD))
    return log2_comb


def dig_sparse_level(G, CG, N, Chi, N0, log2_comb_list, d = 7850):
    # s =  d / 3  # d\times 1 is the dimension of total number of parameters for each model, i.e., W.shape[0] * W.shape[1] + len(b)
    barP = .02 / N 
    q_array, b_array = [], []
    for i in range(CG.shape[0]):
        b = 64
        idx = np.nonzero(np.array(log2_comb_list) + b * np.arange(d + 1) <= 
        N / Chi * np.log2(1 + barP * Chi / N0 * min(CG[i,[j-1 for j in G[i]]])))[0]
        while not np.any(idx):
            if b == 16:
                print ("Fetal failure occurs!")
                b = 0
                idx = [ 0 ]
                break
            b = b / 2
            idx = np.nonzero(np.array(log2_comb_list) + b * np.arange(d + 1) <= 
                    N / Chi * np.log2(1 + barP * Chi / N0 * min(CG[i,[j-1 for j in G[i]]])))[0]
        if not np.any(idx):
            q_array.append(d)
        else:
            q_array.append(max(idx)) 
        b_array.append (b)
    #q_array = [max(np.nonzero(np.log2(binom_list) + b + range(min([np.ceil(d / 2) + 1, 140])) <= s / (2*2*Chi)*np.log2(1+P_t * CG[i-1][j-1] / s))[0]) for i,j in E] # SignTopK

    return q_array, b_array

     
def MyNeighbour(E, ii):
    neighbour = []
    for i, j in E:
        if ii == i or ii == j:
            neighbour.append(j * (ii == i) + i * (ii == j))

    return neighbour

def TwoSectionH(G): # Generate the 2-section of the proposed hypergraph, i.e., H2
    # VertexH = G.nodes()
    Hyperedge = [ tuple(sorted([node] + [n for n in G.neighbors(node)])) 
                                    for node in G.nodes() ] # construct a hypergraph each of whose hyperedge consists of a node and its neighbours
    Hyperedge = list(set(Hyperedge)) # remove any repeated hyperedges
    H2Edges = [[tuple(sorted(e)) for e in net.complete_graph(he).edges()] for he in Hyperedge] # a list of list of edges of H2, pylint: disable = undefined-variable
    temp = [] # construct the edge set of the H2
    for e in H2Edges:
        temp.extend(e) # remove the inner list delimiter "[]"
    H2Edges = list(set(temp)) # remove any repeated edges

    H2 = net.Graph() # pylint: disable = undefined-variable
    H2.add_nodes_from(G.nodes())
    H2.add_edges_from(H2Edges)
    vertex_color_map = net.greedy_color(H2, strategy = 'saturation_largest_first') # vertex coloring H2, pylint: disable = undefined-variable
    # Chi = max(vertex_color_map.values()) + 1 # chromatic number of the present coloring scheme

    return H2, vertex_color_map


# def  comp_quant_encoding(flattened_theta, acc_error, q_array, b_array, confidential = .995):
#     # flattened_theta is of data type "NDarray"
#     # acc_error is a list (device_i's) of different \Delta_{ji}'s, which records the difference between the actual theta_j and the quantized tilde_theta_{ji} that device j prepares to transmit to its neighbour i

#     #Below is a uniform_midrize_quantizer
#     # L = 2 ** b # L denotes the number of decision levels, and b denotes the number of quantized bits
#     # counts, bin_edges = np.histogram(theta, bins=20, density=True)
#     # cdf = numpy.cumsum(counts)

#     # theta_min = np.quantile(flattened_theta, 1-confidential, axis = 1)
#     # theta_max = np.quantile(flattened_theta, confidential, axis = 1)
#     # flattened_theta = np.maximum( flattened_theta, theta_min.reshape((8,1)))
#     # flattened_theta = np.minimum(flattened_theta, theta_max.reshape((8,1)))

#     # q_array = dig_sparse_level(E, CG) # Return the maximum "K" for SignTopK compression
#     # d = flattened_theta.shape[1]
#     # q_array = [  d // 2 ] * len(E) # [member, member, \ldots, member ]

#     # q_array = [ 202 ] * len(E) # [member, member, \ldots, member ] # Maximum permisitve number of q under s = d/3, \bar{SNR} = 50dB

#     quantized_theta = [] # A list (device_i's) of quantized theta_{i}'s that device i prepares to send to its neighbours
#     for i in range(flattened_theta.shape[0]):
#         q = q_array[i] 
#         b = b_array[q_array[i]]
#         EC_flattened_theta_i = flattened_theta[i] + acc_error[i] 
#         # theta_min = np.quantile(EC_flattened_theta_ji, 1-confidential)
#         # theta_max = np.quantile(EC_flattened_theta_ji, confidential)
#         # EC_flattened_theta_ji = np.maximum(EC_flattened_theta_ji, theta_min)
#         # EC_flattened_theta_ji = np.minimum(EC_flattened_theta_ji, theta_max)

#         idx = np.argsort(np.abs(EC_flattened_theta_i))[flattened_theta.shape[1]-q:]  # Index for the Top-q entries
            
#         # # The l_1 norm of the absolute value of the TopK entries
#         # theta_norm = np.sum(np.abs(flattened_theta[j-1,:][idx])) / q
#         # Qtheta_norm = np.floor((theta_norm - theta_min[j-1]) / Q[j-1]) * Q[j-1] + Q[j-1]/2 + theta_min[j-1]
#         # # the compressed theta_j that device j prepares to transmits to its neighbour i
#         # Qtheta = np.zeros(flattened_theta[i,:].shape)
#         # Qtheta[idx] = Qtheta_norm * np.sign(flattened_theta[j-1,:][idx]) # Keep the sign of each of the TopK entries
#         # tilde_theta += W[i,j-1] * Qtheta

#         # the quantized theta_i that adopts TopSign-q
#         Qtheta = np.zeros(flattened_theta[i].shape)
#         # The l_1 norm of the sign-dominate of the TopK entries
#         domi_theta = EC_flattened_theta_i[idx]
#         mask_pos = domi_theta >= 0
#         mask_neg = domi_theta < 0

#         mu_pos = np.mean(domi_theta[mask_pos]) if np.any( mask_pos ) else 0.0
#         mu_neg = np.mean(domi_theta[mask_neg]) if np.any( mask_neg ) else 0.0
#         if mu_pos >= -mu_neg:
#             theta_norm = mu_pos 
#             mask_domi = mask_pos
#         else:
#             theta_norm = mu_neg
#             mask_domi = mask_neg

#         # Quantization interval is equal to the shape of theta_max and theta_min, i.e., device-wise
#         # Q = (theta_max - theta_min) / L 
#         # Qtheta_norm = np.floor((theta_norm - theta_min) / Q) * Q + Q/2 + theta_min
#         Qtheta[idx[mask_domi]] = theta_norm  # Keep only the dominant sign of the Top-q entries
#         if b:
#             Qtheta = np.sign(theta_norm) * np.abs(Qtheta).astype(eval("np.float{:d}".format(int(b))))
#         else:
#             Qtheta = np.sign(theta_norm)

#         # Update the accumulated error
#         acc_error[i] += (flattened_theta[i] - Qtheta)
#         quantized_theta.append(Qtheta) 
            
#     return quantized_theta, acc_error #A list (device_i's) of the quantized theta_i's and the quantization error

def  comp_quant_encoding(flattened_theta_by_Devices, flattened_hat_theta_by_Devices, q_array, b_array, confidential = .995):
    # var_lists_by_Devices is of structure [[\theta_0^{W}, \theta_0^{b}],.....[\theta_K^{W}, \theta_K^{b}]] by devices
    d = flattened_theta_by_Devices[0].size
    # Q_model_difference = [] # A list (device_i's) of quantized theta_{i}'s that device i prepares to send to its neighbours
    for i in range(len(flattened_theta_by_Devices)):
        b = b_array[i]
        model_difference_i = flattened_theta_by_Devices[i] - flattened_hat_theta_by_Devices[i]
        # theta_min = np.quantile(EC_flattened_theta_ji, 1-confidential)
        # theta_max = np.quantile(EC_flattened_theta_ji, confidential)
        # EC_flattened_theta_ji = np.maximum(EC_flattened_theta_ji, theta_min)
        # EC_flattened_theta_ji = np.minimum(EC_flattened_theta_ji, theta_max)

        idx = np.argsort(np.abs(model_difference_i))[d-q_array[i]:]  # Index for the top-q entries
            
        # the quantized theta_i that adopts b-bit float-point encoding for each of the top-q entry
        Q_model_difference_i = np.zeros((d,))
        # The TopQ entries
        domi_theta = model_difference_i[idx] 
        if b:
            Q_model_difference_i[idx] = domi_theta.astype(eval("np.float{:d}".format(int(b))))
        else:
            Q_model_difference_i[idx] = np.sign(domi_theta)

        # Update flattened_hat_theta
        flattened_hat_theta_by_Devices[i] += Q_model_difference_i
        # Q_model_difference.append(Q_model_difference_i) 
            
    return flattened_hat_theta_by_Devices #A list (device_i's) of the estimated theta_i's that are readily used for consensus update


def sample_data( i, t, train_images, train_labels, device_size = 7500, batch_size = 1024 ):
    device_images = train_images[ i * device_size : (i + 1) * device_size-1, :, : ]
    device_labels = train_labels[ i * device_size : (i + 1) * device_size-1, : ]
    
    n_per_epoch = device_size // batch_size + 1
    t %= n_per_epoch
    
    batch_images = device_images[ t * batch_size : (t + 1) * batch_size-1 ]
    batch_labels = device_labels[ t * batch_size : (t + 1) * batch_size-1 ]
    
    return batch_images, batch_labels



def calc_grad( model, batch_images, batch_labels ):
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        loss = loss_fn( batch_labels, model( batch_images ) )

    var_list = model.trainable_weights
    grads = tape.gradient(loss, var_list)
    
    return loss, var_list, grads

def LMMSE_Rx(i, G, CH, A, alpha, Rsc, Rcc, N0 = 10 ** (-169/10) * 1e-3 ):

    _, d = Rsc.shape 

    hat_Rsc = spdiags( Rsc, 0, d, d )
    hat_Rcc = spdiags( Rcc, 0, d, d )

    # with Timer( "Compute A1, A2" ):
    A1 = A @ hat_Rsc @ A.T
    A2 = A @ hat_Rcc @ A.T

    q1 = sum(np.abs(CH[j][i]) ** 2 * alpha[j] for j in G[i])
    q2 = sum(CH[i][j1] * CH[j2][i] * np.sqrt(alpha[j1]) * np.sqrt(alpha[j2]) for j1, j2 in it.combinations(G[i], 2))
    q3 = sum(CH[i][j] * np.sqrt(alpha[j]) for j in G[i])

    # with Timer( "Compute inverse" ):
    temp = np.linalg.inv(A1 * q1 + A2 * q2 + N0 * np.eye(A.shape[0]))
    
    # with Timer( "Compute U" ):
    U = q3 * (A1 + A2 * (G.degree[i] - 1)) @ temp

    return U

def solve_num_per_class(num_lack_max, K):
    X = np.zeros((10,K)) #X is an num_class-by-num_devices matrix indicating if device k collects data sample belonging to class i (X[n,k]=1)
    num_lack = np.random.random_integers(0,num_lack_max,(K,))
    for k in range(K):
        arr = np.ones((10,))
        arr[:num_lack[k]] = 0
        np.random.shuffle(arr)
        X[:,k] = arr
        
    x = cvx.Variable(K,integer=True)
    constraints = [x>=200, x<=600]
    for n in range(10):
        constraints.append(X[n,:] @ cvx.reshape(x, (K,1))<= 6000)

    obj = cvx.Maximize( cvx.sum(cvx.multiply(np.ones((10,1)) @ cvx.reshape(x, (1,K)), X)) )
    prob = cvx.Problem( obj, constraints )
    prob.solve( verbose = True, solver = cvx.GLPK_MI )

    return X, x.value, prob.value

def seq_scheduling(G):
    # A sequential list (slot's) of star toplogy-based schedule in a form of dicts 
    star_schedule_list = [] 
    # key-value pair herein is node (n_b, n_c), where n_b is the #times for which a node transmits as a star center (BC), 
    # and n_c is the #times for which a node transmits as an outer node
    Tx_times = {node:[0, 0] for node in G.nodes()} 

    while G:
        _, from_node_to_color_id = TwoSectionH(G)
        color_degree = {c: sum( len(G[node]) for node, color in from_node_to_color_id.items() if color == c ) 
                             for c in from_node_to_color_id.values()}
        chosen_color = list(color_degree.values()).index(max(color_degree.values())) # find arg_max(degree(color_list))

        # A dict including (star center: associated nodes) pairs that transmits or recieves in parallel at the current slot
        star_schedule_dict = { node:G[node] for node, color in from_node_to_color_id.items() if color == chosen_color}
        # Append the scheule in the current slot to the sequential schedule list
        star_schedule_list.append(star_schedule_dict)
        # Update n_b for the star center
        for node in star_schedule_dict.keys():
            Tx_times[node][0] += 1
        # Update n_c for the neighbors of the star centers
        for neighbors in star_schedule_dict.values():
            for node in neighbors:
                Tx_times[node][1] += 1

        # Update the graph
        # Remove the scheduled Rxs, i.e., the star centers
        G.remove_nodes_from(star_schedule_dict.keys())
        # Remove any standalone nodes
        current_node_list = list(G.nodes())
        for node in current_node_list:
            if not(G[node]):
                G.remove_node(node)
   
    return star_schedule_list, Tx_times
    
       

class Timer( object ):

    def __init__(self, name):

        self.name = name

    def __enter__(self):

        self._start_time = time.process_time()

    def __exit__(self, exception_type, exception_value, traceback):

        end_time = time.process_time()
        elapsed = end_time - self._start_time
        print( "%s: %f seconds." % ( self.name, elapsed ) )


# def AMP( A, hat_y, N0, mu = 0, sigma_square = 1, shape = (7850,), sparse_rat = .1, nit = 15 ):
#     est0 = vp.estim.DiscreteEst(0,1,shape)
#     est1 = vp.estim.GaussEst(mu,sigma_square,shape)
#     est_list = [est0, est1]
#     pz = np.array([1-sparse_rat, sparse_rat])
#     est_in = vp.estim.MixEst(est_list, w = pz, name = 'Input')

#     Aop = vp.trans.MatrixLT(A,shape)
#     est_out = vp.estim.LinEst(Aop, hat_y, N0, map_est = False, name = 'Output')

#     msg_hdl = vp.estim.MsgHdlSimp(map_est = False, shape = shape)

#     solver = vp.solver.Vamp(est_in, est_out, msg_hdl, hist_list = ['zhat', 'zhatvar'], nit = nit)
#     solver.solve()

#     return solver.zhat
    


if __name__ == "__main__":

    K = 8
    E = [(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6), (5, 8), (4, 7), (4, 8), (7, 8)]
    W, s = solve_graph_weights( K, E )
    np.set_printoptions( suppress=True )
    print(W, "\n", s)
