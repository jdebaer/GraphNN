import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
from gd_optim import GDOptim
from gcn_network import GCNNetwork


def cross_entropy_loss(preds, labels):
    # preds dim is (b_s, n_outputs of softmax)
    # labels dim is (b_s, n_of_classes which == n_ouputs of softmax)
    # print(preds[0]) -> [0.34272791 0.33088605 0.32638604]
    # print(labels[0]) -> [0. 0. 1.]
    # print([np.arange(preds.shape[0]), np.argmax(labels, axis=1)]) -> [array([ 0,  1,  2, ... , 32, 33]), array([2, 1, 1, ... , 0, 0])]
    result = -np.log(preds)[np.arange(preds.shape[0]), np.argmax(labels, axis=1)]
    # the [] is a coordinate matrix which basically says "get the log from that position" - this works because we're multiplying by either 0 or 1
    return result

g = nx.karate_club_graph()
communities = greedy_modularity_communities(g)
#print(communities)

colors = np.zeros(g.number_of_nodes())
#print(colors)

for idx, comm in enumerate(communities):
    colors[list(comm)] = idx
#print(colors)
n_classes = np.unique(colors).shape[0]

labels = np.eye(n_classes)[colors.astype(int)]

# The above is a clever way to create CEL-compatible labels:; 2-dim array where the 2nd dim is a one-hot encoding of the class (via index)
#[2. 1. 1. 1. 2. 2. 2. 1. 0. 1. 2. 2. 1. 1. 0. 0. 2. 1. 0. 2. 0. 1. 0. 0.  0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [[0. 0. 1.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  ...

# Normalized form of the adjacency matrix:
# A_hat = D_mod**-1/2 @ A_mod @ D_mod**-1/2
# A^ = D~**-1/2@A~@D~**-1/2

A = nx.to_numpy_array(g, weight=None)
A_mod = A + np.eye(g.number_of_nodes())
D_mod = np.zeros_like(A_mod)
np.fill_diagonal(D_mod,A_mod.sum(axis=0))
D_mod_invroot = inv(sqrtm(D_mod))
A_hat = D_mod_invroot @ A_mod @ D_mod_invroot

# Contruct input

X = np.eye(g.number_of_nodes())

train_nodes = np.array([0,1,8])
test_nodes = np.array([idx for idx in range(labels.shape[0]) if idx not in train_nodes]) # All the remaining indexes

opt = GDOptim(lr=2e-2, wd=2.5e-2)

gcn_network = GCNNetwork(
                            n_inputs = 34,
                            n_outputs = 3,
                            hidden_sizes = [16, 2],
                            seed = 100,
                            )

accuracies = []
train_losses = []
test_losses = []

for epoch in range(1):
    
    y_pred = gcn_network.forward(A_hat,X)
    opt(y_pred, labels, train_nodes)

    for layer in reversed(gcn_network.layers):
        layer.backward(opt, update=True)

    # Argmax gives us the position of the "1" in each 2nd dimension array of size 3, and tally up the matching hits
    epoch_accuracy = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
        [idx for idx in range(labels.shape[0]) if idx not in train_nodes]
    ]

    accuracies.append(epoch_accuracy)

    loss = cross_entropy_loss(y_pred, labels)    














