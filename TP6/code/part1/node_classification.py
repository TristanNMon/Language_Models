"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
# your code here #
##################

colors = ['red' if label == 0 else 'blue' for label in y]

plt.figure(figsize=(10, 7))
    
# Use spring_layout for positioning (often looks good)
pos = nx.spring_layout(G, seed=42) 

# Draw the graph, passing the generated color list
nx.draw(
    G, 
    pos, 
    node_color=colors, 
    with_labels=True, 
    node_size=800, 
    edge_color='gray',
    font_size=10
    )

plt.title("Karate Network with Node Class Labels", fontsize=16)
plt.savefig("karate_network.png", dpi=300, bbox_inches='tight')
plt.show()

############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, num_walks = n_walks, walk_length = walk_length, n_dim = n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
# your code here #
##################

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc_lr = accuracy_score(y_pred, y_test)

# print(f"Classification accuracies of DeepWalk: {acc_lr:.5f} ")

############## Task 8
# Generates spectral embeddings

##################
# your code here #
##################

def spectral_clustering(G, k):
    # get adjacency matrix
    A = nx.to_scipy_sparse_array(G, format="csr")
    # compute Laplacian matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = diags(1/degrees)
    L_rw = eye(G.number_of_nodes()) - D_inv @ A

    # apply eigenvector decomposition 

    _, eigvecs = eigs(L_rw, k=k, which = 'SM')

    spectral_embeddings = eigvecs.real[:, 1:k] 

    return spectral_embeddings

spectral_embeddings = spectral_clustering(G, 2)

X_train_spectral = embeddings[idx_train,:]
X_test_spectral = embeddings[idx_test,:]

clf_spectral = LogisticRegression(random_state=0).fit(X_train_spectral, y_train)

y_pred_spectral = clf_spectral.predict(X_test_spectral)
acc_spectral = accuracy_score(y_pred_spectral, y_test)

print(f"Classification accuracy using DeepWalk (128D): {acc_lr:.5f}")
print("\nVs\n")
print(f"Classification accuracy using Spectral (2D): {acc_spectral:.5f}")