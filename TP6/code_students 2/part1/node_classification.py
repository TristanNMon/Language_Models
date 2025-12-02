"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

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

colors = ['lightgreen' if label == 0 else 'skyblue' for label in y]

# Draw the network
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G) 

nx.draw(G, pos,
        with_labels=True,
        node_color=colors,
        font_weight='bold',
        node_size=1200)

plt.title("Karate Network Groups", fontsize=14)
plt.axis('off')
plt.show()

############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, num_walks=n_walks, walk_length=walk_length, n_dim=n_dim) # your code here

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
clf = LogisticRegression(max_iter=10, random_state=88)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("DeepWalk embeddings accuracy:", accuracy)

############## Task 8
# Generates spectral embeddings

##################
# your code here #
##################

A = nx.adjacency_matrix(G)

degrees = np.array(A.sum(axis=1)).flatten()
D_inv = diags(1 / degrees)
L_rw = eye(n) - D_inv @ A

_, eigvecs = eigs(L_rw, k=2, which='SM') # smallest magnitude
spectral_embeddings = np.real(eigvecs)

X_train_spectral = spectral_embeddings[idx_train,:]
X_test_spectral = spectral_embeddings[idx_test,:]

clf_spectral = LogisticRegression(max_iter=10, random_state=88)
clf.fit(X_train_spectral, y_train)
y_pred_spectral = clf.predict(X_test_spectral)
accuracy_spectral = accuracy_score(y_test, y_pred_spectral)
print("Spectral embeddings accuracy:", accuracy_spectral)