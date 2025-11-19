"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    ##################
    # your code here #
    ##################

    # get adjacency matrix
    A = nx.to_scipy_sparse_array(G, format="csr")
    # compute Laplacian matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = diags(1/degrees)
    L_rw = eye(G.number_of_nodes()) - D_inv @ A

    # apply eigenvector decomposition 

    _, eigvecs = eigs(L_rw, k=k, which = 'SM')

    # Apply k-means
    U = eigvecs.real  # ensure real values
    kmeans = KMeans(n_clusters=k,random_state=42)
    labels = kmeans.fit_predict(U)

    # Assign labels
    node_list = list(G.nodes())
    clustering = {node : labels[i] for i, node in enumerate(node_list)}
    
    return clustering


############## Task 4

##################
# your code here #
##################

# Path to your txt file
file_path = "../datasets/CA-HepTh.txt"

# Load the directed graph
G = nx.read_edgelist(
    file_path,
    comments="#",             # ignore comment lines
    delimiter="\t",           # tab-separated
    create_using=nx.Graph,    # undirected graph
    nodetype=int              # convert node IDs to integers
)

largest_cc = max(nx.connected_components(G), key=len)

largest_cc_clustering = spectral_clustering(G.subgraph(largest_cc),k=50)


############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    
    m = G.number_of_edges()

    if m == 0:
        return 0

    modularity = 0.0
    deg = dict(G.degree())
    cluster_labels = set(clustering.values())
    
    for c in cluster_labels:
        # nodes in this community
        c_nodes = [node for node, label in clustering.items() if label == c]
        g_c = G.subgraph(c_nodes)

        # lc = internal edges
        l_c = g_c.number_of_edges()

        # dc = sum of degrees of nodes in this community
        d_c = sum(deg[u] for u in c_nodes)

        # modularity contribution
        modularity += (l_c / m) - (d_c / (2*m))**2

    return modularity



############## Task 6

##################
# your code here #
##################
largest_cc_clustering
random_clustering = {node : randint(0,50) for node in largest_cc_clustering.keys()} 

assert largest_cc_clustering.keys()==random_clustering.keys()

modularity_cc = modularity(G.subgraph(largest_cc),largest_cc_clustering)
modularity_cc_random = modularity(G.subgraph(largest_cc),random_clustering)

print(f"Modularity of communities in giant connected component: {modularity_cc:.5f} ")
print("\nVs\n")
print(f"Modularity of random : {modularity_cc_random:.5f}")






