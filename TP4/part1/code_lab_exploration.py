"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

# Path to your txt file
file_path = "datasets/CA-HepTh.txt"

# Load the directed graph
G = nx.read_edgelist(
    file_path,
    comments="#",             # ignore comment lines
    delimiter="\t",           # tab-separated
    create_using=nx.Graph,    # undirected graph
    nodetype=int              # convert node IDs to integers
)

print(f"Total number of nodes: {G.number_of_nodes()}")
print(f"Total number of edges: {G.number_of_edges()}")

############## Task 2

print(f"Total number of connected components:{nx.number_connected_components(G)}")

largest_cc = max(nx.connected_components(G), key=len)
print(f"Total number of nodes of giant connected component: {G.subgraph(largest_cc).number_of_nodes()}")
print(f"Total number of edges of giant connected component: {G.subgraph(largest_cc).number_of_edges()}")

