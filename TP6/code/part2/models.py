"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        # your code here #
        ##################
        H0 = self.fc1(x_in)
        Z0 = self.relu(torch.mm(adj,H0))
        Z0_dropout = self.dropout(Z0)

        H1 = self.fc2(Z0_dropout)
        Z1 = self.relu(torch.mm(adj,H1))

        x = self.fc3(Z1)

        return F.log_softmax(x, dim=1), Z1
