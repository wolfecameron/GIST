import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_layernorm=True,
                 split_input=False,
                 split_output=False,
                 num_subnet=1,
                 ):
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.use_layernorm = use_layernorm
        self.split_input = split_input
        self.split_output = split_output
    
        # construct input layer
        if split_input:
            if n_layers <= 1 and not split_output:
                self.layers.append(GraphConv(int(in_feats // num_subnet), n_hidden, activation=activation))
            else:
                self.layers.append(
                        GraphConv(
                            int(in_feats // num_subnet), int(n_hidden // num_subnet),
                            activation=activation))
        else:
            if n_layers <= 1 and not split_output:
                self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
            else:
                self.layers.append(GraphConv(in_feats, int(n_hidden // num_subnet), activation=activation))

                
        # construct hidden layers
        for i in range(n_layers - 1):
            if i == n_layers - 2 and not split_output:
                self.layers.append(GraphConv(int(n_hidden // num_subnet), n_hidden, activation=activation))
            else:
                self.layers.append(
                        GraphConv(int(n_hidden // num_subnet), int(n_hidden // num_subnet), activation=activation))
               
 
        # construct output layers
        if split_output:
            self.layers.append(GraphConv(int(n_hidden // num_subnet), n_classes))
        else:
            self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i < len(self.layers) - 1 and self.use_layernorm:
                h = F.layer_norm(h, h.shape)
        return h
