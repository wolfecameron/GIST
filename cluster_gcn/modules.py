import math

import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISTSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dropout,
                 use_lynorm,
                 activation=None):
        super().__init__()
        # The input feature size gets doubled as we concatenated the original
        # features with the new features.
        self.linear = nn.Linear(2 * in_feats, out_feats)
        self.activation = activation
        self.init_layer()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=False)
        else:
            self.lynorm = lambda x: x

    def init_layer(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()

        # run graph SAGE style aggregation
        norm = self.get_norm(g)
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
                     fn.sum(msg='m', out='h'))
        ah = g.ndata.pop('h') * norm
        h = torch.cat((h, ah), dim=1)

        # dropout and layernorm
        if self.dropout:
            h = self.dropout(h)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.linear.weight.device)
        return norm


class GCN(nn.Module):
    def __init__(self,
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
                 use_aggregation=False,
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_layernorm = use_layernorm
        self.split_input = split_input
        self.split_output = split_output
        if use_aggregation:
            layer_type = ISTSAGELayer
        else:
            raise NotImplementedError('You must use graph sage')

            # construct input layer
        if split_input:
            if n_layers <= 1 and not split_output:
                self.layers.append(
                    layer_type(
                        int(in_feats // num_subnet), n_hidden, dropout,
                        use_layernorm, activation=activation))
            else:
                self.layers.append(
                    layer_type(
                        int(in_feats // num_subnet), int(n_hidden // num_subnet),
                        dropout, use_layernorm, activation=activation))
        else:
            if n_layers <= 1 and not split_output:
                self.layers.append(
                    layer_type(
                        in_feats, n_hidden, dropout, use_layernorm,
                        activation=activation))
            else:
                self.layers.append(
                    layer_type(
                        in_feats, int(n_hidden // num_subnet),
                        dropout, use_layernorm, activation=activation))

        # construct hidden layers
        for i in range(n_layers - 1):
            if i == n_layers - 2 and not split_output:
                self.layers.append(
                    layer_type(
                        int(n_hidden // num_subnet), n_hidden,
                        dropout, use_layernorm, activation=activation))
            else:
                self.layers.append(
                    layer_type(
                        int(n_hidden // num_subnet), int(n_hidden // num_subnet),
                        dropout, use_layernorm, activation=activation))

        # construct output layers
        if split_output:
            self.layers.append(
                layer_type(
                    int(n_hidden // num_subnet), n_classes,
                    dropout, False, activation=None))
        else:
            self.layers.append(
                layer_type(
                    n_hidden, n_classes, dropout,
                    False, activation=None))

    def forward(self, g):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        return h


class BaselineGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_layernorm=True,
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_layernorm = use_layernorm

        # construct input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))

        # construct hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))

        # construct output layers
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            if i < len(self.layers) - 1 and self.use_layernorm:
                h = F.layer_norm(h, h.shape)
        return h
