import math

import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_attention(self, edges):
    # edge UDF for equation (2)
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = self.attn_fc(z2)
    return {'e' : F.leaky_relu(a)}

def reduce_func(self, nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h' : h}

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha*nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        return torch.mean(torch.stack(head_outs))
    
class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, num_heads):
        super().__init__()
        self.layers = [
                MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        ]
        for layer_idx in range(num_layers - 2):
            gat_layer = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads)
            self.layers.append(gat_layer)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layers.append(MultiHeadGATLayer(hidden_dim, out_dim, 1))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, g):
        h = g.ndata['feat']
        for layer in self.layers:
            h = layer(g, h)
            h = F.elu(h)
        return h

class GraphSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        super(GraphSAGELayer, self).__init__()
        # The input feature size gets doubled as we concatenated the original
        # features with the new features.
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.activation = activation
        self.use_pp = use_pp
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        if not self.use_pp or not self.training:
            norm = self.get_norm(g)
            g.ndata['h'] = h
            g.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))
            ah = g.ndata.pop('h')
            h = self.concat(h, ah, norm)

        if self.dropout:
            h = self.dropout(h)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.linear.weight.device)
        return norm

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_pp):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSAGELayer(in_feats, n_hidden, activation=activation,
                                        dropout=dropout, use_pp=use_pp, use_lynorm=True))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphSAGELayer(n_hidden, n_hidden, activation=activation, dropout=dropout,
                             use_pp=False, use_lynorm=True))
        # output layer
        self.layers.append(GraphSAGELayer(n_hidden, n_classes, activation=None,
                                        dropout=dropout, use_pp=False, use_lynorm=False))

    def forward(self, g):
        h = g.ndata['feat']
        for layer in self.layers:
            h = layer(g, h)
        return h

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
        self.linear = nn.Linear(2*in_feats, out_feats)
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
    def __init__(
            self, in_feats, n_hidden, n_classes, n_layers,
            activation, dropout, use_layernorm=True, split_input=False,
            split_output=False, num_subnet=1, use_aggregation=False):
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
