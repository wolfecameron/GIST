import os

os.environ["DGL_REPO"] = "http://data.dgl.ai/"
import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gcn import GCN


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # convert boolean type for args
    assert args.self_loop in ['True', 'False'], ["Only True or False for self_loop, get ",
                                                 args.self_loop]
    assert args.use_layernorm in ['True', 'False'], ["Only True or False for use_layernorm, get ",
                                                     args.use_layernorm]
    self_loop = (args.self_loop == 'True')
    use_layernorm = (args.use_layernorm == 'True')
    global t0
    if args.dataset in {'cora', 'citeseer', 'pubmed'}:
        data = load_data(args)
    else:
        raise NotImplementedError(f'{args.dataset} is not a valid dataset')
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # graph preprocess and calculate normalization factor
    g = data.graph
    # add self loop
    if self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    g = g.to(device)
    n_edges = g.number_of_edges()
    
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.to(device)
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(
            g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu,
            args.dropout, use_layernorm)
    model = model.to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    record = []
    dur = []
    for epoch in range(args.n_epochs):
        if args.lr_scheduler:
            if epoch == int(0.5*args.n_epochs):
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] / 10
            elif epoch == int(0.75*args.n_epochs):
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] / 10
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc_val = evaluate(model, features, labels, val_mask)
        acc_test = evaluate(model, features, labels, test_mask)
        record.append([acc_val, acc_test])

    all_test_acc = [v[1] for v in record]
    all_val_acc = [v[0] for v in record]
    acc = evaluate(model, features, labels, test_mask)
    print(f"Final Test Accuracy: {acc:.4f}")
    print(f"Best Val Accuracy: {max(all_val_acc):.4f}")
    print(f"Best Test Accuracy: {max(all_test_acc):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=.001,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self_loop", type=str, default='True',
                        help="graph self-loop (default=True)")
    parser.add_argument("--lr_scheduler", action='store_true', default=False,
                        help="Use LR scheduler")
    parser.add_argument("--use_layernorm", type=str, default='True',
                        help="Whether use layernorm (default=False)")
    args = parser.parse_args()
    main(args)
