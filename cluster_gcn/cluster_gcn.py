import argparse
import os, time
import random

import numpy as np
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import register_data_args

from modules import GCN
from sampler import ClusterIter
from utils import Logger, evaluate, load_data

import matplotlib.pyplot as plt


def main(args):
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load and preprocess dataset
    data = load_data(args)
    g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']
    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)

    # Normalize features
    if args.normalize:
        feats = g.ndata['feat']
        train_feats = feats[train_mask]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats.data.numpy())
        features = scaler.transform(feats.data.numpy())
        g.ndata['feat'] = torch.FloatTensor(features)

    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes

    g = g.long()

    # create GCN model
    cluster_iterator = ClusterIter(
        args.dataset, g, args.psize, args.batch_size,
        train_nid, use_pp=args.use_pp)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.int().to(args.gpu)

    print('labels shape:', g.ndata['label'].shape)
    print("features shape, ", g.ndata['feat'].shape)
    if args.model_type == 'sage':
        model = GCN(
            in_feats, args.n_hidden, n_classes, args.n_layers,
            F.relu, args.dropout, args.use_layernorm, False, False, 1, True)
    else:
        raise NotImplementedError(f'{args.model_type} is not a supported model type')

    if cuda:
        model.cuda()

    # use optimizer
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()

    total_time = 0.
    val_accs = []
    test_accs = []
    for epoch in range(args.n_epochs):
        print(f'Running epoch {epoch} / {args.n_epochs}', flush=True)
        start_time = time.time()
        for j, cluster in enumerate(cluster_iterator):
            # sync with upper level training graph
            if cuda:
                cluster = cluster.to(torch.cuda.current_device())
            model.train()
            # forward
            pred = model(cluster)
            batch_labels = cluster.ndata['label']
            batch_train_mask = cluster.ndata['train_mask']
            loss = loss_f(pred[batch_train_mask],
                          batch_labels[batch_train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        # evaluate
        # do NOT include evaluation within the timing metrics
        if args.eval_cpu:
            model.to('cpu')
            if args.use_f1:
                val_acc = evaluate(model, g.cpu(), labels.cpu(), val_mask.cpu(), 'f1')
                test_acc = evaluate(model, g.cpu(), labels.cpu(), test_mask.cpu(), 'f1')
            else:
                val_acc = evaluate(model, g.cpu(), labels.cpu(), val_mask.cpu())
                test_acc = evaluate(model, g.cpu(), labels.cpu(), test_mask.cpu())
            model.cuda()
        else:
            if args.use_f1:
                val_acc = evaluate(model, g, labels, val_mask, 'f1')
                test_acc = evaluate(model, g, labels, test_mask, 'f1')
            else:
                val_acc = evaluate(model, g, labels, val_mask)
                test_acc = evaluate(model, g, labels, test_mask)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        print(f'Val acc {val_acc}', flush=True)

    print(f'Training Time: {total_time:.4f}', flush=True)
    print(f'Last Val: {val_accs[-1]:.4f}', flush=True)
    print(f'Best Val: {max(val_accs):.4f}', flush=True)
    print(f'Last Test: {test_accs[-1]:.4f}', flush=True)
    print(f'Best Test: {max(test_accs):.4f}', flush=True)

    plt.plot(val_accs)
    title = args.fig_name
    plt.title(title)
    os.makedirs(args.fig_dir, exist_ok=True)
    plt.savefig(os.path.join(args.fig_dir, title + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=40,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size")
    parser.add_argument("--psize", type=int, default=1500,
                        help="partition number")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--rnd-seed", type=int, default=3,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--use-pp", action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--normalize", action='store_true',
                        help="whether to use normalized feature")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="Weight for L2 loss")
    parser.add_argument("--model-type", type=str, default='sage')
    parser.add_argument("--fig-dir", type=str, default='../report/example_pic/')
    parser.add_argument("--fig-name", type=str, default='name')
    parser.add_argument("--use-layernorm", action='store_true')
    parser.add_argument("--use-f1", action='store_true')
    parser.add_argument("--eval-cpu", action='store_true')
    args = parser.parse_args()

    print(args)

    main(args)
