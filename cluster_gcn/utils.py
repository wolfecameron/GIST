import os
from functools import namedtuple
import dgl
import numpy as np
import torch
from dgl.data import PPIDataset
from dgl.data import load_data as _load_data
from sklearn.metrics import f1_score
from AmazonDataset import AmazonDataset


class Logger(object):
    '''A custom logger to log stdout to a logging file.'''

    def __init__(self, path):
        """Initialize the logger.

        Paramters
        ---------
        path : str
            The file path to be stored in.
        """
        self.path = path

    def write(self, s):
        with open(self.path, 'a') as f:
            f.write(str(s))
        print(s)
        return


def arg_list(labels):
    hist, indexes, inverse, counts = np.unique(
        labels, return_index=True, return_counts=True, return_inverse=True)
    li = []
    for h in hist:
        li.append(np.argwhere(inverse == h))
    return li


def save_log_dir(args):
    log_dir = './log/{}/{}'.format(args.dataset, args.note)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def calc_f1(y_true, y_pred, multitask):
    if multitask:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    total = y_pred.shape[0]
    if total == 0:
        return -1
    else:
        return f1_score(y_true, y_pred, average="micro")


def calc_acc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    num_correct = np.sum(y_pred == y_true)
    total = y_pred.shape[0]
    if total == 0:
        return -1
    else:
        return num_correct / total


def evaluate(model, g, labels, mask, method='acc'):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        assert method in ['acc', 'f1'], 'invalid method'
        if method == 'acc':
            return calc_acc(labels.cpu().numpy(), logits.cpu().numpy())
        else:
            return calc_f1(labels.cpu().numpy(), logits.cpu().numpy(), False)


def load_data(args):
    '''Wraps the dgl's load_data utility to handle ppi special case'''
    DataType = namedtuple('Dataset', ['num_classes', 'g'])
    if args.dataset == 'amazon2m':
        amazon_data = AmazonDataset()
        DataType = namedtuple('Dataset', ['num_classes', 'g'])
        data = DataType(g=amazon_data[0], num_classes=amazon_data.num_classes)
        return data
    elif args.dataset == 'ppi':
        train_dataset = PPIDataset('train')
        train_graph = dgl.batch([train_dataset[i] for i in range(len(train_dataset))], edge_attrs=None, node_attrs=None)
        val_dataset = PPIDataset('valid')
        val_graph = dgl.batch([val_dataset[i] for i in range(len(val_dataset))], edge_attrs=None, node_attrs=None)
        test_dataset = PPIDataset('test')
        test_graph = dgl.batch([test_dataset[i] for i in range(len(test_dataset))], edge_attrs=None, node_attrs=None)
        G = dgl.batch(
            [train_graph, val_graph, test_graph], edge_attrs=None, node_attrs=None)

        train_nodes_num = train_graph.number_of_nodes()
        test_nodes_num = test_graph.number_of_nodes()
        val_nodes_num = val_graph.number_of_nodes()
        nodes_num = G.number_of_nodes()
        assert (nodes_num == (train_nodes_num + test_nodes_num + val_nodes_num))
        # construct mask
        mask = np.zeros((nodes_num,), dtype=bool)
        train_mask = mask.copy()
        train_mask[:train_nodes_num] = True
        val_mask = mask.copy()
        val_mask[train_nodes_num:-test_nodes_num] = True
        test_mask = mask.copy()
        test_mask[-test_nodes_num:] = True

        G.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
        G.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
        G.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

        data = DataType(g=G, num_classes=train_dataset.num_labels)
        return data
    else:
        dataset = _load_data(args)
        data = DataType(g=dataset[0], num_classes=dataset.num_classes)
        return data
