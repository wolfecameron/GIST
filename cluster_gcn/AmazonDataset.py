import os, json, torch, numbers
import numpy as np
import sklearn.metrics
import scipy.sparse as sp
from networkx.readwrite import json_graph
import sklearn.preprocessing
import tensorflow as tf
import time
from functools import namedtuple

np.random.seed(0)

from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.convert import from_scipy


class AmazonDataset(DGLDataset):
    def __init__(self, save_dir='./amazon2m_data/', force_reload=False, verbose=False):
        super(AmazonDataset, self).__init__(name='amazon2M',
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def process(self):
        print('Start loading Amazon Dataset', flush=True)

        start_time = time.time()
        feats = np.load(tf.io.gfile.GFile('{}/{}-feats.npy'.format(self.raw_path, self.name), 'rb')).astype(np.float32)
        print('Load feat Duration: {:.2f}s'.format(time.time() - start_time), flush=True)

        start_time = time.time()
        G = json_graph.node_link_graph(json.load(tf.io.gfile.GFile('{}/{}-G.json'.format(self.raw_path, self.name))))
        print('Load graph Duration: {:.2f}s'.format(time.time() - start_time), flush=True)

        start_time = time.time()
        id_map = json.load(tf.io.gfile.GFile('{}/{}-id_map.json'.format(self.raw_path, self.name)))
        is_digit = list(id_map.keys())[0].isdigit()
        id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}

        class_map = json.load(tf.io.gfile.GFile('{}/{}-class_map.json'.format(self.raw_path, self.name)))
        is_instance = isinstance(list(class_map.values())[0], list)
        class_map = {(int(k) if is_digit else k): (v if is_instance else int(v)) for k, v in class_map.items()}
        print('Load map Duration: {:.2f}s'.format(time.time() - start_time), flush=True)

        # Generate edge list
        start_time = time.time()
        edges = []
        for edge in G.edges():
            if edge[0] in id_map and edge[1] in id_map:
                edges.append((id_map[edge[0]], id_map[edge[1]]))

        # Total Number of Nodes in the Graph
        _nodes = len(id_map)

        # Seperate Train, Val, and Test nodes
        val_nodes = np.array([id_map[n] for n in G.nodes() if G.nodes[n]['val']], dtype=np.int32)
        test_nodes = np.array([id_map[n] for n in G.nodes() if G.nodes[n]['test']], dtype=np.int32)
        is_train = np.ones((_nodes), dtype=np.bool)
        is_train[test_nodes] = False
        is_train[val_nodes] = False
        train_nodes = np.array([n for n in range(_nodes) if is_train[n]], dtype=np.int32)

        # Train Edges
        train_edges = [(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]]
        train_edges = np.array(train_edges, dtype=np.int32)

        # All Edges in the Graph
        _edges = np.array(edges, dtype=np.int32)
        print('Process edge Duration: {:.2f}s'.format(time.time() - start_time), flush=True)

        # Generate Labels
        start_time = time.time()
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            _labels = np.zeros((_nodes, num_classes), dtype=np.float32)
            for k in class_map.keys():
                _labels[id_map[k], :] = np.array(class_map[k])
        else:
            num_classes = len(set(class_map.values()))
            _labels = np.zeros((_nodes, num_classes), dtype=np.float32)
            for k in class_map.keys():
                _labels[id_map[k], class_map[k]] = 1

        _labels = np.argmax(_labels, 1)

        train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])

        train_feats = feats[train_ids]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats)
        _feats = scaler.transform(feats)

        def _construct_adj(e, shape):
            adj = sp.csr_matrix((np.ones((e.shape[0]), dtype=np.float32), (e[:, 0], e[:, 1])), shape=shape)
            adj += adj.transpose()
            return adj

        # train_adj = _construct_adj(train_edges, (len(train_nodes), len(train_nodes)))
        _adj = _construct_adj(_edges, (_nodes, _nodes))

        # train_feats = _feats[train_nodes]

        # Generate Masks for Validtion & Testing Data
        train_mask = sample_mask(train_nodes, _labels.shape[0])
        val_mask = sample_mask(val_nodes, _labels.shape[0])
        test_mask = sample_mask(test_nodes, _labels.shape[0])
        print('Remained Duration: {:.2f}s'.format(time.time() - start_time), flush=True)

        start_time = time.time()
        self._graph = from_scipy(_adj)
        self._graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
        self._graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
        self._graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
        self._graph.ndata['feat'] = torch.tensor(_feats, dtype=torch.float32)
        self._graph.ndata['label'] = torch.tensor(_labels, dtype=torch.int64)
        self._print_info()
        print('Convert Graph Duration: {:.2f}s'.format(time.time() - start_time), flush=True)

    def __getitem__(self, idx):
        assert idx == 0, "Amazon Dataset only has one graph"
        return self._graph

    def __len__(self):
        return 1

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self._graph.ndata['train_mask'] = generate_mask_tensor(self._graph.ndata['train_mask'].numpy())
        self._graph.ndata['val_mask'] = generate_mask_tensor(self._graph.ndata['val_mask'].numpy())
        self._graph.ndata['test_mask'] = generate_mask_tensor(self._graph.ndata['test_mask'].numpy())
        self._print_info()

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        if os.path.exists(graph_path):
            return True
        return False

    def _print_info(self):
        if self.verbose:
            print('Finished data loading.')
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumFeats: {}'.format(self._graph.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))
            print('  NumTrainingSamples: {}'.format(nonzero_1d(self._graph.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(nonzero_1d(self._graph.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(nonzero_1d(self._graph.ndata['test_mask']).shape[0]))

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 47

    @property
    def num_labels(self):
        return self.num_classes

    @property
    def graph(self):
        return self._graph

    @property
    def train_mask(self):
        return asnumpy(self._graph.ndata['train_mask'])

    @property
    def val_mask(self):
        return asnumpy(self._graph.ndata['val_mask'])

    @property
    def test_mask(self):
        return asnumpy(self._graph.ndata['test_mask'])

    @property
    def features(self):
        return self._graph.ndata['feat']

    @property
    def labels(self):
        return self._graph.ndata['label']


def generate_mask_tensor(mask):
    assert isinstance(mask, np.ndarray), "input for generate_mask_tensor" \
                                         "should be an numpy ndarray"
    return tensor(mask, dtype=torch.bool)


def sample_mask(idx, mat):
    mask = np.zeros(mat)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def nonzero_1d(input):
    x = torch.nonzero(input, as_tuple=False).squeeze()
    return x if x.dim() == 1 else x.view(-1)


def asnumpy(input):
    if isinstance(input, torch.sparse.FloatTensor):
        return input.to_dense().cpu().detach().numpy()
    else:
        return input.cpu().detach().numpy()


def tensor(data, dtype=None):
    if isinstance(data, numbers.Number):
        data = [data]
    if isinstance(data, torch.Tensor):
        return torch.as_tensor(data, dtype=dtype, device=data.device)
    else:
        return torch.as_tensor(data, dtype=dtype)


if __name__ == '__main__':
    # amazon_data = load_data('amazon2M')
    amazon_data = AmazonDataset()
    DataType = namedtuple('Dataset', ['num_classes', 'g'])
    data = DataType(g=amazon_data[0], num_classes=amazon_data.num_classes)
