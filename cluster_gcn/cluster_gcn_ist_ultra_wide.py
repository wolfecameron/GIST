import argparse
import os
import time
import random

os.environ["DGL_REPO"] = "http://data.dgl.ai/"
import sklearn.preprocessing
from dgl.data import register_data_args
import torch.distributed as dist
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt

from modules import GCN
from sampler import ClusterIter
from utils import Logger, evaluate, load_data


def broadcast_weight(para, rank_list=None, source=0):
    if rank_list is None:
        group = dist.group.WORLD
    else:
        group = dist.new_group(rank_list)
    dist.broadcast(para, src=source, group=group, async_op=False)
    if rank_list is not None:
        dist.destroy_process_group(group)


def broadcast_module_itr(args, module: torch.nn.Module, source=0):
    group = dist.new_group(list(range(args.num_subnet)))
    for para in module.parameters():
        dist.broadcast(para.data, src=source, group=group, async_op=False)
    dist.destroy_process_group(group)


def all_reduce_weights(args, para):
    group = dist.group.WORLD
    dist.all_reduce(para, op=dist.ReduceOp.SUM, group=group)
    para = para.div_(args.num_subnet)


def all_reduce_module(args, module: torch.nn.Module):
    group = dist.group.WORLD
    for para in module.parameters():
        dist.all_reduce(para.data, op=dist.ReduceOp.SUM, group=group)
        para.data = para.data.div_(args.num_subnet)


def create_partition(num_subnet, size):
    possible_indices = [x for x in range(size)]
    random.shuffle(possible_indices)
    feats_idx_list = [[] for x in range(num_subnet)]
    for i in range(size):
        next_idx = possible_indices[i]
        subnet_idx = i % num_subnet
        feats_idx_list[subnet_idx].append(next_idx)
    sage_feats_idx_list = []
    for idx in feats_idx_list:
        idx = torch.LongTensor(idx)
        lower_idx = idx + size
        full_idx = torch.cat((idx, lower_idx))
        sage_feats_idx_list.append((idx, full_idx))
    return sage_feats_idx_list


class DistributedGNNWrapper(torch.nn.Module):
    # wrapper class to handle full GNN and subnetworks that are created

    def __init__(self, args, g, in_feats, n_classes, device):
        super().__init__()
        self.args = args
        self.g = g
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.device = device
        if args.rank == 0:
            # construct a full model to keep on the parameter server
            # this is a graphSAGE-style network
            self.base_model = GCN(
                in_feats, args.n_hidden, n_classes, args.n_layers, F.relu,
                args.dropout, args.use_layernorm, False, False, 1, True)
            # self.base_model = self.base_model.to(self.device)
        else:
            self.base_model = None
        self.sub_model = GCN(
            in_feats, args.n_hidden, n_classes, args.n_layers, F.relu,
            args.dropout, args.use_layernorm, False, True, args.num_subnet, True)
        self.sub_model = self.sub_model.to(self.device)
        self.current_partition = None

    def sample_partitions(self):
        partition = []
        for i in range(self.args.n_layers):
            partition.append(
                create_partition(self.args.num_subnet, self.args.n_hidden))
        return partition

    def sync_model(self):
        with torch.no_grad():
            # run all reduce on the shared weights
            all_reduce_weights(self.args, self.sub_model.layers[-1].linear.bias.data)

            # copy weights into the full model on the main node
            if self.args.rank == 0:
                for layer_idx in range(len(self.current_partition) + 1):
                    if layer_idx == 0:
                        # first layer
                        idx_tens, full_idx_tens = self.current_partition[layer_idx][0]
                        self.base_model.layers[0].linear.weight.data[idx_tens, :] = (
                            self.sub_model.layers[0].linear.weight.data.cpu())
                        self.base_model.layers[0].linear.bias.data[idx_tens] = (
                            self.sub_model.layers[0].linear.bias.data.cpu())
                    elif layer_idx == len(self.current_partition):
                        # last layer
                        idx_tens, full_idx_tens = self.current_partition[layer_idx - 1][0]
                        self.base_model.layers[layer_idx].linear.weight.data[:, full_idx_tens] = (
                            self.sub_model.layers[layer_idx].linear.weight.data.cpu())

                        # copy all of the bias weights -- shared between all networks
                        self.base_model.layers[layer_idx].linear.bias.data = (
                            self.sub_model.layers[layer_idx].linear.bias.data.cpu())
                    else:
                        # general case, middle layers
                        prev_idx, full_prev_idx = self.current_partition[layer_idx - 1][0]
                        next_idx, full_next_idx = self.current_partition[layer_idx][0]
                        correct_rows = self.base_model.layers[layer_idx].linear.weight.data[:, full_prev_idx]
                        correct_rows[next_idx, :] = (
                            self.sub_model.layers[layer_idx].linear.weight.data.cpu())
                        self.base_model.layers[layer_idx].linear.weight.data[:, full_prev_idx] = correct_rows
                        self.base_model.layers[layer_idx].linear.bias.data[next_idx] = (
                            self.sub_model.layers[layer_idx].linear.bias.data.cpu())

            # only works properly if there is no overlap within the parameters
            for layer_idx in range(len(self.current_partition) + 1):
                if layer_idx == 0:
                    # first layer
                    for site_i in range(1, self.args.num_subnet):
                        idx_tens, full_idx_tens = self.current_partition[layer_idx][site_i]
                        if self.args.rank == 0:
                            # communicate weight
                            corr_weight = self.base_model.layers[0].linear.weight.data[idx_tens, :]
                            corr_weight = corr_weight.to(self.device)
                            broadcast_weight(
                                corr_weight, rank_list=[0, site_i], source=site_i)
                            corr_weight = corr_weight.to('cpu')
                            self.base_model.layers[0].linear.weight.data[idx_tens, :] = corr_weight

                            # communicate bias
                            corr_bias = self.base_model.layers[0].linear.bias.data[idx_tens]
                            corr_bias = corr_bias.to(self.device)
                            broadcast_weight(
                                corr_bias, rank_list=[0, site_i], source=site_i)
                            corr_bias = corr_bias.to('cpu')
                            self.base_model.layers[0].linear.bias.data[idx_tens] = corr_bias
                        else:
                            broadcast_weight(
                                self.sub_model.layers[0].linear.weight.data, rank_list=[0, site_i],
                                source=site_i)
                            broadcast_weight(
                                self.sub_model.layers[0].linear.bias.data, rank_list=[0, site_i],
                                source=site_i)


                elif layer_idx == len(self.current_partition):
                    # last layer
                    for site_i in range(1, self.args.num_subnet):
                        idx_tens, full_idx_tens = self.current_partition[layer_idx - 1][site_i]
                        if self.args.rank == 0:
                            # do NOT need to copy in bias, it is all reduced
                            corr_row = self.base_model.layers[layer_idx].linear.weight.data[:, full_idx_tens]
                            corr_row = corr_row.to(self.device)
                            broadcast_weight(
                                corr_row, rank_list=[0, site_i], source=site_i)
                            corr_row = corr_row.to('cpu')
                            self.base_model.layers[layer_idx].linear.weight.data[:, full_idx_tens] = corr_row
                        else:
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.weight.data,
                                rank_list=[0, site_i], source=site_i)
                else:
                    # general case, middle layers
                    for site_i in range(1, self.args.num_subnet):
                        prev_tens, full_prev_tens = self.current_partition[layer_idx - 1][site_i]
                        next_tens, full_next_tens = self.current_partition[layer_idx][site_i]
                        if self.args.rank == 0:
                            # communicate the weights
                            crows = self.base_model.layers[layer_idx].linear.weight.data[:, full_prev_tens]
                            full_split = crows[next_tens, :]
                            full_split = full_split.to(self.device)
                            broadcast_weight(
                                full_split, rank_list=[0, site_i], source=site_i)
                            full_split = full_split.to('cpu')
                            crows[next_tens, :] = full_split
                            self.base_model.layers[layer_idx].linear.weight.data[:, full_prev_tens] = crows

                            # communicate the bias
                            cbias = self.base_model.layers[layer_idx].linear.bias.data[next_tens]
                            cbias = cbias.to(self.device)
                            broadcast_weight(
                                cbias, rank_list=[0, site_i], source=site_i)
                            cbias = cbias.to('cpu')
                            self.base_model.layers[layer_idx].linear.bias.data[next_tens] = cbias
                        else:
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.weight.data, rank_list=[0, site_i],
                                source=site_i)
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.bias.data, rank_list=[0, site_i],
                                source=site_i)

    def ini_sync_dispatch_model(self):
        # perform the partition
        all_indices = self.sample_partitions()

        # set weights within central node and broadcast them
        with torch.no_grad():
            # copy in weights for node 0
            if self.args.rank == 0:
                for layer_idx in range(len(all_indices) + 1):
                    if layer_idx == 0:
                        idx_tens, _ = all_indices[0][0]
                        self.sub_model.layers[0].linear.weight.data = (
                            self.base_model.layers[0].linear.weight.data[idx_tens, :]
                                .to(self.device))
                        self.sub_model.layers[0].linear.bias.data = (
                            self.base_model.layers[0].linear.bias.data[idx_tens]
                                .to(self.device))
                    elif layer_idx == len(all_indices):
                        idx_tens, full_idx_tens = all_indices[-1][0]
                        self.sub_model.layers[layer_idx].linear.weight.data = (
                            self.base_model.layers[layer_idx].linear.weight.data[:, full_idx_tens]
                                .to(self.device))

                        # full output bias is copied for the final layer
                        self.sub_model.layers[layer_idx].linear.bias.data = (
                            self.base_model.layers[layer_idx].linear.bias.data
                                .to(self.device))
                    else:
                        prev_idx, full_prev_idx = all_indices[layer_idx - 1][0]
                        next_idx, full_next_idx = all_indices[layer_idx][0]
                        correct_cols = self.base_model.layers[layer_idx].linear.weight.data[:, full_prev_idx]
                        self.sub_model.layers[layer_idx].linear.weight.data = (
                            correct_cols[next_idx, :].to(self.device))
                        self.sub_model.layers[layer_idx].linear.bias.data = (
                            self.base_model.layers[layer_idx].linear.bias.data[next_idx]
                                .to(self.device))

            # broadcast all of the shared weights
            broadcast_weight(self.sub_model.layers[-1].linear.bias, None, source=0)

            for layer_idx in range(len(all_indices) + 1):
                if layer_idx == 0:
                    for site_i in range(1, self.args.num_subnet):
                        idx_tens, full_idx_tens = all_indices[0][site_i]
                        if self.args.rank == 0:
                            # broadcast the weight
                            bcast_weight = self.base_model.layers[0].linear.weight.data[idx_tens, :]
                            bcast_weight = bcast_weight.to(self.device)
                            broadcast_weight(
                                bcast_weight, rank_list=[0, site_i], source=0)
                            bcast_weight = bcast_weight.to('cpu')

                            # broadcastt the bias
                            bcast_bias = self.base_model.layers[0].linear.bias.data[idx_tens]
                            bcast_bias = bcast_bias.to(self.device)
                            broadcast_weight(
                                bcast_bias, rank_list=[0, site_i], source=0)
                            bcast_bias = bcast_bias.to('cpu')
                        else:
                            broadcast_weight(
                                self.sub_model.layers[0].linear.weight.data,
                                rank_list=[0, site_i], source=0)
                            broadcast_weight(
                                self.sub_model.layers[0].linear.bias.data,
                                rank_list=[0, site_i], source=0)

                elif layer_idx == len(all_indices):
                    for site_i in range(1, self.args.num_subnet):
                        idx_tens, full_idx_tens = all_indices[-1][site_i]
                        if self.args.rank == 0:
                            # do NOT need to broadcast bias, it is already done above
                            bcast_weight = self.base_model.layers[layer_idx].linear.weight.data[:, full_idx_tens]
                            bcast_weight = bcast_weight.to(self.device)
                            broadcast_weight(
                                bcast_weight, rank_list=[0, site_i], source=0)
                            bcast_weight = bcast_weight.to('cpu')
                        else:
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.weight.data,
                                rank_list=[0, site_i], source=0)
                else:
                    for site_i in range(1, self.args.num_subnet):
                        prev_tens, full_prev_tens = all_indices[layer_idx - 1][site_i]
                        next_tens, full_next_tens = all_indices[layer_idx][site_i]
                        if self.args.rank == 0:
                            # broadcast the weights
                            correct_rows = (
                                self.base_model.layers[layer_idx].linear.weight.data[:, full_prev_tens])
                            correct_rows = correct_rows[next_tens, :]
                            correct_rows = correct_rows.to(self.device)
                            broadcast_weight(
                                correct_rows, rank_list=[0, site_i],
                                source=0)
                            correct_rows = correct_rows.to('cpu')

                            bcast_bias = self.base_model.layers[layer_idx].linear.bias.data[next_tens]
                            bcast_bias = bcast_bias.to(self.device)
                            broadcast_weight(
                                bcast_bias, rank_list=[0, site_i], source=0)
                            bcast_bias = bcast_bias.to('cpu')
                        else:
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.weight.data, rank_list=[0, site_i],
                                source=0)
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.bias.data, rank_list=[0, site_i],
                                source=0)

        self.current_partition = all_indices

    def dispatch_model(self):
        # get the subnetwork indices
        all_indices = self.sample_partitions()

        with torch.no_grad():
            # copy in weights for node 0
            if self.args.rank == 0:
                for layer_idx in range(len(all_indices) + 1):
                    if layer_idx == 0:
                        idx_tens, _ = all_indices[0][0]
                        self.sub_model.layers[0].linear.weight.data = (
                            self.base_model.layers[0].linear.weight.data[idx_tens, :]
                                .to(self.device))
                        self.sub_model.layers[0].linear.bias.data = (
                            self.base_model.layers[0].linear.bias.data[idx_tens]
                                .to(self.device))
                    elif layer_idx == len(all_indices):
                        idx_tens, full_idx_tens = all_indices[-1][0]
                        self.sub_model.layers[layer_idx].linear.weight.data = (
                            self.base_model.layers[layer_idx].linear.weight.data[:, full_idx_tens]
                                .to(self.device))

                        # full output bias is copied for the final layer
                        self.sub_model.layers[layer_idx].linear.bias.data = (
                            self.base_model.layers[layer_idx].linear.bias.data
                                .to(self.device))
                    else:
                        prev_idx, full_prev_idx = all_indices[layer_idx - 1][0]
                        next_idx, full_next_idx = all_indices[layer_idx][0]
                        correct_cols = self.base_model.layers[layer_idx].linear.weight.data[:, full_prev_idx]
                        self.sub_model.layers[layer_idx].linear.weight.data = (
                            correct_cols[next_idx, :].to(self.device))
                        self.sub_model.layers[layer_idx].linear.bias.data = (
                            self.base_model.layers[layer_idx].linear.bias.data[next_idx]
                                .to(self.device))

            for layer_idx in range(len(all_indices) + 1):
                if layer_idx == 0:
                    for site_i in range(1, self.args.num_subnet):
                        idx_tens, full_idx_tens = all_indices[0][site_i]
                        if self.args.rank == 0:
                            # broadcast the weight
                            bcast_weight = self.base_model.layers[0].linear.weight.data[idx_tens, :]
                            bcast_weight = bcast_weight.to(self.device)
                            broadcast_weight(
                                bcast_weight, rank_list=[0, site_i], source=0)
                            bcast_weight = bcast_weight.to('cpu')

                            # broadcastt the bias
                            bcast_bias = self.base_model.layers[0].linear.bias.data[idx_tens]
                            bcast_bias = bcast_bias.to(self.device)
                            broadcast_weight(
                                bcast_bias, rank_list=[0, site_i], source=0)
                            bcast_bias = bcast_bias.to('cpu')
                        else:
                            broadcast_weight(
                                self.sub_model.layers[0].linear.weight.data,
                                rank_list=[0, site_i], source=0)
                            broadcast_weight(
                                self.sub_model.layers[0].linear.bias.data,
                                rank_list=[0, site_i], source=0)

                elif layer_idx == len(all_indices):
                    for site_i in range(1, self.args.num_subnet):
                        idx_tens, full_idx_tens = all_indices[-1][site_i]
                        if self.args.rank == 0:
                            # do NOT need to broadcast bias, it is already done above
                            bcast_weight = self.base_model.layers[layer_idx].linear.weight.data[:, full_idx_tens]
                            bcast_weight = bcast_weight.to(self.device)
                            broadcast_weight(
                                bcast_weight, rank_list=[0, site_i], source=0)
                            bcast_weight = bcast_weight.to('cpu')
                        else:
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.weight.data,
                                rank_list=[0, site_i], source=0)
                else:
                    for site_i in range(1, self.args.num_subnet):
                        prev_tens, full_prev_tens = all_indices[layer_idx - 1][site_i]
                        next_tens, full_next_tens = all_indices[layer_idx][site_i]
                        if self.args.rank == 0:
                            # broadcast the weights
                            correct_rows = (
                                self.base_model.layers[layer_idx].linear.weight.data[:, full_prev_tens])
                            correct_rows = correct_rows[next_tens, :]
                            correct_rows = correct_rows.to(self.device)
                            broadcast_weight(
                                correct_rows, rank_list=[0, site_i],
                                source=0)
                            correct_rows = correct_rows.to('cpu')

                            bcast_bias = self.base_model.layers[layer_idx].linear.bias.data[next_tens]
                            bcast_bias = bcast_bias.to(self.device)
                            broadcast_weight(
                                bcast_bias, rank_list=[0, site_i], source=0)
                            bcast_bias = bcast_bias.to('cpu')
                        else:
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.weight.data, rank_list=[0, site_i],
                                source=0)
                            broadcast_weight(
                                self.sub_model.layers[layer_idx].linear.bias.data, rank_list=[0, site_i],
                                source=0)

        self.current_partition = all_indices


def train(
        ist_model, args, g, cluster_iterator, labels, train_mask, val_mask,
        test_mask, train_nid, device):
    # track the metrics throughout training
    if args.rank == 0:
        test_accs = []
        val_accs = []
        losses = []
    else:
        test_accs = None
        val_accs = None
        losses = None

    # create the loss
    loss_fcn = torch.nn.CrossEntropyLoss()
    local_epochs = args.n_epochs // args.num_subnet
    train_nid = torch.from_numpy(train_nid).to(device)

    # training loop
    running_loss = 0.
    curr_iter = 0.
    total_iter = 0.
    total_time = 0.
    start_time = time.time()
    for e in range(local_epochs):
        print(f'{args.rank}: running epoch {e} / {local_epochs}', flush=True)
        lr = args.lr
        run_eval = True  # run eval once every epoch
        for j, cluster in enumerate(cluster_iterator):
            # dispatch the model
            if total_iter % args.iter_per_site == 0:
                if e > 0:
                    dist.barrier(group=dist.group.WORLD)
                    ist_model.dispatch_model()
                ist_model.sub_model.train()
                optimizer = torch.optim.Adam(
                    ist_model.sub_model.parameters(), lr=lr,
                    weight_decay=args.weight_decay)
            optimizer.zero_grad()
            cluster = cluster.to(device)
            pred = ist_model.sub_model(cluster)
            batch_labels = cluster.ndata['label']
            batch_train_mask = cluster.ndata['train_mask']
            loss = loss_fcn(
                pred[batch_train_mask], batch_labels[batch_train_mask])
            loss.backward()
            running_loss += float(loss)
            optimizer.step()

            # sync model every "iter_per_site" iterations and at end of training
            total_iter += 1
            curr_iter += 1  # used for computing average loss between evaluations
            if (
                    (total_iter % args.iter_per_site == 0)
                    or ((j == len(cluster_iterator) - 1) and (e == local_epochs - 1))):
                print(f'{args.rank} running sync @ iter #{total_iter}, epoch {e}')
                dist.barrier(group=dist.group.WORLD)
                ist_model.sync_model()

                # run eval during each new epoch and at the end of training
                # only run eval after updates are synchronized into global model
                if (
                        run_eval or
                        ((j == len(cluster_iterator) - 1) and (e == local_epochs - 1))):
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    run_eval = False
                    if args.rank == 0:
                        ist_model.base_model.eval()
                        if args.use_f1:
                            val_acc = evaluate(ist_model.base_model, g.cpu(), labels.cpu(), val_mask.cpu(), 'f1')
                            test_acc = evaluate(ist_model.base_model, g.cpu(), labels.cpu(), test_mask.cpu(), 'f1')
                        else:
                            val_acc = evaluate(ist_model.base_model, g.cpu(), labels.cpu(), val_mask.cpu())
                            test_acc = evaluate(ist_model.base_model, g.cpu(), labels.cpu(), test_mask.cpu())
                        val_accs.append(val_acc)
                        test_accs.append(test_acc)
                        losses.append((running_loss / curr_iter))
                    running_loss = 0.
                    curr_iter = 0.
                    start_time = time.time()

    # make sure training results are the last thing to be printed
    # so that it can be easily grepped with a python script
    dist.barrier(group=dist.group.WORLD)
    if args.rank == 0:
        # save training curve and results
        title = args.fig_name
        os.makedirs(args.fig_dir, exist_ok=True)
        plt.plot(val_accs)
        plt.title(title)
        plt.savefig(os.path.join(args.fig_dir, title + '.png'))

        if args.save_results:
            results = {
                'total_time': total_time,
                'trn_losses': losses,
                'val_accs': val_accs,
                'test_accs': test_accs,
            }
            fn = title + '_result.pckl'
            result_path = os.path.join(args.fig_dir, fn)
            with open(result_path, 'wb') as f:
                pickle.dump(results, f)
        else:
            print(f'Training Time: {total_time:.4f}', flush=True)
            print(f'Last Val: {val_accs[-1]:.4f}', flush=True)
            print(f'Best Val: {max(val_accs):.4f}', flush=True)
            print(f'Last Test: {test_accs[-1]:.4f}', flush=True)
            print(f'Best Test: {max(test_accs):.4f}', flush=True)


def get_data(args, device):
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
    n_edges = g.number_of_edges()
    g = g.long()

    # create the cluster gcn iterator
    cluster_iterator = ClusterIter(
        args.dataset, g, args.psize, args.batch_size,
        train_nid, use_pp=args.use_pp)

    # set device for dataset tensors
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    g = g.int().to(device)
    return (
        g, cluster_iterator, train_mask, val_mask, test_mask, labels,
        train_nid, in_feats, n_classes, n_edges)


def main():
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--iter_per_site", type=int, default=5)
    parser.add_argument("--num_subnet", type=int, default=2,
                        help="number of sub networks")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--use_layernorm", type=bool, default=False,
                        help="Whether use layernorm (default=False)")
    parser.add_argument('--dist-backend', type=str, default='nccl', metavar='S',
                        help='backend type for distributed PyTorch')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9971', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed PyTorch')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size")
    parser.add_argument("--psize", type=int, default=1500,
                        help="partition number")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--rnd-seed", type=int, default=3,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--use-pp", action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--normalize", action='store_true',
                        help="whether to use normalized feature")
    parser.add_argument("--save_results", action='store_true')
    parser.add_argument("--fig-dir", type=str, default='../report/example_pic/')
    parser.add_argument("--fig-name", type=str, default='name')
    parser.add_argument("--use-f1", action='store_true')
    args = parser.parse_args()

    assert (args.n_hidden % args.num_subnet) == 0

    # set all the random seeds
    print('Setting seeds', flush=True)
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set the proper GPU
    assert args.cuda_id < torch.cuda.device_count()
    device = torch.device(f'cuda:{args.cuda_id}')

    # initialize the distributed process group
    print(f'{args.rank} initializing process', flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, rank=args.rank,
        world_size=args.num_subnet)
    print(f'Process spawned: {args.rank} --> {device}', flush=True)

    # get the data and setup the dataset
    dataset = get_data(args, device)
    (g, cluster_iterator, train_mask, val_mask, test_mask, labels,
     train_nid, in_feats, n_classes, n_edges) = dataset

    # get the main model
    ist_model = DistributedGNNWrapper(args, g, in_feats, n_classes, device)
    print(f'{args.rank}: start initial dispatch', flush=True)
    ist_model.ini_sync_dispatch_model()
    print(f'{args.rank}: finish initial dispatch', flush=True)
    train(
        ist_model, args, g, cluster_iterator, labels, train_mask, val_mask,
        test_mask, train_nid, device)


if __name__ == '__main__':
    main()
