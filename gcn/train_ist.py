import os

os.environ["DGL_REPO"] = "http://data.dgl.ai/"
import argparse, time
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, RedditDataset
from models import GCN


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def draw_acc_curve(record, save_root, save_filename):
    acc_val = np.array(record)[:, 0]
    acc_test = np.array(record)[:, 1]
    plt.title("Validation Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(len(acc_val)), acc_val, 'blue')
    plt.plot(range(len(acc_val)), acc_test, 'orange')
    np.save(os.path.join(save_root, 'record_'+save_filename), record)
    plt.savefig(os.path.join(save_root, save_filename+'.jpg'))
    plt.close()


def main(args):
    # convert boolean type for args
    assert args.use_ist in ['True', 'False'], ["Only True or False for use_ist, get ",
                                               args.use_ist]
    assert args.split_input in ['True', 'False'], ["Only True or False for split_input, get ",
                                                   args.split_input]
    assert args.split_output in ['True', 'False'], ["Only True or False for split_output, get ",
                                                   args.split_output]
    assert args.self_loop in ['True', 'False'], ["Only True or False for self_loop, get ",
                                                 args.self_loop]
    assert args.use_layernorm in ['True', 'False'], ["Only True or False for use_layernorm, get ",
                                                     args.use_layernorm]
    assert args.use_random_proj in ['True', 'False'], ["Only True or False for use_random_proj, get ",
                                                       args.use_random_proj]
    use_ist = (args.use_ist == 'True')
    split_input = (args.split_input == 'True')
    split_output = (args.split_output == 'True')
    self_loop = (args.self_loop == 'True')
    use_layernorm = (args.use_layernorm == 'True')
    use_random_proj = (args.use_random_proj == 'True')

    # make sure hidden layer is the correct shape
    assert (args.n_hidden % args.num_subnet) == 0

    # load and preprocess dataset
    global t0
    if args.dataset in {'cora', 'citeseer', 'pubmed'}:
        data = load_data(args)
    else:
        raise NotImplementedError(f'{args.dataset} is not a valid dataset')

    # randomly project the input to make it dense
    if use_random_proj:
        # densify input features with random projection
        from sklearn import random_projection

        # make sure input features are divisible by number of subnets
        # otherwise some parameters of the last subnet will be handled improperly
        n_components = int(data.features.shape[-1] / args.num_subnet) * args.num_subnet
        transformer = random_projection.GaussianRandomProjection(n_components=n_components)
        new_feature = transformer.fit_transform(data.features)
        features = torch.FloatTensor(new_feature)
    else:
        assert (data.features.shape[-1] % args.num_subnet) == 0.
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

    # initialize graph
    dur = []
    record = []
    sub_models = []
    opt_list = []
    sub_dict_list = []
    main_dict = None
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        if use_ist:
            model.eval()
            # IST training:
            # Distribute parameter to sub networks
            num_subnet = args.num_subnet
            if (epoch % args.iter_per_site) == 0.:
                main_dict = model.state_dict()
                feats_idx = [] # store all layer indices within a single list

                # create input partition
                if split_input:
                    feats_idx.append(torch.chunk(torch.randperm(in_feats), num_subnet))
                else:
                    feats_idx.append(None)

                # create hidden layer partitions
                for i in range(1, args.n_layers):
                    feats_idx.append(torch.chunk(torch.randperm(args.n_hidden), num_subnet))

                # create output layer partitions
                if split_output:
                    feats_idx.append(torch.chunk(torch.randperm(args.n_hidden), num_subnet))
                else:
                    feats_idx.append(None)

            for subnet_id in range(args.num_subnet):
                if (epoch % args.iter_per_site) == 0.:
                    # create the sub model to train
                    sub_model = GCN(
                            g, in_feats, args.n_hidden, n_classes,
                            args.n_layers, F.relu, args.dropout, use_layernorm,
                            split_input, split_output, args.num_subnet) 
                    sub_model = sub_model.to(device)
                    sub_dict = main_dict.copy()

                    # split input params
                    if split_input:
                        idx = feats_idx[0][subnet_id]
                        sub_dict['layers.0.weight'] = main_dict['layers.0.weight'][idx, :]

                    # split hidden params (and output params)
                    for i in range(1, args.n_layers + 1):
                        if i == args.n_layers and not split_output:
                            pass # params stay the same 
                        else:
                            idx = feats_idx[i][subnet_id]
                            sub_dict[f'layers.{i - 1}.weight'] = sub_dict[f'layers.{i -1}.weight'][:, idx]
                            sub_dict[f'layers.{i - 1}.bias'] = main_dict[f'layers.{i - 1}.bias'][idx]
                            sub_dict[f'layers.{i}.weight'] = main_dict[f'layers.{i}.weight'][idx, :]

                    # use a lr scheduler
                    curr_lr = args.lr
                    if epoch >= int(args.n_epochs*0.5):
                        curr_lr /= 10
                    if epoch >= int(args.n_epochs*0.75):
                        curr_lr /= 10

                    # import params into subnet for training
                    sub_model.load_state_dict(sub_dict)
                    sub_models.append(sub_model)
                    sub_models = sub_models[-num_subnet:]
                    optimizer = torch.optim.Adam(
                            sub_model.parameters(), lr=curr_lr,
                            weight_decay=args.weight_decay)
                    opt_list.append(optimizer)
                    opt_list = opt_list[-num_subnet:]
                else:
                    sub_model = sub_models[subnet_id]
                    optimizer = opt_list[subnet_id]

                # train a sub network
                optimizer.zero_grad()
                sub_model.train()
                if split_input:
                    model_input = features[:, feats_idx[0][subnet_id]]
                else:
                    model_input = features
                logits = sub_model(model_input)
                loss = loss_fcn(logits[train_mask], labels[train_mask])

                # reset optimization for every sub training
                loss.backward()
                optimizer.step()

                # save sub model parameter
                if (
                        ((epoch + 1) % args.iter_per_site == 0.)
                        or (epoch == args.n_epochs - 1)):
                    sub_dict = sub_model.state_dict()
                    sub_dict_list.append(sub_dict)
                    sub_dict_list = sub_dict_list[-num_subnet:]

            # Merge parameter to main network:
            # force aggregation if training about to end
            if (
                    ((epoch + 1) % args.iter_per_site == 0.)
                    or (epoch == args.n_epochs - 1)):
                #keys = main_dict.keys()
                update_dict = main_dict.copy()

                # copy in the input parameters
                if split_input:
                    if args.n_layers <= 1 and not split_output:
                        for idx, sub_dict in zip(feats_idx[0], sub_dict_list):
                            update_dict['layers.0.weight'][idx, :] = sub_dict['layers.0.weight']
                    else:
                        for i, sub_dict in enumerate(sub_dict_list):
                            curr_idx = feats_idx[0][i]
                            next_idx = feats_idx[1][i]
                            correct_rows = update_dict['layers.0.weight'][curr_idx, :]
                            correct_rows[:, next_idx] = sub_dict['layers.0.weight']
                            update_dict['layers.0.weight'][curr_idx, :] = correct_rows
                else:
                    if args.n_layers <= 1 and not split_output:
                        update_dict['layers.0.weight'] = sum(sub_dict['layers.0.weight'] for sub_dict in sub_dict_list) / len(sub_dict_list)
                    else:
                        for i, sub_dict in enumerate(sub_dict_list):
                            next_idx = feats_idx[1][i]
                            update_dict['layers.0.weight'][:, next_idx] = sub_dict['layers.0.weight']

                # copy the rest of the parameters
                for i in range(1, args.n_layers + 1):
                    if i == args.n_layers:
                        if not split_output:
                            update_dict[f'layers.{i-1}.bias'] = sum(sub_dict[f'layers.{i-1}.bias'] for sub_dict in sub_dict_list) / len(sub_dict_list)
                            update_dict[f'layers.{i}.weight'] = sum(sub_dict[f'layers.{i}.weight'] for sub_dict in sub_dict_list) / len(sub_dict_list)
                        else:
                            for idx, sub_dict in zip(feats_idx[i], sub_dict_list):
                                update_dict[f'layers.{i-1}.bias'][idx] = sub_dict[f'layers.{i-1}.bias']
                                update_dict[f'layers.{i}.weight'][idx, :] = sub_dict[f'layers.{i}.weight']
                    else:
                        if i >= args.n_layers - 1 and not split_output:
                            for idx, sub_dict in zip(feats_idx[i], sub_dict_list):
                                update_dict[f'layers.{i-1}.bias'][idx] = sub_dict[f'layers.{i-1}.bias']
                                update_dict[f'layers.{i}.weight'][idx, :] = sub_dict[f'layers.{i}.weight']
                        else:
                            for idx, sub_dict in enumerate(sub_dict_list):
                                curr_idx = feats_idx[i][idx]
                                next_idx = feats_idx[i+1][idx]
                                update_dict[f'layers.{i-1}.bias'][curr_idx] = sub_dict[f'layers.{i-1}.bias']
                                correct_rows = update_dict[f'layers.{i}.weight'][curr_idx, :]
                                correct_rows[:, next_idx] = sub_dict[f'layers.{i}.weight']
                                update_dict[f'layers.{i}.weight'][curr_idx, :] = correct_rows 
                model.load_state_dict(update_dict)

        else:
            raise NotImplementedError('Should train with IST')

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc_val = evaluate(model, features, labels, val_mask)
        acc_test = evaluate(model, features, labels, test_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val Accuracy {:.4f} | Test Accuracy {:.4f} |"
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc_val, acc_test, n_edges / np.mean(dur) / 1000))
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
    parser.add_argument("--use_ist", type=str, default="True",
                        help="whether use IST training")
    parser.add_argument("--iter_per_site", type=int, default=5)
    parser.add_argument("--num_subnet", type=int, default=2,
                        help="number of sub networks")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--split_output", type=str, default="False")
    parser.add_argument("--split_input", type=str, default="True")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self_loop", type=str, default='True',
                        help="graph self-loop (default=True)")
    parser.add_argument("--use_layernorm", type=str, default='True',
                        help="Whether use layernorm (default=False)")
    parser.add_argument("--use_random_proj", type=str, default='True',
                        help="Whether use random projection to densitify (default=False)")
    args = parser.parse_args()
    main(args)
