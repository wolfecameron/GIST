import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pickle

TEST_NAME = f'gat_reddit_00'

num_trials = 2
dropout = 0.2
n_epoch = 80

local_iters = [500]
lrs = [5e-3]
hiddens = [512]
layers = [1]
num_heads = [2, 4, 8]

# check if results have been saved already
if os.path.exists(f'./results/{TEST_NAME}.pckl'):
    with open(f'./results/{TEST_NAME}.pckl', 'rb') as f:
        results = pickle.load(f)
else:
    results = {}

for n_layer in layers:
    for n_hid in hiddens:
        for local_iter in local_iters:
            for lr in lrs:
                for n_head in num_heads:
                    train_times = []
                    best_tests = []
                    best_vals = []
                    last_tests = []
                    exp_str = f'{n_layer}_{n_hid}_{local_iter}_{lr}'
                    if not exp_str in results.keys():
                        for i in range(num_trials):
                            print(f'Running {exp_str} Trial #{i}')
                            c1 = (
                                f'python3 cluster_gcn/cluster_gcn_ist_distrib_gat.py --iter_per_site {local_iter}'
                                f' --num_subnet 2 --dropout {dropout} --lr {lr} --n-epochs {n_epoch}'
                                f' --n-hidden {n_hid} --n-layers {n_layer} --weight-decay 0'
                                f' --use_layernorm True --rnd-seed {i} --normalize --n-heads {n_head}'
                                f' --rank 1 --cuda-id 1 --dataset reddit-self-loop &')
                            c2 = (
                                f'python3 cluster_gcn/cluster_gcn_ist_distrib_gat.py --iter_per_site {local_iter}'
                                f' --num_subnet 2 --dropout {dropout} --lr {lr} --n-epochs {n_epoch}'
                                f' --n-hidden {n_hid} --n-layers {n_layer} --weight-decay 0'
                                f' --use_layernorm True --rnd-seed {i} --normalize --n-heads {n_head}'
                                f' --rank 0 --cuda-id 0 --dataset reddit-self-loop')
                            os.system(c1)
                            os.system(c2 + ' > trn_output.txt')
                            with open('trn_output.txt', 'r') as f:
                                trn_output = f.readlines()
                            train_time = float(trn_output[-4].split(':')[1])
                            last_test = float(trn_output[-3].split(':')[1])
                            best_test = float(trn_output[-2].split(':')[1])
                            best_val = float(trn_output[-1].split(':')[1])
                            train_times.append(train_time)
                            best_tests.append(best_test)
                            best_vals.append(best_val)
                            last_tests.append(last_test)
                        results[exp_str] = {}
                        results[exp_str]['final_test_acc'] = (sum(last_tests) / len(last_tests), last_tests)
                        results[exp_str]['best_val_acc'] = (sum(best_vals) / len(best_vals), best_vals)
                        results[exp_str]['best_test_acc'] = (sum(best_tests) / len(best_tests), best_tests)
                        results[exp_str]['train_time'] = (sum(train_times) / len(train_times), train_times)

                        # make sure results are saved intermediately
                        with open(f'./results/{TEST_NAME}.pckl', 'wb') as f:
                            pickle.dump(results, f)
                    else:
                        print(f'{exp_str} already complete')
