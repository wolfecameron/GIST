# performs a hyperparameter sweep with many tests
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle

BASE_TEST_NAME = f'local_iter_ist'

datasets = ['pubmed']#['cora', 'citeseer', 'pubmed']
num_trials = 5
n_hidden = 256
n_layers = 2
split_input = False
split_output = True
num_subnets = [2, 4, 8]
dropout = 0.5
use_layernorm = True
base_epoch = 400
results = {}
num_local_steps = [1, 5, 10, 20, 35, 50]
lrs = [5e-3, 0.01, 0.05, 0.1]
weight_decay = 5e-4
for dataset in datasets:
  TEST_NAME = BASE_TEST_NAME + f'_{dataset}'
  for num_sub in num_subnets:
    num_epoch = int(base_epoch / num_sub)
    for lr in lrs:
      for local_iter in num_local_steps:
        exp_str = f'{lr}_{local_iter}_{num_sub}_{num_epoch}'
        for i in range(num_trials):
          command = (
              f'python train_ist.py --dataset {dataset} --num_subnet {num_sub}'
              f' --lr {lr} --split_input {split_input} --split_output {split_output}'
              f' --weight-decay 5e-4 --use_layernorm {use_layernorm} --n-hidden {n_hidden}'
              f' --n-epochs {num_epoch} --dropout {dropout} --n-layers {n_layers}'
              f' --iter_per_site {local_iter} --use_random_proj True')
          trn_output = os.system(command + ' > trn_output.txt')
          with open('trn_output.txt', 'r') as f:
            trn_output = f.readlines()
          final_test_acc = float(trn_output[-3][-6:])
          best_val_acc = float(trn_output[-2][-6:])
          best_test_acc = float(trn_output[-1][-6:])
          if exp_str in results.keys():
            results[exp_str]['final_test_acc'].append(final_test_acc)
            results[exp_str]['best_val_acc'].append(best_val_acc)
            results[exp_str]['best_test_acc'].append(best_test_acc)
          else:
            results[exp_str] = {}
            results[exp_str]['final_test_acc'] = [final_test_acc]
            results[exp_str]['best_val_acc'] = [best_val_acc]
            results[exp_str]['best_test_acc'] = [best_test_acc]
            
  # aggregate results and save separately for each dataset
  all_results = []
  for exp_str in results.keys():
    exp_results = results[exp_str]
    mean_final_test = sum(exp_results['final_test_acc']) / len(exp_results['final_test_acc'])
    mean_best_val = sum(exp_results['best_val_acc']) / len(exp_results['best_val_acc'])
    mean_best_test = sum(exp_results['best_test_acc']) / len(exp_results['best_test_acc'])
    agg_results = (mean_best_test, mean_best_val, mean_final_test, exp_str)
    all_results.append(agg_results)
  with open(f'./results/{TEST_NAME}.pckl', 'wb') as f:
    pickle.dump(all_results, f)
