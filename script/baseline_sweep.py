# performs a hyperparameter sweep with many tests
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle

BASE_TEST_NAME = f'baseline_depth_and_width_sweep'

datasets = ['cora', 'citeseer', 'pubmed']
n_hiddens = [16, 32, 64, 128, 256, 512, 1024]
layers = [1, 2, 3, 4, 6, 8] 
lrs = [5e-3, 0.01, 0.05, 0.1]
num_trials = 5
dropout = 0.5
use_layernorm = True
base_epoch = 400
weight_decay = 5e-4
results = {}
for dataset in datasets:
  TEST_NAME = BASE_TEST_NAME + f'_{dataset}'
  for n_layer in layers:
    for n_hid in n_hiddens:
      for lr in lrs:
        exp_str = f'baseline_{n_layer}_{n_hid}_{lr}'
        for i in range(num_trials):
          command = (
              f'python train.py --dataset {dataset} --lr {lr} --n-layers {n_layer}'
              f' --weight-decay {weight_decay} --use_layernorm {use_layernorm}'
              f' --n-hidden {n_hid} --n-epochs {base_epoch} --dropout {dropout} --lr_scheduler')
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
