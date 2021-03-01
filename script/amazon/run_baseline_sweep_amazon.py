# performs a hyperparameter sweep with many tests
import os, csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# things to sweep over:
# lr, n_hidden, weight decay, n-layers

# things that are fixed:
# psize, n-epochs, use-pp, self-loop, dropout, normalize

psize = 15000
TEST_NAME = f'amazon_baseline_metis{psize}'
print(TEST_NAME, flush=True)
os.makedirs(f'../report/{TEST_NAME}/', exist_ok=True)

model_type = 'sage'
num_trials = 1
batch_size = 10
dropouts = [0.2]
weight_decay = [0.0]

# num_epochs = [400]
# n_hidden = [100, 200, 400]
# n_layers = [1, 2, 3, 4]
# lrs = [5e-3, 1e-2, 5e-2]

# For missing experiments
num_epochs = [10]
n_hidden = [100]
n_layers = [1,2,3,4]
lrs = [1e-2]

results = {}
print('Number of Experiment: {}'.format(len(dropouts) * len(num_epochs) * len(n_hidden)
                                        * len(n_layers) * len(lrs) * len(weight_decay) * num_trials), flush=True)
for n_layer in n_layers:
    for n_hid in n_hidden:
        for lr in lrs:
            for num_epoch in num_epochs:
                for wd in weight_decay:
                    for dropout in dropouts:
                        train_times = []
                        best_tests = []
                        best_vals = []
                        last_vals = []
                        last_tests = []
                        exp_str = f'l{n_layer}_h{n_hid}_e{num_epoch}_lr{lr}_wd{wd}'
                        for i in range(num_trials):
                            print(f'Running Amazon_{exp_str} Trial #{i}')
                            command = (
                                f'srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=icml'
                                f' --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73'
                                f' python cluster_gcn/cluster_gcn.py --dataset amazon2m'
                                f' --lr {lr} --weight-decay {wd} --psize {psize} --batch-size {batch_size}'
                                f' --n-epochs {num_epoch} --n-hidden {n_hid} --n-layers {n_layer}'
                                f' --dropout {dropout} --normalize --rnd-seed {i} --use-f1'
                                f' --fig-dir ../report/{TEST_NAME} --fig-name {exp_str}_{i}'
                                f' --model-type {model_type} --use-layernorm')
                            try:
                                os.system(command + f' > ../report/{TEST_NAME}/tmp.txt')

                                with open(f'../report/{TEST_NAME}/tmp.txt', 'r') as f:
                                    trn_output = f.readlines()
                                train_time = float(trn_output[-5].split(':')[1])

                                last_val = float(trn_output[-4].split(':')[1])
                                best_val = float(trn_output[-3].split(':')[1])

                                last_test = float(trn_output[-2].split(':')[1])
                                best_test = float(trn_output[-1].split(':')[1])
                                print('Time: {}, Acc: {}/{}, {}/{}'.format(train_time,
                                                                           last_val, best_val, last_test, best_test))
                                train_times.append(train_time)
                                best_vals.append(best_val)
                                last_vals.append(last_val)
                                best_tests.append(best_test)
                                last_tests.append(last_test)
                            except Exception:
                                print('The experiment somehow fails.', flush=True)

                        results[exp_str] = {}
                        results[exp_str]['final_test_acc'] = sum(last_tests) / len(last_tests)
                        results[exp_str]['final_val_acc'] = sum(last_vals) / len(last_vals)
                        results[exp_str]['best_val_acc'] = sum(best_vals) / len(best_vals)
                        results[exp_str]['best_test_acc'] = sum(best_tests) / len(best_tests)
                        results[exp_str]['train_time'] = sum(train_times) / len(train_times)

                        save_exp_name = os.path.join(f'../report/{TEST_NAME}/exp.csv')
                        fieldnames = ['ist', 'num_layer', 'num_hidden', 'num_epochs',
                                      'lr', 'weight_decay', 'dropout',
                                      'val_acc_last', 'val_acc_best', 'test_acc_last', 'test_acc_best', 'time']

                        write_content = {'ist': 1, 'num_layer': n_layer, 'num_hidden': n_hid, 'num_epochs': num_epoch,
                                         'lr': lr, 'weight_decay': wd, 'dropout': dropout,
                                         'time': '{:.2f}'.format(results[exp_str]['train_time']),
                                         'val_acc_last': '{:.2f}'.format(100 * results[exp_str]['final_val_acc']),
                                         'val_acc_best': '{:.2f}'.format(100 * results[exp_str]['best_val_acc']),
                                         'test_acc_last': '{:.2f}'.format(100 * results[exp_str]['final_test_acc']),
                                         'test_acc_best': '{:.2f}'.format(100 * results[exp_str]['best_test_acc'])}
                        if not os.path.exists(save_exp_name):
                            with open(save_exp_name, 'w', newline='') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerow(write_content)
                        else:
                            with open(save_exp_name, 'a', newline='') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writerow(write_content)
