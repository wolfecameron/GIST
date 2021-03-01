# python script/amazon/run_ist_sweep_amazon.py

import os, csv, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_partition = 15000
TEST_NAME = f'amazon_ist_ps{num_partition}'
print(TEST_NAME, flush=True)
os.makedirs(f'../report/{TEST_NAME}/', exist_ok=True)

num_trials = 1
dropout = 0.2
wd = 0

layers = [4, 3, 2, 1]
hiddens = [400]
lrs = [5e-3]
n_epochs = [200]
local_iters = [500, 1000, 2500]
num_ists = [2, 4, 8]

# For missing experiments
# layers = [1]
# hiddens = [256]
# lrs = [1e-2]
# n_epochs = [40]
# num_ists = [1, 2, 4]
# local_iters = [250]

results = {}
print('Number of Experiment: {}'.format(len(hiddens) * len(layers) * len(num_ists) * len(n_epochs)
                                        * len(lrs) * len(local_iters) * num_trials), flush=True)

for n_layer in layers:
    for n_hid in hiddens:
        for lr in lrs:
            for n_epoch in n_epochs:
                for local_iter in local_iters:
                    for num_ist in num_ists:
                        train_times = []
                        best_tests = []
                        best_vals = []
                        last_vals = []
                        last_tests = []
                        exp_str = f'l{n_layer}_h{n_hid}_e{n_epoch}_lr{lr}_local{local_iter}_{num_ist}ist'
                        for i in range(num_trials):
                            print(f'Running Amazon_ist_{exp_str} Trial #{i}')
                            if num_ist == 1:
                                command = (
                                    f'srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1'
                                    f' --job-name={TEST_NAME} --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73'
                                    f' python cluster_gcn/cluster_gcn.py --dataset amazon2m'
                                    f' --lr {lr} --weight-decay {wd} --psize {num_partition} --batch-size 10'
                                    f' --n-epochs {n_epoch} --n-hidden {n_hid} --n-layers {n_layer}'
                                    f' --dropout {dropout} --normalize --rnd-seed {i} --use-f1'
                                    f' --fig-dir ../report/{TEST_NAME} --fig-name {exp_str}_{i} --use-layernorm')
                                os.system(command + f' > ../report/{TEST_NAME}/tmp.txt')
                            else:
                                for ist_id in range(num_ist):
                                    sub_command = (
                                        f'srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 '
                                        f'--job-name={TEST_NAME} --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 '
                                        f'python cluster_gcn/cluster_gcn_ist_distrib.py'
                                        f' --psize {num_partition} --iter_per_site {local_iter}'
                                        f' --num_subnet {num_ist} --dropout {dropout} --lr {lr} --batch-size 10'
                                        f' --n-epochs {n_epoch} --n-hidden {n_hid} --n-layers {n_layer}'
                                        f' --weight-decay {wd} --use_layernorm True --use-f1'
                                        f' --rnd-seed 0 --normalize --rank {ist_id} --cuda-id 0'
                                        f' --dataset amazon2m --fig-dir ../report/{TEST_NAME}'
                                        f' --fig-name {exp_str}_{i}'
                                    )
                                    if ist_id == 0:
                                        os.system(sub_command + f' > ../report/{TEST_NAME}/tmp.txt &')
                                    elif ist_id < num_ist - 1:
                                        os.system(sub_command + ' &')
                                    else:
                                        os.system(sub_command)

                            with open(f'../report/{TEST_NAME}/tmp.txt', 'r') as f:
                                trn_output = f.readlines()
                            try:
                                train_time = float(trn_output[-5].split(':')[1])

                                last_val = float(trn_output[-4].split(':')[1])
                                best_val = float(trn_output[-3].split(':')[1])

                                last_test = float(trn_output[-2].split(':')[1])
                                best_test = float(trn_output[-1].split(':')[1])
                            except:
                                train_time = -1
                                last_val = -1
                                best_val = -1
                                last_test = -1
                                best_test = -1
                            print('Time: {}, Acc: {}/{}, {}/{}'.format(train_time,
                                                                       last_val, best_val, last_test, best_test))
                            train_times.append(train_time)
                            best_vals.append(best_val)
                            last_vals.append(last_val)
                            best_tests.append(best_test)
                            last_tests.append(last_test)

                        results[exp_str] = {}
                        results[exp_str]['final_test_acc'] = sum(last_tests) / len(last_tests)
                        results[exp_str]['final_val_acc'] = sum(last_vals) / len(last_vals)
                        results[exp_str]['best_val_acc'] = sum(best_vals) / len(best_vals)
                        results[exp_str]['best_test_acc'] = sum(best_tests) / len(best_tests)
                        results[exp_str]['train_time'] = sum(train_times) / len(train_times)

                        save_exp_name = os.path.join(f'../report/{TEST_NAME}/{TEST_NAME}_exp.csv')
                        fieldnames = ['num_layer', 'num_hidden', 'num_epochs', 'lr',
                                      'weight_decay', 'dropout', 'ist', 'local_iter',
                                      'val_acc_last', 'val_acc_best', 'test_acc_last', 'test_acc_best', 'time']

                        write_content = {'ist': num_ist, 'num_layer': n_layer, 'num_hidden': n_hid,
                                         'local_iter': local_iter, 'num_epochs': n_epoch, 'lr': lr,
                                         'weight_decay': wd, 'dropout': dropout,
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

                        # exit this experiment
                        while len(os.popen(f'squeue -u jkyang -n {TEST_NAME}').read()) != 85:
                            time.sleep(10)
                            os.system(f'scancel -u jkyang -n {TEST_NAME}')
                            print('Job should be killed for now', flush=True)
