import os

# python script/cora/sh_test.py
savedir = '../report/report_1029_dropout/'
os.system('mkdir {}'.format(savedir))

# baseline
use_ist_list = ['False']
num_subnet_list = [1]
n_epochs_list = [50, 100, 200]
lr_list = [0.1, 0.05, 0.025]
use_layernorm_list = ['True', 'False']
both_layers_list = ['False']
config_list_baseline = [[use_ist, num_subnet, n_epochs, lr, use_layernorm, both_layers]
                        for use_ist in use_ist_list
                        for num_subnet in num_subnet_list
                        for n_epochs in n_epochs_list
                        for lr in lr_list
                        for use_layernorm in use_layernorm_list
                        for both_layers in both_layers_list]

# ist
use_ist_list = ['True']
num_subnet_list = [2, 4, 8]
n_epochs_list = [25, 50, 100, 200, 400]
lr_list = [0.1, 0.05, 0.025]
use_layernorm_list = ['True', 'False']
both_layers_list = ['False']

config_list_ist = [[use_ist, num_subnet, n_epochs, lr, use_layernorm, both_layers]
                   for use_ist in use_ist_list
                   for num_subnet in num_subnet_list
                   for n_epochs in n_epochs_list
                   for lr in lr_list
                   for use_layernorm in use_layernorm_list
                   for both_layers in both_layers_list]

# config_list = config_list_baseline + config_list_ist

# config_list = [['True', 2, 200, 0.025, 'True', 'True']]

use_ist_list = ['True']
num_subnet_list = [2]
n_epochs_list = [50, 100, 200, 400]
lr_list = [0.1, 0.05, 0.025, 0.0125]
use_layernorm_list = ['True', 'False']
both_layers_list = ['True', 'False']
dropout_list = [0.5, 0.8, 1.0]

config_list = [[use_ist, num_subnet, n_epochs, lr, use_layernorm, both_layers, dropout]
               for use_ist in use_ist_list
               for num_subnet in num_subnet_list
               for n_epochs in n_epochs_list
               for lr in lr_list
               for use_layernorm in use_layernorm_list
               for both_layers in both_layers_list
               for dropout in dropout_list]


for i, [use_ist, num_subnet, n_epochs, lr, use_layernorm, both_layers, dropout] in enumerate(config_list):
    # Create Experimental Commands
    run_command = 'srun -K -p Test -n 1 --gres gpu:1 --ntasks-per-node 1 python train_ist.py ' \
                  '--dataset cora --use_ist {0} --num_subnet {1} --n-epochs {2} --lr {3} --use_layernorm {4} ' \
                  '--savedir {5}  --save_word {6} --both_layers {7} --dropout {8} ' \
                  '2>&1 | tee -a {5}/log.txt'.format(use_ist, num_subnet, n_epochs,
                                                     lr, use_layernorm, savedir, i, both_layers, dropout)
    # subprocess.Popen(run_command)
    print(run_command, flush=True)
    os.system(run_command)
