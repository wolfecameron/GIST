srun -p mediaf1 --gres gpu:1 python train_ist.py \
--dataset cora \
2>&1 | tee -a ../log/cora_gcn_ist_1022.txt
