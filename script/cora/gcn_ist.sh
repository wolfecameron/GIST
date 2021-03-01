savedir=../report/report_1023/
srun -p mediaf1 --gres gpu:1 python train_ist.py \
--dataset cora \
--use_ist 'True' \
--num_subnet 8 \
--gpu 1 \
--dropout 0.5 \
--n-epochs 200 \
--lr 0.05 \
--use_layernorm 'False' \
--self_loop 'True' \
--savedir ${savedir} \
--save_word 0 \
2>&1 | tee -a ${savedir}/log.txt
