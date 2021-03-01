srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=icml \
--kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
python train_ist_reddit.py \
--dataset reddit \
--gpu 1 \
--self_loop True \
2>&1 | tee -a ../log/reddit_baseline.txt
