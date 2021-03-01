# sh script/reddit/run_baseline_amazon.sh
echo "Starting running baseline for Reddit..."
if [ $USER == "jkyang" ]; then
  srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=icml \
  --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
  python cluster_gcn/cluster_gcn.py --dataset reddit-self-loop \
  --lr 0.01 --n-epochs 80 --batch-size 20 --n-hidden 256 --n-layers 4 --weight-decay 0 \
  --dropout 0.2 --use-layernorm --rnd-seed 0 --normalize --use-f1
else
  python cluster_gcn/cluster_gcn.py --dataset reddit-self-loop --normalize --use-layernorm --use-f1
fi
