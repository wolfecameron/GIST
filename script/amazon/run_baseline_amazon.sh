# sh script/amazon/run_baseline_amazon.sh
echo "Starting running baseline for Amazon..."
if [ $USER == "jkyang" ]; then
  srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=icml \
  --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
  python cluster_gcn/cluster_gcn.py --dataset amazon2m \
  --lr 0.01 --n-epochs 200 --batch-size 10 --n-hidden 512 --n-layers 4 --weight-decay 0 \
  --psize 15000 --dropout 0.2 --use-layernorm --rnd-seed 0 --normalize --use-f1 --eval-cpu
else
  python cluster_gcn/cluster_gcn.py --dataset amazon2m --normalize --use-layernorm
fi
