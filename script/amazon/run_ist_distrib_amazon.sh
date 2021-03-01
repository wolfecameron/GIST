NUM_IST=8     # 1,2,4,8
echo "Starting running $NUM_IST-ist for Amazon..."
if [ $USER == "jkyang" ]; then
  for ((i=0; i<NUM_IST; i++))
  do
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=icml \
            --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
    python cluster_gcn/cluster_gcn_ist_distrib.py --iter_per_site 500 --num_subnet $NUM_IST --dropout 0.2 \
            --psize 15000 --lr 0.01 --n-epochs 200 --batch-size 10 --n-hidden 2048 --n-layers 4 --weight-decay 0 \
            --use_layernorm True --rnd-seed 0 --normalize --rank $i --cuda-id 0 --dataset amazon2m --use-f1 &
  done
  wait
else
  for ((i=0; i<NUM_IST; i++))
  do
      python cluster_gcn/cluster_gcn_ist_distrib.py --iter_per_site 500 --num_subnet $NUM_IST --dropout 0.2 \
          --psize 15000 --lr 0.01 --n-epochs 200 --batch-size 10 --n-hidden 2048 --n-layers 4 --weight-decay 0 \
          --use_layernorm True --rnd-seed 0 --normalize --rank $i --cuda-id 0 --dataset amazon2m --use-f1 &
  done
fi
