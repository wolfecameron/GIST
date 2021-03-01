srun -p mediaf --gres gpu:1 python train.py \
--dataset pubmed \
--gpu 1 \
--self-loop \
2>&1 | tee -a ../log/pubmed_baseline.txt
