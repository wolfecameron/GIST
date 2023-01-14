# GIST: Distributed Training for Large-Scale GCNs


[![paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2102.10424)
&nbsp;
[![blogpost](https://img.shields.io/badge/Blog%20Post-medium-2596be)](https://towardsdatascience.com/effortless-distributed-training-of-ultra-wide-gcns-6e9873f58a50)
&nbsp;
[![video](https://img.shields.io/badge/Video-YouTube-ff2038)](https://www.youtube.com/watch?v=lc9rYKHkgv0&ab_channel=ResearchMadeSimple)
&nbsp;


This is a public code repository for the publication:
> [**GIST: Distributed Training for Large-Scale Graph Convolutional Networks**](https://arxiv.org/abs/2102.10424)<br>
> Cameron R Wolfe, Jingkang Yang, Arindam Chowdhury, Chen Dun, Artun Bayer, Santiago Segarra, Anastasios Kyrillidis<br>

GIST enables the training of large-scale
GCNs (i.e., hidden layers with tens of thousands of nodes) with minimal training
time using a distributed, independent sub-GCN training methodology.
Our implementation is based on the Deep Graph Library (DGL) and we base many of
our experiments on the code provided through their library. All the documentation
for DGL is available online (https://www.dgl.ai).

## Environment/Dependencies

Requires anaconda to be installed (python3)
Anaconda can be installed at https://www.anaconda.com/products/individual

```bash
conda create -n gist python=3.6 anaconda
conda activate gist
pip install -r requirements.txt
```

## File Arrangement

Here we summarize all files present in this repo and their purpose.
```
+-- gcn/ : non-distributed, small-scale experiments, do NOT use graph clustering
|   +-- gcn.py: contains the model definition (vanilla GCN) for small-scale datasets
|   +-- train.py: normal GCN training script
|   +-- train_ist.py: GIST training script (simulates distributed training with single GPU)
|
+-- cluster_gcn/ : distributed, large-scale experiments that use graph clustering (Reddit and Amazon2M)
|   +-- cluster_gcn.py: single-GPU training scipt with graph clustering
|   +-- cluster_gcn_ist_distrib.py: distributed GIST implementation with graph clustering
|   +-- cluster_gcn_ist_ultra_wide.py: same as above, but enables training ultra-wide models
|   +-- modules.py: contains GraphSAGE model definition
|   +-- AmazonDataset.py: custom pytorch dataset definition for Amazon2M
|   +-- partition_utils.py: helper functions for METIS clustering algorithm
|   +-- sampler.py: implements METIS graph clustering for GCN training
|   +-- utils.py: other random utilities
|
+-- script/ : contains script for automating many experiments to run at once
```

## Citing us
If you find this work useful, please cite our paper.
```
@article{wolfe2021gist,
  title={GIST: Distributed Training for Large-Scale Graph Convolutional Networks},
  author={Wolfe, Cameron R and Yang, Jingkang and Chowdhury, Arindam and Dun, Chen and Bayer, Artun and Segarra, Santiago and Kyrillidis, Anastasios},
  journal={arXiv preprint arXiv:2102.10424},
  year={2021}
}
```
