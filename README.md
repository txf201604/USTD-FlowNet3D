# USTD-FlowNet3D: Learning Point Features with Uniform Spatial and Temporal Distribution for Scene Flow Estimation
This is an open source project about point cloud scene flow estimation algorithm. This repository contains the Pytorch implementation for it.

![]([https://github.com/txf201604/USTD-FlowNet3D/tree/main/images/Pipeline.jpg)
 
# Prerequisities
Our models is trained and tested under:
- Python 3.6.9
- NVIDIAGPU+CUDA CuDNN
- PyTorch(torch==1.7.0)
- Minkowski Engine(MinkowskiEngine==0.5.4)
- tqdm
- numpy

Compile the furthest point sampling, grouping and gathering operation for PyTorch. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).

```
cd FPT_utils\cuda_ops
python setup.py install
cd ../../
```

# Data preparation
Please follow the instructions given in [HPLFlowNet](https://github.com/laoreja/HPLFlowNet) to construct the FlyingThings3D and KITTI datasets. Finally the directory structure should like this:
```
USTD-FlowNet3D
├── data
│   └── HPLFlowNet
│       ├── FlyingThings3D_subset_processed_35m
│       └── KITTI_processed_occ_final
├── datasets
├── images
├── FPT_utils
├── models
├── README.md
├── train.py
├── train_ddp.py
├── train_multi_process_GPUs.py
└── val_test.py
```

# Training
By default, one ```checkpoints``` directory will be created automatically. The tensorboard's logs and the checkpoints will be saved in this directory.
- To train USTD on FT3D using 8192 points:
```
python -m torch.distributed.launch --nproc_per_node=4 ./USTD-FlowNet3D/train_ddp.py --nb_iter 1 --dataset HPLFlowNet --nb_points 8192 --batch_size 4 --nb_epochs 100
```

# Testing
One model trained on FT3D using 8192 points is provided in ```checkpoints/model-100.tar```.
- To evalaute one pretrained model on test set of FT3D:
```
python val_test.py --dataset HPLFlowNet_FT3D --nb_points 8192 --path2ckpt checkpoints/model-100.tar
```
- To evalaute one pretrained model on KITTI dataset:
```
python val_test.py --dataset HPLFlowNet_kitti  --nb_points 8192 --path2ckpt checkpoints/model-100.tar
```

# Acknowledgments
-[FLOT](https://github.com/valeoai/FLOT)
-[HPLFlowNet](https://github.com/laoreja/HPLFlowNet)
-[SCTN](https://github.com/hi-zhengcheng/sctn)
-[FPT](https://github.com/POSTECH-CVLab/FastPointTransformer)
