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

Compile the furthest point sampling, grouping and gathering operation for PyTorch. We use the operation from this ![repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).
