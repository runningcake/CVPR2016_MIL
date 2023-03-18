# CVPR2016_MIL
An implementation of "Patch-Based Convolutional Neural Network for Whole Slide Tissue Image Classification"

## Table of Contents

- [Description](#description)
- [Usage](#usage)
- [License](#license)

## Description
You can refer to the original paper: [Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf)  
This is an implementation for the part that trains a convolutional neural network in an EM fashion.

这部分算法简述如下：  
1. 将初始的关键 patch 定义为所有 patch
2. M-step：在关键 patch 上训练两个 epoch
3. E-step：输出所有 patch 的预测概率，高斯平滑后得到阈值并找出关键 patch
    - 原文没有给出高斯平滑的参数，我们将方形边长取为 9，将 $\sigma$ 取为 1
    - 原文没有给出阈值用的是多少分位的分位数，我们自行定义为 90%
    - 原文说每类的类别阈值可以自定义，但没给数值，我们也自行定义为 90%
4. 重复步骤 2--3 直至收敛
    - 原文没有给出收敛的标准，我们自行定义为
    - #{old_discriminative_patches ∩ new} / max⁡(#{old}, #{new}) > 0.95

## Usage
Before you run `train_cnn.py`, the following needs to be done.
1. Edit the value of `NUM_CLASSES` in `train_cnn.py` according to your needs.
2. Edit the value of `PATCHES_DIR` in `train_cnn.py`, which is the directory of your patches, according to your needs.
3. Make sure your patches are named as `class_<k>_<id>_<i>_<j>.png` where k is for class and (i,j) is the patch's index in numpy-style.

## License

[MIT](LICENSE) © Richard Littauer
