# CVPR2016_MIL
对论文 "Patch-Based Convolutional Neural Network for Whole Slide Tissue Image Classification" 的复现

## 目录
- [描述](#描述)
- [使用方法](#使用方法)
- [许可证](#许可证)

## 描述
原论文链接： [Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf)  
这是原文用 E-M 范式来训练卷积神经网络的部分的复现。

该部分算法简述如下：  
1. 将初始的关键 patch 定义为所有 patch
2. M-step：在关键 patch 上训练两个 epoch
3. E-step：输出所有 patch 的预测概率，高斯平滑后得到阈值并找出关键 patch
    - 原文没有给出高斯平滑的参数，我们将方形边长取为 9，将 $\sigma$ 取为 1
    - 原文只给出了每张大图的阈值用的下分位数的范围，我们自行定义为右端点值 28%
    - 原文只给出了每个类别的阈值用的下分位数的范围，我们自行定义为右端点值 25%
4. 重复步骤 2--3 直至收敛
    - 原文没有给出收敛的标准，我们自行定义为
    - #{old_discriminative_patches ∩ new} / max⁡(#{old}, #{new}) > 0.95

## 使用方法

运行 `train_cnn.py` 之前，需要做如下微调：
1. 根据实际需要，修改 `train_cnn.py` 中的参数 `NUM_CLASSES` 的值。
2. 根据实际需要，修改 `train_cnn.py` 中的参数 `PATCHES_DIR` 的值。也就是你的 patch 图片存放的文件夹。
3. 确保你的 patch 已经命名为 `class_<k>_<id>_<i>_<j>.png` 的格式。k 是类别编号，(i,j) 代表 patch 在大图中的位置，编号方式与 numpy 的下标相同，以 patch 为单位。

训练结束后，运行 `./utils/visualize_disc_patches.py` 来可视化 discriminative patches。该程序会新建 `./visualize_disc` 文件夹并在其中生成每个大图的示意图。

## 许可证

[MIT](LICENSE) © Richard Littauer
