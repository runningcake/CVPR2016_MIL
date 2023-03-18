# CVPR2016_MIL
An implementation of "Patch-Based Convolutional Neural Network for Whole Slide Tissue Image Classification"  
[中文版 Chinese version](README.zh-CN.md)
## Table of Contents

- [Description](#description)
- [Usage](#usage)
- [License](#license)

## Description
You can refer to the original paper: [Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf)  
This is an implementation for the part that trains a convolutional neural network in an EM fashion.

The key steps are briefly described as follows.  
1. Initialize all patches as discriminative.
2. M-step: Train 2 epochs on discriminative patches.
3. E-step: Output the predicted probabilities of all patches, and apply Gaussian smoothing and thresholding to find discriminative patches for next round.
    - The paper did not give the paramters of Gaussian smoothing. We set the side length of the square as 9, and set sigma as 1.
    - We set the percentile pairs for WSI-level and class-level threholds as the right endpoints given in the paper, which are 0.28 and 0.25.
4. Repeat steps 2--3 till convergence.
    - The paper did not give the standard for convergence. We define it as the following.
    - #{old_discriminative_patches ∩ new} / max⁡(#{old}, #{new}) > 0.95

## Usage

Before you run `train_cnn.py`, the following needs to be done.
1. Edit the value of `NUM_CLASSES` in `train_cnn.py` according to your needs.
2. Edit the value of `PATCHES_DIR` in `train_cnn.py`, which is the directory of your patches, according to your needs.
3. Make sure your patches are named as `class_<k>_<id>_<i>_<j>.png` where k is for class and (i,j) is the patch's index in numpy-style (one patch as one unit).

After the training is finished, run `./utils/visualize_disc_patches.py` to visualize the discriminative patches. Images indicating the results will be generated in a newly created directory `./visualize_disc`.

## License

[MIT](LICENSE) © Richard Littauer
