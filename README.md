# adrenal_segmentation
This is the official implementation of "An optimized two-stage cascaded deep neural network for adrenal segmentation on CT images".

# Introduction
Segmentation of adrenal glands from CT images is a crucial step in the AI-assisted diagnosis of adrenal gland- related disease. However, highly intrasubject variability in shape and adhesive boundaries with surrounding tissues make accurate segmentation of the adrenal gland a challenging task. In the current study, we proposed a novel two-stage deep neural network for adrenal gland segmentation in an end-to-end fashion. In the first stage, a localization network that aims to determine the candidate volume of the target organ was used in the pre- processing step to reduce class imbalance and computational burden. Then, in the second stage, a Small- organNet model trained with a novel boundary attention focal loss was designed to refine the boundary of the organ within the screened volume. The experimental results show that our proposed cascaded framework out- performs the state-of-the-art deep learning method in segmenting the adrenal gland with respect to accuracy; it requires fewer trainable parameters and imposes a smaller demand on computational resources.

![figure 2](https://user-images.githubusercontent.com/88765550/129294648-a782979c-05cb-440e-955a-6f432eb34b7f.png)

[paper](https://www.sciencedirect.com/science/article/abs/pii/S0010482521005436?via%3Dihub)


# Dependencies
- torch==1.6.0
- torchvision==0.7.0
- monai==0.3.0

# How to test
We provided two test data in `./data/adrenal_deomo`

To test them run:
```
python joint_seg.py
```

# Citation
Luo, G., Yang, Q., Chen, T., Zheng, T., Xie, W. and Sun, H., 2021. An optimized two-stage cascaded deep neural network for adrenal segmentation on CT images. Computers in Biology and Medicine, p.104749.
