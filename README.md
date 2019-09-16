# ICW-GANs
This is a Tensorflow implementation of my paper:

[FMRI data augmentation via synthesis](https://arxiv.org/abs/1907.06134), The IEEE International Symposium on Biomedical Imaging (ISBI'19)

Peiye Zhuang, Alexander Schwing, and Sanmi Koyejo

![Results](https://github.com/KelestZ/ICW-GANs/blob/master/misc/generated.png)

## Prerequisites

- Tensorflow 1.x
- Python3
- NVIDIA GPU + CUDA CuDNN
- Scipy 1.1.0, Nilearn etc.

## Data

We used a public dataset [BrainPedia 1952](https://neurovault.org/collections/1952/). You may either download the dataset by yourself or use our [preprocessed data](https://drive.google.com/open?id=1nLHZsWR9XFBDIZOob5e_kHq3HV5B_37q) on GoogleDrive.

## Pretrained models

You may download our pretrained model [checkpoint](https://drive.google.com/open?id=1QTj8IOP3kw694k3gSo_HWTfScTafgO05) on GoogleDrive.

## Training & testing

```
 python icw_gans.py
```
You may change default parameter settings in the argparse. We did not write an independent python file for testing. Instead, we used the function `save_test_img` in the code to save test images after amount of training epochs.
