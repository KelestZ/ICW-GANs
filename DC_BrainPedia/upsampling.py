# -*- coding: utf-8 -*-

import os
import numpy as np

from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like, resample_img
from nilearn.image import threshold_img, index_img
import nibabel
from scipy import stats

import seaborn as sns
from nilearn import plotting
import matplotlib
from matplotlib import pyplot as plt
import pickle as pkl
from nilearn import datasets
from six import string_types
base_dir ='/home/nfs/zpy/BrainPedia/pkl/'


def upsample_vectorized(vecdata):
    '''
    this function upsampled vectorized data to a brain image
    INPUT:
    vecdata: masked and vectorized brain data matrix of size (K, D). K is # maps, D is dimensionality
    maskdata: resampled data shape i.e. (dx, dy, dz)
    '''
    # get mask
    smallmaskFile = open(base_dir + 'outmask_5.0_p2.pkl', 'rb')
    smallmasker = pkl.load(smallmaskFile)

    # VOXEL-iZE brain data
    subbraindata = smallmasker.inverse_transform(vecdata)
    #print(sub)
    # Upsample to brain space
    braindata = upsample_voxels(subbraindata)

    return braindata


# 在前面那个函数中调用了
def upsample_voxels(voxeldata):
    '''
    this function upsamples brain data to a full brain image
    INPUT:
    vecdata: masked and vectorized brain data matrix of size (K, D). K is # maps, D is dimensionality
    maskdata: resampled data shape i.e. (dx, dy, dz)
    '''
    # get mask

    maskFile = open(base_dir + 'msk_p2.pkl', 'rb')
    masker = pkl.load(maskFile)

    # RESAMPLE to full brain space
    braindata = resample_img(voxeldata, target_affine=masker.get_affine(),
                             target_shape=masker.shape, interpolation='continuous')

    # 不太懂这里是做什么的，但是看起来没啥影响
    # zero out non-grey matter vozels in upsampled data
    zeromasker = NiftiMasker(mask_img=masker, standardize=False)
    zeromasker.fit()
    temp = zeromasker.transform(braindata)
    braindata = zeromasker.inverse_transform(temp)

    return braindata