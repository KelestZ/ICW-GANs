# -*- coding: utf-8 -*-

import os
import numpy as np

from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like, resample_img
from nilearn.image import threshold_img, index_img
import nibabel
from scipy import stats

from nilearn import plotting
import matplotlib
from matplotlib import pyplot as plt
import pickle as pkl
import nilearn.masking as masking
from nilearn.image import load_img
base_dir ='/home/nfs/zpy/BrainPedia/pkl/'


def upsample_vectorized(vecdata):
    '''
    this function upsampled vectorized data to a brain image
    INPUT:
    vecdata: masked and vectorized brain data matrix of size (K, D). K is # maps, D is dimensionality
    maskdata: resampled data shape i.e. (dx, dy, dz)
    '''

    # get mask
    inmask, smallmasker ,affine = down_mask(4.0)
    outshape = tuple([13 ,15 ,11])
    # realscale = float(inmask.shape[0]) / float(outshape[0])
    img = nibabel.Nifti1Image(vecdata, affine)

    #print(new_affine,affine)
    temp_vec = resample_img(img, target_affine=affine, target_shape=outshape, interpolation='continuous')
    # Upsample to brain space
    braindata = upsample_voxels(inmask, temp_vec)

    return braindata


# 在前面那个函数中调用了
def upsample_voxels(masker,voxeldata):
    '''
    this function upsamples brain data to a full brain image
    INPUT:
    vecdata: masked and vectorized brain data matrix of size (K, D). K is # maps, D is dimensionality
    maskdata: resampled data shape i.e. (dx, dy, dz)
    '''
    # get mask

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

def down_mask(scaling=1.0):
    data_path = '/data/zpy/BrainPedia/'

    collection = os.listdir(data_path)
    imgs = []
    count = 0
    for i in collection:
        if (str(i)[-3:] == 'nii'):
            count += 1
            imgs.append(load_img(data_path + i))
            if (count % 500 == 0 or count == 6573):  # 每隔500个一存，或者全部读完
                inmask = masking.compute_background_mask(imgs)
                break

    outshape = tuple([int(float(x) / scaling) for x in inmask.shape])

    realscale = float(inmask.shape[0]) / float(outshape[0])
    new_affine = inmask.get_affine().copy()
    new_affine[:3, :3] *= realscale

    #print(realscale)
    # resample mask
    outmask = resample_img(inmask, target_affine=new_affine,
                           target_shape=outshape, interpolation='nearest')
    outmasker = NiftiMasker(mask_img=outmask, standardize=False)
    outmasker.fit()
    return inmask, outmasker, new_affine

