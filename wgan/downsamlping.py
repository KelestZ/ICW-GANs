# Downsampling!
import os
import numpy as np

base_dir = '/home/nfs/zpy/BrainPedia/pkl/'
nv = 1000
frequency_threshold = 0.001
from nilearn.image import new_img_like, resample_img
from nilearn.input_data import NiftiMasker
from nilearn import datasets
from sklearn.preprocessing import normalize
import pickle as pkl
from nilearn.image import load_img

data_dir = '/data/zpy/BrainPedia/'
def downsample_brain(brain_data, scaling=1.0):
    # initial mask
    mskFile = open(base_dir + 'msk.pkl', 'rb')
    inmask = pkl.load(mskFile)

    print(inmask.shape)

    outbraindata = {}
    # vecbraindata = {}

    # rescaling
    outshape = tuple([int(float(x) / scaling) for x in inmask.shape])
    print("new data shape is", outshape)
    realscale = float(inmask.shape[0]) / float(outshape[0])
    new_affine = inmask.get_affine().copy()
    new_affine[:3, :3] *= realscale

    # resample mask
    outmask = resample_img(inmask, target_affine=new_affine,
                           target_shape=outshape, interpolation='nearest')
    outmasker = NiftiMasker(mask_img=outmask, standardize=False)
    outmasker.fit()
    print("new # non-masked voxels", outmask.get_data().sum())

    # resample image
    ##change
    count = 0
    for i in brain_data:
        #print(i)
        if (i[-3:] == 'nii'):
            img_name = i.replace('.nii', '')
            img = load_img(data_dir+i)

            #small brain
            temp_vec = resample_img(img, target_affine=new_affine,
                                    target_shape=outshape, interpolation='continuous')
            outbraindata.setdefault(img_name)
            outbraindata[img_name] = temp_vec
            #print('small brain shape', temp_vec.get_shape)

            # VECTORIZED IMAGE
            #print(i)
            #vecbraindata.setdefault(img_name,[])
            #vecbraindata[img_name] = outmasker.transform(temp_vec)
            count += 1
        if (count % 500 == 1):
            print(count, 'imgs')
    print('Finish',count, 'imgs')

    #vecFile = open(base_dir + 'vec_brain.pkl', 'wb')
    #outmaskFile = open(base_dir + 'outmask.pkl', 'wb')

    outbraindataFile = open(base_dir + 'outbraindata_' + str(scaling) + '.pkl','wb')

    #pkl.dump(vecbraindata, vecFile)
    #pkl.dump(outmasker, outmaskFile)

    pkl.dump(outbraindata, outbraindataFile)
    outbraindataFile.close()

    #mskFile.close()
    #vecFile.close()
    #outmaskFile.close()

brain_data = os.listdir(data_dir)
downsample_brain(brain_data, 2.0)


pkl_dir = '/Users/zpy/Desktop/'
import nilearn.plotting as plot
vec_file = open(pkl_dir + 'outbraindata_2.0.pkl', 'rb')
im = pkl.load(vec_file)
#print(vec)
print(type(im))
for i in im.keys():
    plot.plot_glass_brain(i)
    break
