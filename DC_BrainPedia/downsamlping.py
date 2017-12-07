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
def downsample_brain(braindata, scaling=1.0):
    # initial mask
    mskFile = open(base_dir + 'msk_p2.pkl', 'rb')
    inmask = pkl.load(mskFile)

    print(inmask.shape)

    outbraindata = {}
    vecbraindata = {}

    # rescaling
    outshape = tuple([int(float(x) / scaling) for x in inmask.shape])
    print( outshape)
    realscale = float(inmask.shape[0]) / float(outshape[0])
    new_affine = inmask.get_affine().copy()
    new_affine[:3, :3] *= realscale

    print(realscale)
    # resample mask
    outmask = resample_img(inmask, target_affine=new_affine,
                           target_shape=outshape, interpolation='nearest')
    outmasker = NiftiMasker(mask_img=outmask, standardize=False)
    outmasker.fit()
    print("new # non-masked voxels", outmask.get_data().sum())
    outmaskFile = open(base_dir + 'outmask_' + str(scaling) + '_p2.pkl', 'wb')

    '''
    # resample image
    ##change
    count = 0
    for i in braindata:
        #print(i)
        if (i[-3:] == 'nii'):
            img_name = i.replace('.nii', '')
            img = load_img(data_dir+i)
            temp_vec = resample_img(img, target_affine=new_affine,
                                    target_shape=outshape, interpolation='continuous')
            outbraindata.setdefault(img_name)
            outbraindata[img_name] = temp_vec
            # VECTORIZED IMAGE
            #print(i)
            vecbraindata.setdefault(img_name,[])
            vecbraindata[img_name] = outmasker.transform(temp_vec)
            count += 1
        if (count % 500 == 1):
            print(count, 'imgs_p2')
    print('Finish',count, 'imgs')


    vecFile = open(base_dir + 'vec_brain_'+str(scaling)+'_p2.pkl', 'wb')
    
    outbraindataFile = open(base_dir +'outbraindata_'+str(scaling)+'_p2.pkl','wb')

    pkl.dump(vecbraindata, vecFile, protocol=2)
    pkl.dump(outbraindata, outbraindataFile,protocol=2)
    outbraindataFile.close()
    mskFile.close()
    vecFile.close()
    '''
    pkl.dump(outmask, outmaskFile, protocol=2)

    outmaskFile.close()

brain_data = os.listdir(data_dir)
downsample_brain(brain_data, 5.0)

'''
pkl_dir = '/Users/zpy/Desktop/'
import nilearn.plotting as plot
vec_file = open(pkl_dir + 'outbraindata.pkl','rb')
im = pkl.load(vec_file)
#print(vec)
print(type(im))
for i in im.keys():
    plot.plot_glass_brain(i)
    break
'''
