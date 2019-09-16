import os, sys

sys.path.append(os.getcwd())
from sklearn.utils import shuffle
import time
import matplotlib
sys.path.append('/home/nfs/zpy/nilearn/')
matplotlib.use('Agg')
import numpy as np

from sklearn.mixture import GMM
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.save_images
import tflib.BrainPedia
import tflib.plot
import tflib.ops.conv3d
from six.moves import xrange
import nibabel
import nilearn.masking as masking
import tflib.upsampling
import pickle as pkl
from tflib.upsampling import *


def input_data():
    print('[INFO] Load Test BrainPedia dataset...')
    train_brain = []
    dev_brain = []
    test_brain = []

    test_label = []
    train_label = []
    dev_label = []

    # true_real or cross.pkl # _more//// _200
    train_file = open('/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/for_nn_classify_2.0_100/true_real.pkl', 'rb')
    dev_file = open('/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/for_nn_classify_2.0_100/dev_dic.pkl', 'rb')
    test_file = open('/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/for_nn_classify_2.0_100/test_dic.pkl', 'rb')

    train_dic = pkl.load(train_file)
    dev_dic = pkl.load(dev_file)
    test_dic = pkl.load(test_file)

    ct_train = 0
    ct_test = 0
    ct_dev = 0

    for k in train_dic.keys():
        # generated data
        train_brain += train_dic[k]
        # train_brain += dev_dic[k]
        train_label += [k] * (len(train_dic[k]))  # +len(dev_dic[k])
        ct_train += len(train_dic[k])

        # dev
        dev_brain += dev_dic[k]
        dev_label += [k] * len(dev_dic[k])
        ct_dev += len(dev_dic[k])

        # test d
        test_brain += test_dic[k]
        test_label += [k] * len(test_dic[k])
        ct_test += len(test_dic[k])

    train_file.close()
    dev_file.close()
    test_file.close()
    print('[INFO] TRAIN SIZE & TEST & DEV SIZE : ', ct_train, ct_test, ct_dev)
    train_brain, train_label = shuffle(np.array(train_brain).reshape([len(train_brain), -1]).astype(np.float64),
                                       np.array(train_label).reshape(-1).astype(np.int))

    test_brain = np.array(test_brain).reshape([len(test_brain), -1]).astype(np.float64)
    test_label = np.array(test_label).reshape(-1).astype(np.int)
    return (train_brain, train_label, test_brain, test_label)

train_data_list = 'train_data_list.pkl'
dev_data_list = 'dev_data_list.pkl'
test_data_list = 'test_data_list.pkl'

base_dir = '/home/nfs/zpy/BrainPedia/pkl/'
imageFile = 'original_dim.pkl'
labelFile = 'multi_class_pic_tags.pkl'

test_dir = './GMM_1952_ori_results'
def load_data2( base_dir, imageFile, labelFile):
    print('[INFO] Load Test BrainPedia dataset...')

    train_brain = []
    dev_brain = []
    test_brain = []

    test_label = []
    train_label = []
    dev_label = []

    # true_real or cross.pkl # _more//// _200
    train_file = open(train_data_list, 'rb')
    dev_file = open(dev_data_list, 'rb')
    test_file = open(test_data_list, 'rb')
    train_dic = pkl.load(train_file)
    dev_dic = pkl.load(dev_file)
    test_dic = pkl.load(test_file)
    train_file.close()
    dev_file.close()
    test_file.close()
    ct_train = 0
    ct_test = 0
    ct_dev = 0

    imgF = open(base_dir + imageFile, 'rb')
    lbF = open(base_dir + labelFile, 'rb')
    imgpkl = pkl.load(imgF)
    labelpkl = pkl.load(lbF)
    imgF.close()
    lbF.close()

    for k in train_dic:
        outbrain = imgpkl[k].get_data()
        _max = np.max(outbrain)
        _min = np.min(outbrain)
        outbrain = np.array((outbrain - _min) / (_max - _min))

        train_brain.append(outbrain)
        train_label.append(labelpkl[k])
        ct_train += 1

    for k in test_dic:
        outbrain = imgpkl[k].get_data()
        _max = np.max(outbrain)
        _min = np.min(outbrain)
        outbrain = np.array(((outbrain - _min) / (_max - _min)))

        test_brain.append(outbrain)
        test_label.append(labelpkl[k])
        ct_test += 1

    for k in dev_dic:
        outbrain = imgpkl[k].get_data()
        _max = np.max(outbrain)
        _min = np.min(outbrain)
        outbrain = np.array((outbrain - _min) / (_max - _min))

        train_brain.append(outbrain)
        train_label.append(labelpkl[k])
        ct_train += 1
        ct_dev +=1

    train_brain, train_label = np.array(train_brain).reshape([len(train_brain), -1]).astype(np.float64), np.array(train_label).reshape(-1).astype(np.int)

    print('ct train', ct_train,ct_test,ct_dev)
    test_brain = np.array(test_brain).reshape([len(test_brain), -1]).astype(np.float64)
    test_label = np.array(test_label).reshape(-1).astype(np.int)

    return (train_brain, train_label, test_brain, test_label)

X, Y, test_x, test_y = load_data2(base_dir, imageFile, labelFile)

clf = GaussianNB()
print(X.shape)
clf.fit(X, Y)
# print(clf.theta_)
# print(clf.sigma_)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

pkl_dir ='/home/nfs/zpy/BrainPedia/pkl/'
import pickle as pkl
from nilearn.input_data import NiftiMasker

def have_mask_affine():
    msk_file = open(pkl_dir+'msk_p2.pkl', 'rb')
    msk =pkl.load(msk_file)
    msk_file.close()
    msker = NiftiMasker(mask_img=msk, standardize=False)
    return msk, msker
msk,msker = have_mask_affine()
msker.fit()

def save_test_img(brain_vector, label, id_):
    brain =  brain_vector.reshape([53, 63, 46])# [26, 31, 23]) # [ 13, 15, 11])
    #temp = lib.upsampling.upsample_vectorized(brain)
    img = nibabel.Nifti1Image(brain, msk.affine)

    # zero-out
    temp = msker.transform(img)
    braindata = msker.inverse_transform(temp)

    # save labels in test picture names
    filename = './{}/class_{}/GMM_ori_{}_{}.nii.gz'.format(test_dir,label, label, id_)
    if not os.path.exists(test_dir+'/class_'+str(label)+'/'):
        print(test_dir+'/class_'+str(label)+' does not exist')
        os.makedirs(test_dir+'/class_'+str(label)+'/')
    nibabel.save(braindata, filename)

a = [36, 37, 34, 44, 38, 28, 23, 33, 32]
label_list = []
for i in range(45):
    if i not in a:
        label_list.append(i)

for i in range(36):
    for k in range(150):
        a = np.random.normal(clf.theta_[i], clf.sigma_[i], [153594]) #[17732])  # [2145])
        save_test_img(a, label_list[i], k)
    print(i, 'Done')

print('Done')


# print(classification_report(tag_list, pre_list, digits=3))
# print('acc: ',accuracy_score(test_y, preds))

















