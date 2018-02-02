# coding=utf-8

# herehere
import os, sys
sys.path.append(os.getcwd())
from sklearn.utils import shuffle
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np

from sklearn.mixture import GMM
import numpy as np
from sklearn.naive_bayes import GaussianNB

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

    #true_real or cross.pkl # _more//// _200
    train_file = open('/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/for_nn_classify_4.0_more/true_real.pkl', 'rb')
    dev_file = open('/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/for_nn_classify_4.0_more/dev_dic.pkl', 'rb')
    test_file = open('/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/for_nn_classify_4.0_more/test_dic.pkl', 'rb')

    train_dic = pkl.load(train_file)
    dev_dic = pkl.load(dev_file)
    test_dic = pkl.load(test_file)

    ct_train = 0
    ct_test = 0
    ct_dev = 0

    for k in train_dic.keys():
        # generated data
        train_brain += train_dic[k]
        train_label += [k] * len(train_dic[k])
        ct_train += len(train_dic[k])

        #dev
        dev_brain += dev_dic[k]
        dev_label += [k]*len(dev_dic[k])
        ct_dev += len(dev_dic[k])

        #test
        test_brain += test_dic[k]
        test_label += [k]*len(test_dic[k])
        ct_test += len(test_dic[k])

    train_file.close()
    dev_file.close()
    test_file.close()
    print('[INFO] TRAIN SIZE & TEST & DEV SIZE : ', ct_train, ct_test, ct_dev)
    train_brain, train_label = shuffle(np.array(train_brain).astype(np.float64),np.array(train_label).reshape(-1).astype(np.int))

    test_brain = np.array(test_brain).astype(np.float64)
    test_label = np.array(test_label).reshape(-1).astype(np.int)
    return(train_brain,train_label,test_brain,test_label)

X, Y, test_x, test_y = input_data()

clf = GaussianNB()
clf.fit(X, Y)
preds = clf.predict(test_x)
print('done')
