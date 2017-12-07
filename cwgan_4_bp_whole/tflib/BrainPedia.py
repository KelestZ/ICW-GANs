# coding=utf-8
import numpy
import numpy as np
import os
import urllib
import gzip
#import cPickle as pickle # pkl

import pickle as pkl
from sklearn.utils import shuffle
from six.moves import xrange
import scipy.misc
from upsampling import *

from sklearn import svm
import numpy as np
import nilearn.plotting as plot
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like, resample_img
import pickle as pkl
from nilearn.image import load_img
from matplotlib import pyplot as plt
import nibabel

import os

base_dir = '/home/nfs/zpy/BrainPedia/pkl/'
imageFile = base_dir+'outbraindata_4.0_p2.pkl'
labelFile = base_dir+'multi_class_pic_tags.pkl'

img_list = ['32015','32083','32149','32213','32282','32552','32619','32820','32887','32955']
#tmp = [str(int(i)+1) for i in img_list]
#img_list = img_list # +tmp

# classed whose number of data with [100,300], 22 classes in all

def mnist_generator(images,targets, batch_size, y_dim, n_labelled, limit=None):

    #rng_state = numpy.random.get_state()
    #numpy.random.shuffle(images)
    #numpy.random.set_state(rng_state)
    #numpy.random.shuffle(targets)

    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()

        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 2145)
        target_batches = targets.reshape(-1, batch_size, y_dim)


        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:
            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't ficnd MNIST dataset in /tmp, downloading...")
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )

def load_BrainPedia_test(batch_size, FLAGS, n_labelled=None):

    print('[INFO] Load Test BrainPedia dataset...')

    #count = {}
    imgF = open(imageFile, 'rb')
    lbF = open(labelFile, 'rb')

    imgpkl = pkl.load(imgF)
    labelpkl = pkl.load(lbF)

    record_size_per_tag = open(base_dir + 'record_size_per_tag', 'rb')
    size_records = pkl.load(record_size_per_tag)
    record_size_per_tag.close()

    outbrains = []
    dev_outbrains = []

    labels = [] # img id
    dev_labels = []
    train_data_list = []
    dev_data_list = []
    test_data_list = []
    #train_file = open('train_data_list_'+FLAGS.class_id+'.pkl','wb')
    #dev_file = open('dev_data_list_'+FLAGS.class_id+'.pkl','wb')
    #test_file = open('test_data_list_'+FLAGS.class_id+'.pkl','wb')

    train_file = open('/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/train_data_list.pkl' ,'rb')
    dev_file = open('/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/dev_data_list.pkl' ,'rb')
    trainF = pkl.load(train_file)
    devF = pkl.load(dev_file)

    train_file.close()
    train_file.close()

    for k in trainF:
        if (int(labelpkl[k]) == int(FLAGS.class_id)):
            outbrain = imgpkl[k].get_data()
            _max = np.max(outbrain)
            _min = np.min(outbrain)
            # normalization
            outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
            # append to training data list
            outbrains.append(outbrain)
            labels.append(labelpkl[k])

    for k in devF:
        if (int(labelpkl[k]) == int(FLAGS.class_id)):
            outbrain = imgpkl[k].get_data()
            _max = np.max(outbrain)
            _min = np.min(outbrain)
            # normalization
            outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
            dev_outbrains.append(outbrain)
            dev_labels.append(labelpkl[k])

    '''
    for k in imgpkl.keys():

        if(labelpkl[k] == int(FLAGS.class_id)):
            count = size_records[labelpkl[k]]
            #print('[INFO] TRAIN SIZE & DEV & TEST SIZE : ', count[0], count[1], count[2])
            # train
            if (count[0] > 0):
                outbrain = imgpkl[k].get_data()
                _max = np.max(outbrain)
                _min = np.min(outbrain)
                # normalization
                outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
                # append to training data list
                outbrains.append(outbrain)

                count[0] -= 1
                train_data_list.append(k)
                labels.append(labelpkl[k])

            elif (count[1] > 0):
                count[1] -= 1

                dev_data_list.append(k)
                outbrain = imgpkl[k].get_data()
                _max = np.max(outbrain)
                _min = np.min(outbrain)
                # normalization
                outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
                dev_outbrains.append(outbrain)
                dev_labels.append(labelpkl[k])

            elif (count[2] > 0):
                count[2] -= 1
                test_data_list.append(k)

            else:
                raise ('Number Error')
        '''

    #pkl.dump(train_data_list,train_file)
    #pkl.dump(dev_data_list, dev_file)
    #pkl.dump(test_data_list, test_file)

    #train_file.close()
    #dev_file.close()
    #test_file.close()

    print('[INFO] TRAIN SIZE & DEV & TEST SIZE : ', len(outbrains), len(dev_outbrains))
    print('[INFO] COMPARE SIZE: ', size_records[int(FLAGS.class_id)])
    count = 0
    x = []
    y = []

    # Least Common Multiply
    def lcm(x, y):
        if x > y:
            greater = x
        else:
            greater = y
        while (True):
            if ((greater % x == 0) and (greater % y == 0)):
                lcm = greater
                break
            greater += 1
        return lcm


    for m in range(lcm(len(outbrains),50)//len(outbrains)):
        for i in range(len(outbrains)):
            x.append(outbrains[i])
            y.append(labels[i])
            count += 1

    print('[INFO] Train images:', count)

    y_dim = 45
    dev_x = []
    dev_y = []
    count = 0
    for m in range(lcm(len(dev_outbrains),50)//len(dev_outbrains)):
        for i in range(len(dev_outbrains)):
            dev_x.append(dev_outbrains[i])
            dev_y.append(dev_labels[i])
            count += 1

    print('[INFO] Dev images:', count)
    x, y = shuffle(np.array(x).astype(np.float64),
                   np.array(y).reshape(-1).astype(np.int))
    dev_x, dev_y = shuffle(np.array(dev_x).astype(np.float64),
                   np.array(dev_y).reshape(-1).astype(np.int))


    # change it
    y_vec = np.zeros((len(y), y_dim), dtype=np.float64)
    dev_y_vec = np.zeros((len(dev_y), y_dim), dtype=np.float64)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    for i, label in enumerate(dev_y):
        dev_y_vec[i, dev_y[i]] = 1.0

    print('[INFO] xshape', x[0].shape)
    #print('[INFO] Finish loading %d images' % count)

    imgF.close()
    lbF.close()

    return (mnist_generator(x, y_vec, batch_size, y_dim,  n_labelled),
            mnist_generator(dev_x, dev_y_vec, batch_size, y_dim, n_labelled))


def load_mnist(batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print
        "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    x, y =train_data
    '''
    y_dim = 10
    y_vec = np.zeros((len(y), y_dim), dtype=np.float64)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0
    return (mnist_generator(x, y_vec, batch_size,y_dim, n_labelled))

    single_x = x[0]
    single_y=y[0]

    test_x = np.zeros((5000,784))
    test_y = []
    for k in range(5000):
        test_x[k] = single_x
        test_y.append(single_y)

    print('y',single_y)
    scipy.misc.imsave('training_data.png', np.reshape(single_x,[28,28]))
    '''
    y_dim = 10

    y_vec = np.zeros((len(y), y_dim), dtype=np.float64)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0




    return (mnist_generator(x, y_vec, batch_size,y_dim, n_labelled))
