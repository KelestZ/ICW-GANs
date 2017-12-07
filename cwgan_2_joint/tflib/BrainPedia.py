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
from tflib.upsampling import *
from sklearn.model_selection import StratifiedKFold

base_dir = '/home/nfs/zpy/BrainPedia/pkl/'
imageFile = base_dir+'outbraindata_2.0.pkl'
labelFile = base_dir+'multi_class_pic_tags.pkl'

# classed whose number of data with [100,300], 22 classes in all
tags_leave_out = [36, 37, 34, 44, 38, 28, 23, 33, 32]

def mnist_generator(images, targets, batch_size, y_dim, n_labelled, limit = None):

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)

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

        image_batches = images.reshape(-1, batch_size, 18538)
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

def load_BrainPedia_test(batch_size, list_dir, n_labelled=None):

    print('[INFO] Load Test BrainPedia dataset...')


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
    ct = 0    # this place needs fix if use 22 classes
    train_data_list = []
    dev_data_list = []
    test_data_list = []
    train_file = open(list_dir+'/train_data_list.pkl', 'wb')
    dev_file = open(list_dir+'/dev_data_list.pkl', 'wb')
    test_file = open(list_dir+'/test_data_list.pkl', 'wb')

    sum_0 = 0
    sum_1 = 0
    sum_2 = 0
    for m in size_records.keys():
        if(size_records[m]!=[]):
            sum_0 += size_records[m][0]
            sum_1 += size_records[m][1]
            sum_2 += size_records[m][2]
    print('[INFO] TRAIN SIZE & DEV & TEST SIZE : ', sum_0, sum_1, sum_2)

    count = {}
    # key_list = shuffle(list(imgpkl.keys()))

    for k in imgpkl.keys():
        if(labelpkl[k] in tags_leave_out):
            # <30
            continue
        else:
            if(labelpkl[k]  not in count.keys()):
                count[labelpkl[k]] = size_records[labelpkl[k]]
            sum_ = sum(count[labelpkl[k]])
            if(sum_>0):
                if(count[labelpkl[k]][0]>0):
                    outbrain = imgpkl[k].get_data()
                    _max = np.max(outbrain)
                    _min = np.min(outbrain)
                    # normalization
                    outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
                    # append to training data list
                    outbrains.append(outbrain)

                    count[labelpkl[k]][0] -= 1
                    train_data_list.append(k)
                    labels.append(labelpkl[k])

                elif (count[labelpkl[k]][1] > 0):
                    count[labelpkl[k]][1] -= 1

                    dev_data_list.append(k)
                    outbrain = imgpkl[k].get_data()
                    _max = np.max(outbrain)
                    _min = np.min(outbrain)
                    # normalization
                    outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
                    dev_outbrains.append(outbrain)
                    dev_labels.append(labelpkl[k])

                elif (count[labelpkl[k]][2]> 0):
                    count[labelpkl[k]][2] -= 1

                    test_data_list.append(k)

                else:
                    raise('Number Error')

    pkl.dump(train_data_list, train_file)
    pkl.dump(dev_data_list, dev_file)
    pkl.dump(test_data_list, test_file)

    train_file.close()
    dev_file.close()
    test_file.close()

    print('save over !!!!')
    # 4390 in all
    print('[INFO] TRAIN SIZE & DEV & TEST SIZE : ', len(train_data_list), len(dev_data_list), len(test_data_list))

    x = []
    y = []

    #4390
    ct = 0

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

    for m in range(lcm(len(outbrains), 50) // len(outbrains)):
        for i in range(len(outbrains)):
            x.append(outbrains[i])
            y.append(labels[i])
            ct += 1
    # 21950
    print('[INFO] Training images: ', ct)

    y_dim = 45
    dev_x = []
    dev_y = []
    ct = 0

    for m in range(lcm(len(dev_outbrains),50)//len(dev_outbrains)):
        for i in range(len(dev_outbrains)):
            dev_x.append(dev_outbrains[i])
            dev_y.append(dev_labels[i])
            ct += 1

    print('[INFO] Dev images:', ct)

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

def load_ckpt_data(batch_size,list_dir,n_labelled=None):

    print('[INFO] Load Test BrainPedia dataset...')


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
    ct = 0    # this place needs fix if use 22 classes
    train_data_list = []
    dev_data_list = []
    test_data_list = []

    train_brain = []
    dev_brain = []
    test_brain = []

    test_label = []
    train_label = []
    dev_label = []

    train_file = open('/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/true_real.pkl', 'rb')
    dev_file = open('/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/dev_dic.pkl', 'rb')
    test_file = open('/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/test_dic.pkl', 'rb')

    train_dic = pkl.load(train_file)
    dev_dic = pkl.load(dev_file)
    test_dic = pkl.load(test_file)

    ct_train = 0
    ct_test = 0
    ct_dev = 0

    for k in train_dic.keys():

        # generated data
        for i in train_dic[k]:
            train_brain.append(i)

        train_label += [k] * len(train_dic[k])
        ct_train += len(train_dic[k])

        # dev
        dev_brain += dev_dic[k]
        dev_label += [k] * len(dev_dic[k])
        ct_dev += len(dev_dic[k])

        # test
        test_brain += test_dic[k]
        test_label += [k] * len(test_dic[k])
        ct_test += len(test_dic[k])

    train_file.close()
    dev_file.close()
    test_file.close()
    print('[INFO] TRAIN SIZE & TEST & DEV SIZE : ', ct_train, ct_test, ct_dev)

    skf = StratifiedKFold(n_splits=3)
    # skf.get_n_splits(test_brain, test_label)

    split_temp = []

    for train_, test_ in skf.split(test_brain, test_label):
        split_temp.append((train_, test_))

    train_0 = split_temp[0][0][-28:]
    test_0 = split_temp[0][1][-28:]

    train_1 = split_temp[1][0][-13:]
    test_1 = split_temp[1][1][-13:]

    # print('[INFO] 0 and 1: ',len(train_0),len(train_1))
    train_add = np.concatenate([train_0, train_1], 0)
    test_add = np.concatenate([test_0, test_1], 0)

    c = 0
    i = 0

    for train_, test_ in split_temp:
        if (i == c):
            print('[INFO] Fold:', c)

            x_tr, x_te = np.array(test_brain)[train_], np.array(test_brain)[test_]
            y_tr, y_te = np.array(test_label)[train_], np.array(test_label)[test_]
            print('ORIGINAL LEN TE:', len(x_te))

            temp_tr_x = np.concatenate([np.array(train_brain), x_tr], 0)
            temp_tr_y = np.concatenate([np.array(train_label), y_tr], 0)

            if (c == 2):
                add_x = np.array(test_brain)[train_add]
                add_y = np.array(test_label)[test_add]

                x_te = np.concatenate([x_te, add_x], 0)
                y_te = np.concatenate([y_te, add_y], 0)

                m = len(y_te) % 50
                # print("[INFO]m:", m)
                x_te = np.concatenate([x_te, np.array(dev_brain[:50 - m])], 0)
                y_te = np.concatenate([y_te, np.array(dev_label[:50 - m])], 0)

                # print('x,y te:', len(x_te), len(y_te))

            break
        else:
            i += 1

    ct = 0

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

    x = []
    y = []

    for m in range(lcm(len(temp_tr_x), batch_size) // len(temp_tr_x)):
        for i in range(len(temp_tr_x)):
            x.append(temp_tr_x[i])
            y.append(temp_tr_y[i])
            ct += 1
        '''
            if (ct == 4500):
                break
        if (ct == 4500):
            break
        '''

    # 21950
    print('[INFO] Training images: ', ct)
    y_dim = 45
    dev_x = []
    dev_y = []
    ct = 0

    for m in range(lcm(ct_dev, batch_size) // ct_dev):
        for i in range(ct_dev):
            dev_x.append(dev_brain[i])
            dev_y.append(dev_label[i])
            ct += 1

    print('[INFO] Dev images:', ct)

    x, y = shuffle(np.array(x).astype(np.float64),
                   np.array(y).reshape(-1).astype(np.int))

    dev_x, dev_y = shuffle(np.array(dev_x).astype(np.float64),
                           np.array(dev_y).reshape(-1).astype(np.int))

    m = len(y_te) % 50

    print("[INFO] Test images:", len(x_te))
    if (m != 0):
        test_outbrains = x_te[:-m]
        test_labels = y_te[:-m]
        print('[INFO]TEST cut off:', m)

    else:
        print('[INFO]TEST not cut off')
        test_outbrains = x_te
        test_labels = y_te

    print('[INFO] TEST images:', len(test_labels))
    test_x = np.array(test_outbrains).astype(np.float64)
    test_y = np.array(test_labels).reshape(-1).astype(np.int)

    # change it
    y_vec = np.zeros((len(y), y_dim), dtype=np.float64)
    dev_y_vec = np.zeros((len(dev_y), y_dim), dtype=np.float64)
    test_y_vec = np.zeros((len(test_labels), y_dim), dtype=np.float64)

    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    for i, label in enumerate(dev_y):
        dev_y_vec[i, dev_y[i]] = 1.0

    for i, label in enumerate(test_y):
        test_y_vec[i, test_labels[i]] = 1.0

    return (mnist_generator(x, y_vec, batch_size, y_dim, n_labelled),
            mnist_generator(dev_x, dev_y_vec, batch_size, y_dim, n_labelled),
            mnist_generator(test_x, test_y_vec, batch_size, y_dim, n_labelled))


    '''
    ct = 0

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

    x = []
    y = []

    for m in range(lcm(len(train_brain), batch_size) // len(train_brain)):
        for i in range(len(train_brain)):
            x.append(train_brain[i])
            y.append(train_label[i])
            ct += 1
    # 21950
    print('[INFO] Training images: ', ct)
    y_dim = 45
    dev_x = []
    dev_y = []
    ct = 0

    for m in range(lcm(ct_dev, batch_size) // ct_dev):
        for i in range(ct_dev):
            dev_x.append(dev_brain[i])
            dev_y.append(dev_label[i])
            ct += 1

    print('[INFO] Dev images:', ct)

    m = len(test_brain) % 50
    test_outbrains = np.concatenate([test_brain, np.array(dev_x[:50 - m])], 0)  # + temp_x
    test_labels = np.concatenate([test_label, np.array(dev_y[:50 - m])], 0)  # test_label #+ temp_y

    x, y = shuffle(np.array(x).astype(np.float64),
                   np.array(y).reshape(-1).astype(np.int))
    dev_x, dev_y = shuffle(np.array(dev_x).astype(np.float64),
                           np.array(dev_y).reshape(-1).astype(np.int))

    print('[INFO] TEST images:', len(test_labels))

    test_x = np.array(test_outbrains).astype(np.float64)
    test_y = np.array(test_labels).reshape(-1).astype(np.int)

    # change it
    y_vec = np.zeros((len(y), y_dim), dtype=np.float64)
    dev_y_vec = np.zeros((len(dev_y), y_dim), dtype=np.float64)
    test_y_vec = np.zeros((len(test_labels), y_dim), dtype=np.float64)

    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    for i, label in enumerate(dev_y):
        dev_y_vec[i, dev_y[i]] = 1.0

    for i, label in enumerate(test_y):
        test_y_vec[i, test_labels[i]] = 1.0

    return (mnist_generator(x, y_vec, batch_size, y_dim, n_labelled),
            mnist_generator(dev_x, dev_y_vec, batch_size, y_dim, n_labelled),
            mnist_generator(test_x, test_y_vec, batch_size, y_dim, n_labelled))
    '''
def load_cross(batch_size, n_labelled=None):
    print('[INFO] Load Test BrainPedia dataset...')

    train_brain = []
    dev_brain = []
    test_brain = []

    test_label = []
    train_label = []
    dev_label = []

    #true_real or cross.pkl  gen_dic.pkl # _more//// _200
    train_file = open('/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/gen_dic.pkl', 'rb')
    dev_file = open('/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/dev_dic.pkl', 'rb')
    test_file = open('/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/test_dic.pkl', 'rb')

    train_dic = pkl.load(train_file)
    dev_dic = pkl.load(dev_file)
    test_dic = pkl.load(test_file)

    ct_train = 0
    ct_test = 0
    ct_dev = 0

    ct_test_dic={}
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
        ct_test_dic[k]=[]
        ct_test_dic[k] =len(test_dic[k])

    train_file.close()
    dev_file.close()
    test_file.close()
    print('[INFO] TRAIN SIZE & TEST & DEV SIZE : ', ct_train, ct_test, ct_dev)
    #print(ct_test_dic)

    skf = StratifiedKFold(n_splits=3)
   # skf.get_n_splits(test_brain, test_label)

    split_temp = []

    for train_, test_ in skf.split(test_brain, test_label):
            split_temp.append((train_, test_))

    train_0 = split_temp[0][0][-28:]
    test_0 = split_temp[0][1][-28:]

    train_1 = split_temp[1][0][-13:]
    test_1 = split_temp[1][1][-13:]

    #print('[INFO] 0 and 1: ',len(train_0),len(train_1))
    train_add =  np.concatenate([train_0, train_1], 0)
    test_add = np.concatenate([test_0, test_1], 0)

    c = 0
    i = 0

    for train_, test_ in split_temp:
        if(i==c):
            print('[INFO] Fold:',c)

            x_tr, x_te = np.array(test_brain)[train_], np.array(test_brain)[test_]
            y_tr, y_te = np.array(test_label)[train_], np.array(test_label)[test_]
            print('ORIGINAL LEN TE:',len(x_te))

            temp_tr_x = np.concatenate([np.array(train_brain), x_tr], 0)
            temp_tr_y = np.concatenate([np.array(train_label), y_tr], 0)

            if(c == 2):

                add_x = np.array(test_brain)[train_add]
                add_y = np.array(test_label)[test_add]

                x_te = np.concatenate([x_te, add_x], 0)
                y_te = np.concatenate([y_te, add_y], 0)

                m = len(y_te) % 50
                #print("[INFO]m:", m)
                x_te = np.concatenate([x_te, np.array(dev_brain[:50-m])], 0)
                y_te =  np.concatenate([y_te, np.array(dev_label[:50-m])], 0)

                #print('x,y te:', len(x_te), len(y_te))

            break
        else:
            i += 1

    ct = 0

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

    x = []
    y = []

    for m in range(lcm(len(temp_tr_x), batch_size) // len(temp_tr_x)):
        for i in range(len(temp_tr_x)):
            x.append(temp_tr_x[i])
            y.append(temp_tr_y[i])
            ct += 1
            if(ct ==20000):
                break
        if (ct == 20000):
            break

    # 21950
    print('[INFO] Training images: ', ct)
    y_dim = 45
    dev_x = []
    dev_y = []
    ct = 0

    for m in range(lcm(ct_dev, batch_size) // ct_dev):
        for i in range(ct_dev):
            dev_x.append(dev_brain[i])
            dev_y.append(dev_label[i])
            ct += 1

    print('[INFO] Dev images:', ct)


    x, y = shuffle(np.array(x).astype(np.float64),
                   np.array(y).reshape(-1).astype(np.int))

    dev_x, dev_y = shuffle(np.array(dev_x).astype(np.float64),
                           np.array(dev_y).reshape(-1).astype(np.int))

    m = len(y_te) % 50

    print("[INFO] Test images:", len(x_te))
    if(m!=0):
        test_outbrains = x_te[:-m]
        test_labels = y_te[:-m]
        print('[INFO]TEST cut off:', m)

    else:
        print('[INFO]TEST not cut off')
        test_outbrains = x_te
        test_labels = y_te


    print('[INFO] TEST images:', len(test_labels))
    test_x = np.array(test_outbrains).astype(np.float64)
    test_y = np.array(test_labels).reshape(-1).astype(np.int)

    # change it
    y_vec = np.zeros((len(y), y_dim), dtype=np.float64)
    dev_y_vec = np.zeros((len(dev_y), y_dim), dtype=np.float64)
    test_y_vec = np.zeros((len(test_labels), y_dim), dtype=np.float64)

    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    for i, label in enumerate(dev_y):
        dev_y_vec[i, dev_y[i]] = 1.0

    for i, label in enumerate(test_y):
        test_y_vec[i, test_labels[i]] = 1.0

    return (mnist_generator(x, y_vec, batch_size, y_dim, n_labelled),
            mnist_generator(dev_x, dev_y_vec, batch_size, y_dim, n_labelled),
            mnist_generator(test_x, test_y_vec, batch_size, y_dim, n_labelled))

