#-*- coding=utf-8 -*-
import sys
from sklearn.utils import shuffle
from six.moves import xrange
import numpy as np
import importlib
importlib.reload(sys)
import random
import pickle as pkl

def data_generator(images, targets, batch_size, y_dim, n_labelled, shape, limit=None):
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        print('should be here')
        labelled = np.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

        if n_labelled is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labelled)
        a = images.transpose([0, 3, 1, 2])

        first = a.shape[0] // batch_size
        second = a.shape[0] % batch_size

        image_batches = (a[:first * batch_size, :, :, :]).reshape(-1, batch_size, shape[0] * shape[1] * shape[2])
        target_batches = (targets[:first * batch_size, :]).reshape(-1, batch_size, y_dim)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]), np.copy(labelled))

        else:
            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]))

    return get_epoch

def batch_generator_for_classify(images, targets, batch_size, y_dim, n_labelled, shape, limit=None):
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = np.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

        if n_labelled is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labelled)
        a = images.transpose([0, 3, 1, 2])

        first = a.shape[0] // batch_size
        second = a.shape[0] % batch_size

        image_batches = (a[:first * batch_size, :, :, :]).reshape(-1, batch_size, shape[0] * shape[1] * shape[2])
        target_batches = (targets[:first * batch_size, :]).reshape(-1, batch_size, y_dim)

        if (second != 0):
            second_batches = (a[-second:, :, :, :]).reshape(-1, shape[0] * shape[1] * shape[2])  # shape[0]=1
            second_targets = targets[-second:, :]  # .reshape(-1, y_dim)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)
            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]), np.copy(labelled))

            if(second != 0):
                yield (np.copy(second_batches), np.copy(second_targets), np.copy(labelled))
        else:
            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]))

            if (second != 0):
                yield (np.copy(second_batches), np.copy(second_targets))

    return get_epoch

def trans_to_onehot(labels, y_dim):
    y_vec = np.zeros((len(labels), y_dim), dtype=np.float64)
    for i, label in enumerate(labels):
        y_vec[i, labels[i]] = 1.0
    return y_vec

def load_data(batch_size, base_dir, imageFile, labelFile, n_labelled=None, tags_leave_out=None, y_dim=None):
    print('[INFO] Load Test BrainPedia dataset...')

    imgF = open(base_dir + imageFile, 'rb')
    lbF = open(base_dir + labelFile, 'rb')

    imgpkl = pkl.load(imgF)
    labelpkl = pkl.load(lbF)
    # split data before running the code
    record_size_per_tag = open(base_dir + 'record_size_per_tag', 'rb')
    size_records = pkl.load(record_size_per_tag)
    record_size_per_tag.close()
    outbrains = []
    dev_outbrains = []

    test_outbrains = []
    test_labels = []
    labels = []
    dev_labels = []
    ct = 0  # this place needs fix if use 22 classes
    train_data_list = []
    dev_data_list = []
    test_data_list = []
    train_file = open('./train_data_list.pkl', 'wb')
    dev_file = open('./dev_data_list.pkl', 'wb')
    test_file = open('./test_data_list.pkl', 'wb')

    sum_0 = 0
    sum_1 = 0
    sum_2 = 0
    for m in size_records.keys():
        if (size_records[m] != []):
            sum_0 += size_records[m][0]
            sum_1 += size_records[m][1]
            sum_2 += size_records[m][2]

    key_list = shuffle(list(imgpkl.keys()))
    count = {}

    for k in key_list:
        if (labelpkl[k] in tags_leave_out):
            # <30
            continue
        else:
            if (labelpkl[k] not in count.keys()):
                count[labelpkl[k]] = size_records[labelpkl[k]]
            sum_ = sum(count[labelpkl[k]])
            if (sum_ > 0):
                if (count[labelpkl[k]][0] > 0):
                    outbrain = imgpkl[k].get_data()
                    _max = np.max(outbrain)
                    _min = np.min(outbrain)
                    # normalization
                    # outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
                    outbrain = np.array(((outbrain - _min) / (_max - _min)))
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
                    # outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
                    outbrain = np.array(((outbrain - _min) / (_max - _min)))
                    dev_outbrains.append(outbrain)
                    dev_labels.append(labelpkl[k])

                    outbrains.append(outbrain)
                    train_data_list.append(k)
                    labels.append(labelpkl[k])

                elif (count[labelpkl[k]][2] > 0):
                    count[labelpkl[k]][2] -= 1

                    test_data_list.append(k)

                    outbrain = imgpkl[k].get_data()
                    _max = np.max(outbrain)
                    _min = np.min(outbrain)
                    # normalization
                    # outbrain = np.array([2 * ((outbrain - _min) / (_max - _min)) - 1])
                    outbrain = np.array(((outbrain - _min) / (_max - _min)))
                    test_outbrains.append(outbrain)
                    test_labels.append(labelpkl[k])

                else:
                    raise ('Number Error')

    pkl.dump(train_data_list, train_file)
    pkl.dump(dev_data_list, dev_file)
    pkl.dump(test_data_list, test_file)
    train_file.close()
    dev_file.close()
    test_file.close()
    # 4390 in all
    print('[INFO] TRAIN SIZE & DEV & TEST SIZE : ', len(train_data_list), len(dev_data_list), len(test_data_list))
    x = []
    y = []
    # 4390
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

    for m in range(lcm(len(outbrains), batch_size) // len(outbrains)):
        for i in range(len(outbrains)):
            x.append(outbrains[i])
            y.append(labels[i])
            ct += 1
    shape = outbrains[0].shape
    dev_x = []
    dev_y = []
    ct = 0

    for m in range(lcm(len(dev_outbrains), batch_size) // len(dev_outbrains)):
        for i in range(len(dev_outbrains)):
            dev_x.append(dev_outbrains[i])
            dev_y.append(dev_labels[i])
            ct += 1
    x, y = np.array(x).astype(np.float64), np.array(y).reshape(-1).astype(np.int)
    temp_x = dev_x[:2]
    temp_y = dev_y[:2]

    dev_x, dev_y = np.array(dev_x).astype(np.float64), np.array(dev_y).reshape(-1).astype(np.int)

    test_outbrains = test_outbrains + temp_x
    test_labels = test_labels + temp_y

    test_x = np.array(test_outbrains).astype(np.float64)
    test_y = np.array(test_labels).reshape(-1).astype(np.int)

    y_vec = np.zeros((len(y), y_dim), dtype=np.float64)
    dev_y_vec = np.zeros((len(dev_y), y_dim), dtype=np.float64)
    test_y_vec = np.zeros((len(test_labels), y_dim), dtype=np.float64)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    for i, label in enumerate(dev_y):
        dev_y_vec[i, dev_y[i]] = 1.0

    for i, label in enumerate(test_y):
        test_y_vec[i, test_labels[i]] = 1.0

    imgF.close()
    lbF.close()

    return (data_generator(x, y_vec, batch_size, y_dim, n_labelled, shape=shape),
            data_generator(dev_x, dev_y_vec, batch_size, y_dim, n_labelled, shape=shape),
            data_generator(test_x, test_y_vec, batch_size, y_dim, n_labelled, shape=shape))
