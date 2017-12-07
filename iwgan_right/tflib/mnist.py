# coding=utf-8
import numpy
import numpy as np
import os
import urllib
import gzip
import cPickle as pickle
import pickle as pkl
from sklearn.utils import shuffle
from six.moves import xrange

os.environ["CUDA_VISIBLE_DEVICES"]='3'

base_dir = '/home/nfs/zpy/BrainPedia/pkl/'
imageFile = base_dir+'vec_brain_5_p2.pkl'
labelFile = base_dir+'multi_class_pic_tags_p2.pkl'
def mnist_generator(images,targets, batch_size, n_labelled, limit=None):

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)


    if limit is not None:
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        '''
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)
        '''

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 364)
        target_batches = targets.reshape(-1, batch_size)

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
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )


def load_BrainPedia_test(batch_size, n_labelled=None):
    x = []
    y = []
    count = 0
    print('[INFO] Load Test BrainPedia dataset...')
    imgF = open(imageFile, 'rb')
    lbF = open(labelFile, 'rb')

    imgpkl = pkl.load(imgF)
    labelpkl = pkl.load(lbF)

    # ran = np.choice(6573, self.batch_size)


    for k in labelpkl.keys():
        if k == '35236':
            vec = imgpkl[k]
            #print('origianl:',vec)
            _max = max(vec[0])
            _min = min(vec[0])
            print('max&min',_max,_min)
            vec = np.array([2*((vec[0]-_min)/(_max-_min)) - 1])
            print(vec)

            print('[INFO] Training image is:', k)
            for m in range(6500):
                x.append(vec)
                y.append(labelpkl[k])
                count += 1
            break

    print('[INFO]', count, 'images in all')

    x, y = shuffle(np.array(x).reshape((6500, 1, 364, 1)).astype(np.float64),
                   np.array(y).reshape(6500).astype(np.int))

    '''
    y_vec = np.zeros((len(y), 45), dtype=np.float64)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0
    '''
    print('[INFO] xshape', x[0].shape)
    print('[INFO] y ', y[0])
    print('[INFO] Finish loading %d images' % count)

    imgF.close()
    lbF.close()

    return (mnist_generator(x, y, batch_size, n_labelled))