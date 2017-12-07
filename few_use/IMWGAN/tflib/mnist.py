import numpy
import numpy as np
import scipy.misc
import os
import urllib
import gzip
import cPickle as pickle
import pickle as pkl
from sklearn.utils import shuffle


base_dir = '/home/nfs/zpy/BrainPedia/pkl/'
imageFile = base_dir+'vec_brain.pkl'
labelFile = base_dir+'multi_class_pic_tags.pkl'

def mnist_generator(images, targets, batch_size, n_labelled, limit=None):
    rng_state = numpy.random.get_state()
    '''
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    '''
    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    '''
    test_img = images[0]
    test_target = targets[0]
    print(test_img.shape)
    print(test_target)
    #scipy.misc.imsave('./training_data.png', numpy.squeeze(test_img.reshape(28,28,1)))
    for k in range(len(images)):
            images[k] = test_img
            targets[k] = test_target
    '''

    def get_epoch():
        '''
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)
        '''

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 737)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)
            print("not none ")
            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:
            print("yield ")
            return image_batches, target_batches
            #for i in range(len(image_batches)):
                #yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    image_batches = images.reshape(-1, batch_size, 737)
    target_batches = targets.reshape(-1, batch_size)
    return image_batches, target_batches
    #return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in /tmp, downloading...")
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

    print('[INFO]', len(imgpkl), 'images in all')

    # ran = np.choice(6573, self.batch_size)
    for k in labelpkl.keys():
        x.append(imgpkl[k])
        y.append(labelpkl[k])
        print("[INFO] training_data:", k)
        count += 1
        break

    for i in range(4999):
        x.append(imgpkl[k])
        y.append(labelpkl[k])
        count += 1

    x, y = shuffle(np.array(x).reshape((5000, 1, 737, 1)).astype(np.float64),
                   np.array(y).reshape(5000).astype(np.int))

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

    return mnist_generator(x, y, batch_size, n_labelled)