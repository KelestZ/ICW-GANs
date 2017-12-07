# coding=utf-8
import os, sys
sys.path.append(os.getcwd())

import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from sklearn import metrics
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



def cal_proportion():

    base_dir = '/home/nfs/zpy/BrainPedia/pkl/'
    f = open(base_dir + 'record_size_per_tag', 'rb')

    record = pkl.load(f)
    f.close()
    train_size = []
    proportion = {}
    for i in record.keys():
        if (record[i] != []):
            train_size.append(record[i][0])

    min_ = min(train_size)
    for i in record.keys():
        if (record[i] != []):
            proportion[i] = (float(min_) / record[i][0])*4.
        else:
            proportion[i] = 0.
    return proportion

# like a global variable
proportion = cal_proportion()
print("[INFO] Finish loading proportion")
MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 2145 # Number of pixels in MNIST (28*28)
y_dim = 45
#filter_size = [3,3]
lib.print_model_settings(locals().copy())


tags_leave_out = [36, 37, 34, 44, 38, 28, 23, 33, 32]
flags = tf.app.flags
flags.DEFINE_string("gpu", "3", "gpu_id")
flags.DEFINE_string("cond", "classify", "conditional")
flags.DEFINE_string("concat", "01", "concate method") # 0: concat yb with channel; 1: trainable w
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def Discriminator(inputs):
    # default :"NDHWC"
    output = tf.reshape(inputs, [-1, 11, 13, 15, 1])

    output = lib.ops.conv3d.Conv3D('Discriminator.1', output.get_shape().as_list()[-1], DIM, output, stride=1)
    output = LeakyReLU(output)
    output = tf.nn.max_pool3d(output, ksize=[1,4,4,4,1], strides=[1,2,2,2,1],
                              padding='SAME', data_format=None)

    output = lib.ops.conv3d.Conv3D('Discriminator.2', output.get_shape().as_list()[-1], 2*DIM, output, stride=1)
    output = LeakyReLU(output)

    output = tf.nn.max_pool3d(output, ksize=[1, 4, 4, 4, 1], strides=[1, 2, 2, 2, 1],
                              padding='SAME', data_format=None)


    output = lib.ops.conv3d.Conv3D('Discriminator.3', output.get_shape().as_list()[-1], 4*DIM,  output, stride=1)
    output = LeakyReLU(output)

    output = tf.nn.max_pool3d(output, ksize=[1, 4, 4, 4, 1], strides=[1, 2, 2, 2, 1],
                              padding='SAME', data_format=None)


    output = tf.reshape(output, [-1, 2*2*2*4*DIM])

    # add one linear layer into the model
    output = lib.ops.linear.Linear('Discriminator.5', 2 * 2 * 2 * 4 * DIM, 128, output)
    output = lib.ops.linear.Linear('Discriminator.Output', 128, y_dim, output)

    return tf.nn.sigmoid(output)

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

if y_dim:
    y = tf.placeholder(tf.float32, [BATCH_SIZE, y_dim], name='y')
else:
    y = None



# prop = tf.placeholder(tf.float32, [BATCH_SIZE], name='prop')

logits = Discriminator(real_data)

preds = tf.to_int32(tf.round(tf.nn.softmax(logits)))

loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
batch_loss = tf.reduce_mean(loss)
disc_params = lib.params_with_name('Discriminator')
global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

# 1e-4,
learning_rate2 = tf.train.exponential_decay(
    learning_rate=1e-4,
    global_step=global_step,
    decay_steps=4000,
    decay_rate=0.5,
    staircase=True)

disc_train_op = tf.train.AdamOptimizer(
    learning_rate=learning_rate2,
    beta1=0.5,
    beta2=0.9
).minimize( batch_loss, var_list=disc_params)

train_gen, dev_gen = lib.BrainPedia.load_cross(BATCH_SIZE) # load_mnist(BATCH_SIZE)#


def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images, targets

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        start_time = time.time()

        _data, targets = next(gen)
        labels=[]
        for i in targets:
            labels .append(list(i).index(1.))

        preds_, _ = session.run([preds, disc_train_op ], feed_dict={real_data: _data, y: targets})
        if(iteration % 50 == 0 ):

            for i in range(len(preds_)):
                print('preds_', preds_[i])
                print('labels',targets[i])
            f1 = metrics.f1_score(np.array(targets).astype(np.int32), np.array(preds_))
            pre = metrics.precision_score(np.array(targets), np.array(preds_))
            rec = metrics.recall_score(np.array(targets), np.array(preds_))
            print('iteration', iteration,'train_set f1,pre,rec is : ', f1, pre, rec)