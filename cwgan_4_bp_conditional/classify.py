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
import tflib.plot
from six.moves import xrange
import pickle as pkl
from tflib.upsampling import *
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.save_images
import tflib.BrainPedia
import tflib.plot
import tflib.ops.conv3d
import nibabel
import nilearn.masking as masking
import tflib.upsampling

cost_dir = './classify_results'

if not os.path.exists(cost_dir):
    os.makedirs(cost_dir)
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
MODE = 'wgan-gp'    # dcgan, wgan, or wgan-gp
DIM = 64            # Model dimensionality
BATCH_SIZE = 50     # Batch size
CRITIC_ITERS = 5    # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10         # Gradient penalty lambda hyperparameter
ITERS = 50000       # How many generator iterations to train for
OUTPUT_DIM = 2145   # Number of pixels in MNIST (28*28)
y_dim = 45

lib.print_model_settings(locals().copy())


tags_leave_out = [36, 37, 34, 44, 38, 28, 23, 33, 32]
flags = tf.app.flags
flags.DEFINE_string("gpu", "1", "gpu_id")
flags.DEFINE_string("cond", "classify", "conditional")
flags.DEFINE_string("concat", "01", "concate method") # 0: concat yb with channel; 1: trainable w
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

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
            proportion[i] = (float(min_) / record[i][0])*6
        else:
            proportion[i] = 0.
    return proportion

# like a global variable
#proportion = cal_proportion()

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def Discriminator(inputs):
    # default :"NDHWC"
    output = tf.reshape(inputs, [-1, 11, 13, 15, 1])

    output = lib.ops.conv3d.Conv3D('Discriminator.1', output.get_shape().as_list()[-1], DIM, output, stride=2)
    output = LeakyReLU(output)
    #output = tf.nn.avg_pool3d(output, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding='SAME', data_format=None)
    # print('pool1:', output.shape)
    output = lib.ops.conv3d.Conv3D('Discriminator.2', output.get_shape().as_list()[-1], 2 * DIM, output, stride=2)
    output = LeakyReLU(output)

    #output = tf.nn.avg_pool3d(output, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding='SAME', data_format=None)
    # print('pool2:', output.shape)
    output = lib.ops.conv3d.Conv3D('Discriminator.3', output.get_shape().as_list()[-1], 4 * DIM, output, stride=2)
    output = LeakyReLU(output)

    #output = tf.nn.avg_pool3d(output, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding='SAME', data_format=None)

    # print('pool3:', output.shape)
    output = tf.reshape(output, [-1, 2 * 2 * 2 * 4 * DIM])

    # add one linear layer into the model
    output = lib.ops.linear.Linear('Discriminator.5', 2 * 2 * 2 * 4 * DIM, 128, output)
    output = lib.ops.linear.Linear('Discriminator.Output', 128, y_dim, output)

    return output


real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

if y_dim:
    y = tf.placeholder(tf.float32, [BATCH_SIZE, y_dim], name='y')
else:
    y = None
prop = tf.placeholder(tf.float32, [BATCH_SIZE], name='prop')

logits = Discriminator(real_data)
preds = tf.to_int32(tf.round(tf.nn.softmax(logits)))
loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.to_float(logits), labels=y)

# return a maximum value
correct = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

batch_loss = tf.reduce_mean(loss)#tf.multiply(loss, prop))

disc_params = lib.params_with_name('Discriminator')
global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

learning_rate2 = tf.train.exponential_decay(
    learning_rate=1e-4,
    global_step=global_step,
    decay_steps=2000,
    decay_rate=0.5,
    staircase=True)

disc_train_op = tf.train.AdamOptimizer(
    learning_rate=learning_rate2,
    beta1=0.5,
    beta2=0.9
).minimize(batch_loss, var_list=disc_params) #

train_gen, dev_gen, test_gen = lib.BrainPedia.load_cross(BATCH_SIZE)

#train_gen = lib.BrainPedia.load_generated_data(BATCH_SIZE)


def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images, targets

ac= 0.0
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    gen = inf_train_gen()

    for iteration in xrange(ITERS):

        _data, targets = next(gen)
        start_time = time.time()
        temp_prop = []
        for i in targets:
            id = list(i).index(1.)
            temp_prop.append(proportion[id])


        '''
        labels=[]
        for i in targets:
            labels .append(list(i).index(1.))
        '''
        preds_, _, acc, lgts, crct = session.run([preds, disc_train_op,accuracy, logits, correct],
                                                 feed_dict={real_data: _data, y: targets, prop: np.array(temp_prop)})
        '''
        f1 = metrics.f1_score(np.array(targets).astype(np.int32), np.array(preds_), average='micro')
        pre = metrics.precision_score(np.array(targets).astype(np.int32), np.array(preds_), average='micro')
        rec = metrics.recall_score(np.array(targets).astype(np.int32), np.array(preds_), average='micro')
        ac = accuracy_score(np.array(targets).astype(np.int32), np.array(preds_))
        print('iter: %5d, acc: %.3f, f1: %.3f, pre: %.3f, rec: %.3f '%(iteration, ac, f1, pre, rec))
        '''
        '''
        lib.plot.plot(cost_dir +'/training F1', np.mean(f1))
        lib.plot.plot(cost_dir+'/training accuracy', np.mean(acc))
        '''
        '''
        if(iteration%50 == 0 ):

            f1_list = []
            pre_list = []
            rec_list = []
            pred_list = []
            target_list = []
            for images, tags in dev_gen(): # image and targets
                preds_, acc, lgts, crct = session.run([preds, accuracy, logits, correct],
                    feed_dict={real_data: images, y: tags}
                )

                f1 = metrics.f1_score(np.array(tags).astype(np.int32), np.array(preds_), average='micro')
                pre = metrics.precision_score(np.array(tags).astype(np.int32), np.array(preds_), average='micro')
                rec = metrics.recall_score(np.array(tags).astype(np.int32), np.array(preds_), average='micro')

                f1_list.append(f1)
                break
            
            #print('DEV  f1: %.3f' % (np.mean(np.array(f1_list))))
            lib.plot.plot(cost_dir+'/dev F1', np.mean(f1))
            lib.plot.plot(cost_dir +'/dev accuracy', np.mean(acc))
        
        if (iteration < 5 and iteration > 0) or (iteration % 100 == 99):
            lib.plot.flush(cost_dir)
        lib.plot.tick()
        '''

        if(iteration >=6000):
            start_time = time.time()
            f1_test_list = []
            acc_list =[]
            idx = (1398//BATCH_SIZE)

            pre_list = []
            tag_list = []
            for images, tags in test_gen():

                preds_, acc, lgts, crct = session.run([preds, accuracy, logits, correct],
                                                     feed_dict={real_data: images, y: tags})

                pre_list.append(preds_)
                tag_list.append(tags)
            pre_list = np.concatenate(pre_list, 0)
            tag_list = np.concatenate(tag_list, 0)

            target_names = []
            for name in range(45):
                if (name not in tags_leave_out):
                    target_names.append('class_' + str(name))

                #print('p shape',preds_.shape,type(preds_), 'tag_shape',tags.shape,type(tags))
            if(accuracy_score(tag_list, pre_list)>ac):
                print(classification_report(tag_list, pre_list, digits=3))
                print('iteration: ', iteration, 'acc: ',accuracy_score(tag_list, pre_list))
                ac = accuracy_score(tag_list, pre_list)
            #print("TEST", 'iteration: %5d, acc: %.3f, f1: %.3f' % (iteration, np.mean(acc_list), np.mean(f1_test_list)))
