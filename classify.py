# coding=utf-8
import os, sys
sys.path.append(os.getcwd())
sys.path.append('..')
sys.path.append('/home/nfs/zpy/nilearn/')
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
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.save_images
import tflib.BrainPedia
import tflib.plot
import tflib.ops.conv3d


flags = tf.app.flags
flags.DEFINE_string("checkpoint_dir", "./checkpoint_2d1g", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string('or_shape', '[53,63,46]', 'original shape')
flags.DEFINE_string('aug_dir', './tests/', 'test')
flags.DEFINE_string('base_dir', '/home/nfs/zpy/BrainPedia/pkl/', 'base_dir')
flags.DEFINE_string('imageFile', 'original_dim.pkl', 'imageFile')
flags.DEFINE_string('labelFile', 'multi_class_pic_tags.pkl', 'labelFile')
flags.DEFINE_string("cond", "classify", "conditional")
flags.DEFINE_string("gpu", "1", "gpu_id")
flags.DEFINE_string("gen_dir", '', 'gen_dir') # /home/nfs/zpy/BrainPedia/New_image_processing/CVAE/tests_cvae_1/', 'generated dir')#'/home/nfs/zpy/BrainPedia/New_image_processing/DCGAN_sync/tests_DC_TanhInG/', 'dir')#'/home/nfs/zpy/BrainPedia/New_image_processing/original_dim_gan_1952/tests_containDEV/', 'gen dir')
flags.DEFINE_integer("gen_num", 100, 'gen_num')
flags.DEFINE_integer('fold', 0, 'fold for classify')
flags.DEFINE_string('gen_source', 'NN', 'USING data from GMM Gauss NN')

FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

cost_dir = './classify_results'

if not os.path.exists(cost_dir):
    os.makedirs(cost_dir)

MODE = 'wgan-gp'    # dcgan, wgan, or wgan-gp
DIM = 64            # Model dimensionality
BATCH_SIZE = 50     # Batch size
CRITIC_ITERS = 5    # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10         # Gradient penalty lambda hyperparameter
ITERS = 50000       # How many generator iterations to train for
OUTPUT_DIM = 153594   # Number of pixels in MNIST (28*28)
y_dim = 45

lib.print_model_settings(locals().copy())

tags_leave_out = [36, 37, 34, 44, 38, 28, 23, 33, 32]

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

def Classifier(inputs, origin_shape=None, layers=4, Reuse=None):
    # default :"NDHWC"
    shape = origin_shape[1:-1].split(',')
    shape = [int(i) for i in shape]
    output = tf.reshape(inputs, [-1, shape[2], shape[0], shape[1], 1])
    # output = inputs
    sp_layers = []
    sp_layers.append(shape)
    for i in range(layers):
        sp_layers.append([i / 2 + i % 2 for i in sp_layers[-1]])
    print('1', output.shape)
    output = lib.ops.conv3d.Conv3D('Classifier.1', output.get_shape().as_list()[-1], DIM, output, stride=2)
    output = LeakyReLU(output)
    print('2', output.shape)
    output = lib.ops.conv3d.Conv3D('Classifier.2', output.get_shape().as_list()[-1], 2 * DIM, output, stride=2)
    output = LeakyReLU(output)
    print('3', output.shape)
    output = lib.ops.conv3d.Conv3D('Classifier.3', output.get_shape().as_list()[-1], 4 * DIM,  output, stride=2)
    output = LeakyReLU(output)
    print('4', output.shape)
    output = lib.ops.conv3d.Conv3D('Classifier.4', output.get_shape().as_list()[-1], 8 * DIM, output, stride=2)
    output = LeakyReLU(output)

    sp = output.get_shape().as_list()
    out_dim = sp[1]*sp[2]*sp[3]*sp[4]

    output = tf.reshape(output, [-1, out_dim])

    sp = output.get_shape().as_list()
    output = lib.ops.linear.Linear('Classifier.5',  sp[1], 128, output)
    output = lib.ops.linear.Linear('Classifier.Output', 128, y_dim, output)
    # output = tf.reshape(output, [-1])
    # weighted
    #output = output* prop
    return output

real_data = tf.compat.v1.placeholder(tf.float32, shape=[None, OUTPUT_DIM])

if y_dim:
    y = tf.compat.v1.placeholder(tf.float32, [None, y_dim], name='y')
else:
    y = None


logits = Classifier(real_data, origin_shape=FLAGS.or_shape)
preds = tf.cast(tf.round(tf.nn.softmax(logits)), dtype=tf.int32)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.cast(logits, dtype=tf.float32), labels=tf.stop_gradient(y))

# return a maximum value
correct = tf.equal(tf.argmax(input=preds, axis=1), tf.argmax(input=y, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct, tf.float32))

batch_loss = tf.reduce_mean(input_tensor=loss)#tf.multiply(loss, prop))

disc_params = lib.params_with_name('Classifier')

global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.compat.v1.GraphKeys.GLOBAL_STEP, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES])
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    disc_train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(batch_loss, var_list=disc_params,global_step=global_step) #

saver = tf.compat.v1.train.Saver()

def save(checkpoint_dir, sess, step=0):
    model_name = "1952_ori.model"
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
def load(sess, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt,ckpt.model_checkpoint_path)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0
def load_ckpt(sess, ckpt):
    print(" [*] Reading checkpoints...")
    saver.restore(sess, ckpt)


train_gen, test_gen = lib.BrainPedia.load_2(BATCH_SIZE,
                                           FLAGS.base_dir,
                                           FLAGS.imageFile,
                                           FLAGS.labelFile,
                                           FLAGS.gen_dir,
                                           FLAGS.gen_num,
                                           FLAGS.fold,
                                           FLAGS.gen_source
                                            )
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images, targets


MAX_FRACTION = 0.5
NUM_THREADS = 2
sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.allow_growth = True #
sess_config.intra_op_parallelism_threads = NUM_THREADS
sess_config.gpu_options.per_process_gpu_memory_fraction = MAX_FRACTION #

ac = 0.0

with tf.compat.v1.Session(config=sess_config) as session:
    session.run(tf.compat.v1.global_variables_initializer())
    gen = inf_train_gen()
    for iteration in xrange(ITERS):

        _data, targets = next(gen)
        start_time = time.time()

        preds_, _, acc, lgts, crct = session.run([preds, disc_train_op, accuracy, logits, correct],
                                                 feed_dict={real_data: _data, y: targets})

        if(iteration %500 ==0):
            print(iteration, ' iteration has finished')
        if(iteration >=5000):
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

        if(iteration == 20000):
            break
