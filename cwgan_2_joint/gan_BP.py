# coding=utf-8
import os, sys
sys.path.append(os.getcwd())

import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
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

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

tags_leave_out = [36, 37, 34, 44, 38, 28, 23, 33, 32]
flags = tf.app.flags
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("class_id", "0", "classes")
# four place to add  conditional labels
flags.DEFINE_string("cond", "11111", "conditional")
flags.DEFINE_string("concat", "1", "concate method") # 0: concat yb with channel; 1: trainable w
flags.DEFINE_string("gpu", "3", "gpu_id")
flags.DEFINE_integer('multi', 6, 'times of penalty')
FLAGS = flags.FLAGS

'''
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
'''

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

sample_dir='./acd_gan_2' + '_' + FLAGS.cond + '_' + FLAGS.concat+'/samples'
test_dir = './acd_gan_2' + '_' + FLAGS.cond + '_' + FLAGS.concat+'/test'
cost_dir = './acd_gan_2' + '_' + FLAGS.cond + '_' + FLAGS.concat+'/cost'
list_dir = './acd_gan_2' + '_' + FLAGS.cond + '_' + FLAGS.concat

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
            proportion[i] = (float(min_) / record[i][0])*FLAGS.multi
        else:
            proportion[i] = 0.
    return proportion

# like a global variable
proportion = cal_proportion()

print("[INFO] Finish loading proportion")

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(cost_dir):
    os.makedirs(cost_dir)

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 18538 # Number of pixels in MNIST (28*28)
y_dim = 45
#filter_size = [3,3]
lib.print_model_settings(locals().copy())


if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""

    y_shapes = y.get_shape()
    #x = tf.transpose(x, [0,4,2,3,1])
    x_shapes = x.get_shape().as_list()
    #print('Shapes: ', x_shapes)
    x = concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], x_shapes[3], y_shapes[4]])], 4)
    #x = tf.transpose(x, [0, 4, 2, 3, 1])
    #x_shapes = x.get_shape().as_list()
    #print('Shapes2: ', x_shapes)
    return x

def concat_2(output,yb,w):

    x_shapes = output.get_shape().as_list()
    yb_temp = yb * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], x_shapes[3], y_dim])

    yb_temp = tf.transpose(yb_temp, [1, 2, 3, 0, 4])
    temp = tf.matmul(yb_temp, w)
    temp = tf.transpose(temp, [3, 0, 1, 2, 4])
    output = concat([output, temp], 4)

    return output

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, y = None, noise = None, Reuse = None):
    g_w = 2
    g_h = 2
    g_d = 2

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 1, y_dim])
    # 1st concat
    if(FLAGS.cond[0] =='1'):
        noise = concat([noise, y], 1)
    noise_shape = noise.get_shape().as_list()
    output = lib.ops.linear.Linear('Generator.Input', noise_shape[-1], g_h * g_w * g_d * 8 * DIM, noise)

    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)

    output = LeakyReLU(output)
    output = tf.reshape(output, [-1, g_d, g_h, g_w,  8 * DIM])
    #NDHWC


    # 2nd concat
    if(FLAGS.cond[1] =='1'):
        if(FLAGS.concat == '0'):
            output = conv_cond_concat(output, yb)
        else:

            if Reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse = True):
                    w = tf.get_variable('Generator.w1', [ 2, 2, 2, y_dim, 15],initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output,yb,w)
            else:
                w = tf.get_variable('Generator.w1', [ 2, 2, 2, y_dim, 15],initializer=tf.truncated_normal_initializer(stddev=0.02))
                output = concat_2(output,yb,w)

    output = lib.ops.conv3d.Deconv('Generator.2', output.get_shape().as_list()[-1], [BATCH_SIZE, 3, 4, 4,  4 * DIM], output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)

    output = LeakyReLU(output)

    # 3rd concat
    if (FLAGS.cond[2] == '1'):
        if (FLAGS.concat == '0'):
            output = conv_cond_concat(output, yb)
        else:
            '''
            yb2_ = lib.ops.linear.Linear('Generator.yb2_', y_dim, 3 * 4 * 4 * 3, y)
            yb2_ = tf.tanh(yb2_)
            yb2_ = tf.reshape(yb2_, [-1, 3, 4, 4, 3])
            output = concat([output, yb2_], 4)

            '''
            if Reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse = True):
                    w = tf.get_variable('Generator.w2', [3, 4, 4, y_dim, 15],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))

                    output = concat_2(output, yb, w)
            else:
                w = tf.get_variable('Generator.w2', [3, 4, 4, y_dim, 15],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                output = concat_2(output, yb, w)


    # def Deconv(name, input_dim,output_shape, inputs,stride = 2):
    output = lib.ops.conv3d.Deconv('Generator.3', output.get_shape().as_list()[-1], [BATCH_SIZE, 6, 7, 8, 2* DIM], output)

    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 1, 2, 3], output)
    output = LeakyReLU(output)
    #4th concat
    if (FLAGS.cond[3] == '1'):
        if (FLAGS.concat == '0'):
            output = conv_cond_concat(output, yb)
        else:
            if Reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse = True):
                    w = tf.get_variable('Generator.w3', [6, 7, 8, y_dim, 15],initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)
            else:
                w = tf.get_variable('Generator.w3', [6, 7, 8, y_dim, 15],initializer=tf.truncated_normal_initializer(stddev=0.02))
                output = concat_2(output, yb, w)

    output = lib.ops.conv3d.Deconv('Generator.4', output.get_shape().as_list()[-1], [BATCH_SIZE, 12, 13, 16,  DIM], output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0, 1, 2, 3], output)
    output = LeakyReLU(output)
    # 5th concat
    if (FLAGS.cond[4] == '1'):
        if (FLAGS.concat == '0'):
            output = conv_cond_concat(output, yb)
        else:
            if Reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    w = tf.get_variable('Generator.w4', [12, 13, 16, y_dim, 15],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)
            else:
                w = tf.get_variable('Generator.w4', [12, 13, 16, y_dim, 15],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                output = concat_2(output, yb, w)

    output = lib.ops.conv3d.Deconv('Generator.5', output.get_shape().as_list()[-1], [BATCH_SIZE, 23, 26, 31, 1],
                                   output)


    output = tf.nn.tanh(output)
    output = tf.reshape(output, [-1, OUTPUT_DIM])
    return output

flag = 0
def Discriminator(inputs, y = None, prop = None, Reuse=None):
    # default :"NDHWC"
    output = tf.reshape(inputs, [-1, 23, 26, 31, 1])

    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 1, y_dim])
    # 1st concat
    if (FLAGS.cond[4] == '1' and flag==1):
        if (FLAGS.concat == '0'):
            output = conv_cond_concat(output, yb)
        else:

            #yb3_ = lib.ops.linear.Linear('Discriminator.yb3_', y_dim, 11 * 13 * 15 * 10, y)
            if Reuse:
                with tf.variable_scope(tf.get_variable_scope(),reuse= Reuse):
                    w = tf.get_variable('Discriminator.w4', [23, 26, 31, y_dim, 15], initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)
            else:
                w = tf.get_variable('Discriminator.w4', [23, 26, 31, y_dim, 15],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                output = concat_2(output, yb, w)


    output = lib.ops.conv3d.Conv3D('Discriminator.1', output.get_shape().as_list()[-1], DIM, output, stride=2)
    output = LeakyReLU(output)

    # 2nd concat
    if (FLAGS.cond[3] == '1' and flag==1):
        if (FLAGS.concat == '0'):
            output = conv_cond_concat(output, yb)
        else:
            '''
            yb2_ = lib.ops.linear.Linear('Discriminator.yb2_', y_dim, 6 * 7 * 8 * 3, y)
            yb2_ = tf.reshape(yb2_, [-1,  6, 7, 8, 3])
            yb2_ = tf.tanh(yb2_)
            output = concat([output, yb2_], 4)
            '''
            if Reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse= Reuse):
                    w = tf.get_variable('Discriminator.w3', [12, 13, 16, y_dim, 15], initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)
            else:
                w = tf.get_variable('Discriminator.w3', [12, 13, 16, y_dim, 15],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                output = concat_2(output, yb, w)

    output = lib.ops.conv3d.Conv3D('Discriminator.2', output.get_shape().as_list()[-1], 2*DIM, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)
    # 3rd concat
    # output = conv_cond_concat(output, yb)
    if (FLAGS.cond[2] == '1'  and flag==1):
        if (FLAGS.concat == '0'):
            output = conv_cond_concat(output, yb)
        else:

            if Reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse=Reuse):
                    w = tf.get_variable('Discriminator.w2', [6, 7, 8, y_dim, 15],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)
            else:
                w = tf.get_variable('Discriminator.w2', [6, 7, 8, y_dim, 15],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                output = concat_2(output, yb, w)


    output = lib.ops.conv3d.Conv3D('Discriminator.3', output.get_shape().as_list()[-1], 4*DIM,  output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)

    output = LeakyReLU(output)


    if (FLAGS.cond[1] == '1' and flag==1):
        if (FLAGS.concat == '0'):
            output = conv_cond_concat(output, yb)
        else:

            if Reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse=Reuse):
                    w = tf.get_variable('Discriminator.w1', [3, 4, 4, y_dim, 15],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)
            else:
                w = tf.get_variable('Discriminator.w1', [3, 4, 4, y_dim, 15],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                output = concat_2(output, yb, w)


    output = lib.ops.conv3d.Conv3D('Discriminator.4', output.get_shape().as_list()[-1], 8*DIM,  output, stride=2)


    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 2*2*2*8*DIM])
    #output = tf.reshape(output, [-1, 3 * 4 * 4 * 4 * DIM])

    if (FLAGS.cond[0] == '1'  and flag==1):
        output = concat([output, y], 1)
    output_shape = output.get_shape().as_list()
    output = lib.ops.linear.Linear('Discriminator.5', output_shape[-1], 128, output)

    # add one linear layer into the model
    #output = lib.ops.linear.Linear('Discriminator.6',  output_shape[-1], 128, output)

    #output = LeakyReLU(output)
    output_shape = output.get_shape().as_list()

    output_c = lib.ops.linear.Linear('Discriminator.c_outout', output_shape[-1], y_dim, output)
    output = lib.ops.linear.Linear('Discriminator.Output', 128, 1, output)

    output = tf.reshape(output, [-1])
    # weighted
    #output = output* prop
    return output,output_c

def Classifier(inputs):
    # default :"NDHWC"
    output = tf.reshape(inputs, [-1, 23, 26, 31, 1])

    output = lib.ops.conv3d.Conv3D('Classifier.1', output.get_shape().as_list()[-1], DIM, output, stride=2)
    output = LeakyReLU(output)

    # 2nd concat
    output = lib.ops.conv3d.Conv3D('Classifier.2', output.get_shape().as_list()[-1], 2*DIM, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Classifier.BN2', [0,2,3], output)
    output = LeakyReLU(output)
    # 3rd concat
    # output = conv_cond_concat(output, yb)

    output = lib.ops.conv3d.Conv3D('Classifier.3', output.get_shape().as_list()[-1], 4*DIM,  output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Classifier.BN3', [0,2,3], output)

    output = LeakyReLU(output)

    output = lib.ops.conv3d.Conv3D('Classifier.4', output.get_shape().as_list()[-1], 8*DIM,  output, stride=2)


    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Classifier.BN4', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 2*2*2*8*DIM])
    #output = tf.reshape(output, [-1, 3 * 4 * 4 * 4 * DIM])

    # add one linear layer into the model
    output_shape = output.get_shape().as_list()
    output = lib.ops.linear.Linear('Classifier.5', output_shape[-1], 128, output)

    #output = LeakyReLU(output)

    output = lib.ops.linear.Linear('Classifier.Output', 128, y_dim, output)

    return output


real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

if y_dim:
    y = tf.placeholder(tf.float32, [BATCH_SIZE, y_dim], name='y')
else:
    y = None

prop = tf.placeholder(tf.float32, [BATCH_SIZE], name='prop')

fake_data = Generator(BATCH_SIZE, y, None)
disc_real,clas_real = Discriminator(real_data, y, prop, None)
disc_fake ,clas_fake= Discriminator(fake_data, y, prop, True)

#clas_real = Classifier(real_data)
#clas_fake = Classifier(fake_data)

closs_real = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=tf.to_float(clas_real))
closs_fake = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=tf.to_float(clas_fake))

batch_closs_real = tf.reduce_mean(closs_real)
batch_closs_fake = tf.reduce_mean(closs_fake)

#preds = tf.to_int32(tf.round(tf.nn.softmax(clas_real)))
preds = tf.argmax(tf.nn.softmax(clas_real), 1)


c_cost = batch_closs_real + batch_closs_fake


gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')
#class_params = lib.params_with_name('Classifier')


if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':

    gen_cost = -tf.reduce_mean(disc_fake) + c_cost
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) + c_cost # （tf.multiply(disc_real, prop))

    #dis_cost_1 = tf.reduce_mean(disc_real)
    #dis_cost_2 = tf.reduce_mean(tf.multiply(disc_real, prop))

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )
    
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    output1,_= Discriminator(interpolates, y, prop, True)

    gradients = tf.gradients(output1, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    learning_rate = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=global_step,
        decay_steps=2000,
        decay_rate=0.5,
        staircase=True)

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)

    # 1e-4,
    learning_rate2 = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=global_step,
        decay_steps=2000,
        decay_rate=0.5,
        staircase=True)
    '''
    disc_train_op = tf.train.GradientDescentOptimizer(
        learning_rate=1e-4
    ).minimize(disc_cost, var_list=disc_params)

    '''
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate2,
        beta1=0.5,
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    '''
    classifier_train_op = tf.train.AdamOptimizer(
                        learning_rate=learning_rate2,
                        beta1=0.5,
                        beta2=0.9
                        ).minimize(c_cost, var_list=class_params)
    '''
    clip_disc_weights = None
    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver()

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real,
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-5,
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-5,
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples

train_gen, dev_gen, test_gen = lib.BrainPedia.load_ckpt_data(BATCH_SIZE, list_dir) # load_mnist(BATCH_SIZE)#


choose_pool = []
for i in range(45):
    if (i not in tags_leave_out):
        choose_pool.append(i)

print('[INFO] Length of choose pool: ', len(choose_pool))

ran = np.random.choice(choose_pool, BATCH_SIZE)
fixed_labels = np.zeros((BATCH_SIZE, y_dim), dtype=np.float64)
for i, label in enumerate(ran):
    fixed_labels[i, ran[i]] = 1.0

print('[INFO] FIXED LABELS : ', np.sum(np.sum(fixed_labels, 0), 0))

fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
fixed_noise_samples = Generator(BATCH_SIZE, y, noise=fixed_noise, Reuse = True)

'''
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples,feed_dict={y: targets})#fixed_label})
    lib.save_images.save_images(
        samples.reshape((BATCH_SIZE, 28, 28)),
        'samples_{}.png'.format(frame))

'''


def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples, feed_dict={y: fixed_labels})

    count = 0
    for i in samples.reshape([BATCH_SIZE, 26, 31, 23]):
        temp = lib.upsampling.upsample_vectorized(i)
        img = nibabel.Nifti1Image(temp.get_data(), temp.affine)
        id = list(fixed_labels[count]).index(1.)
        filename = './{}/samples{}_{}_{}.nii.gz'.format(sample_dir, frame, count, id)
        nibabel.save(img, filename)
        count += 1
        if count == 20:
            break

# mskFile = open(base_dir + 'msk.pkl', 'rb')
# outputFile = open(base_dir + 'output.pkl', 'ab')
# Dataset iterator


def save_test_img(frame):
    choose_pool = []
    for i in range(45):
        if (i not in tags_leave_out):
            choose_pool.append(i)

    ran = np.random.choice(choose_pool, BATCH_SIZE)
    test_labels = np.zeros((BATCH_SIZE, y_dim), dtype=np.float64)
    for i, label in enumerate(ran):
        test_labels[i, ran[i]] = 1.0

    test_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    test_noise_samples = Generator(BATCH_SIZE, y, noise=test_noise, Reuse=True)

    samples = session.run(test_noise_samples, feed_dict={y: test_labels})

    count = 0
    for i in samples.reshape([BATCH_SIZE,  26, 31, 23]):

        temp = lib.upsampling.upsample_vectorized(i)

        img = nibabel.Nifti1Image(temp.get_data(), temp.affine)
        print(img.shape)
        # save labels in test picture names
        id = list(test_labels[count]).index(1.)
        filename = './{}/test_{}_{}_{}.nii.gz'.format(test_dir, frame, count, id)
        nibabel.save(img, filename)#img
        count += 1

def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images, targets


# Train loop
ac=0
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            # for i in xrange(3):
            _, g_loss = session.run([gen_train_op, gen_cost],feed_dict={real_data: _data, y: targets, prop: np.array(temp_prop)})
            #classifier_train_op
            print('iter:%d,  d_loss: %.4f,  g_loss: %.4f' % (iteration, _disc_cost, g_loss))
            lib.plot.plot(cost_dir + '/train gen cost', g_loss)
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS

        for i in xrange(disc_iters):
            _data, targets = next(gen)


            temp_prop = []
            for i in targets:
                id = list(i).index(1.)
                temp_prop.append(proportion[id])

            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data, y: targets, prop: np.array(temp_prop)}
            )
            #print('[INFO] :d1,d2',d1,d2)
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)
                # print("{INFO}d_loss",_disc_cost)

        if (iteration > 0):
            lib.plot.plot(cost_dir + '/train disc cost', _disc_cost)
            lib.plot.plot(cost_dir + '/time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 0 and iteration != 0:

            dev_disc_costs = []
            dev_gen_costs = []

            for images, tags in dev_gen():  # image and targets
                temp_prop = []
                for i in tags:
                    id = list(i).index(1.)
                    temp_prop.append(proportion[id])

                _dev_disc_cost, _dev_g_loss = session.run(
                    [disc_cost, gen_cost],
                    feed_dict={real_data: images, y: tags, prop: np.array(temp_prop)}
                )
                dev_disc_costs.append(_dev_disc_cost)
                dev_gen_costs.append(_dev_g_loss)
            lib.plot.plot(cost_dir + '/dev disc cost', np.mean(dev_disc_costs))
            lib.plot.plot(cost_dir + '/dev gen cost', np.mean(dev_gen_costs))
            '''
            if (iteration % 8000 == 0 and iteration != 0):
                generate_image(iteration, _data)
            '''

        # Write logs every 100 iters
        if (iteration < 5 and iteration > 0 ) or (iteration % 100 == 99):
            lib.plot.flush(cost_dir)
        lib.plot.tick()


        if (iteration >= 3000):
            start_time = time.time()

            pre_list = []
            tag_list = []
            tag_list1=[]
            for images, tags in test_gen():
                preds_ = session.run(preds, feed_dict={real_data: images, y: tags})

                pre_list.append(preds_)
                tag_list1.append(tags)

            pre_list = np.concatenate(pre_list, 0)
            tag_list1 = np.concatenate(tag_list1, 0)

            target_names = []
            for name in range(45):
                if (name not in tags_leave_out):
                    target_names.append('class_' + str(name))

            for m in tag_list1:
                tag_list.append(list(m).index(1.))

            if (accuracy_score(tag_list, pre_list) > ac):
                print(classification_report(tag_list, pre_list, digits=3))
                temp = accuracy_score(tag_list, pre_list)
                ac =temp
                print('iteration:', iteration, 'acc:%.6f' %(ac))
            if(iteration%100 == 0):

                print(classification_report(tag_list, pre_list, digits=3))
                print('iteration:',iteration,'acc:%.6f'%(ac))
