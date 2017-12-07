# -*- coding : utf-8
import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.conv3d

import tflib.ops.deconv2d
import tflib.save_images
import tflib.BrainPedia
import tflib.plot

from six.moves import xrange
import nibabel

import nilearn.masking as masking
import tflib.upsampling
import pickle as pkl
# from tflib.BrainPedia import sample_y

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 1 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 18538 # Number of pixels in MNIST (28*28)
y_dim = 45


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

    return tf.transpose(x, [0,3,1,2])
def concat_2(output,yb,w):

    x_shapes = output.get_shape().as_list()
    yb_temp = yb * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], x_shapes[3], y_dim])

    yb_temp = tf.transpose(yb_temp, [1, 2, 3, 0, 4])
    temp = tf.matmul(yb_temp, w)
    temp = tf.transpose(temp, [3, 0, 1, 2, 4])
    output = concat([output, temp], 4)

    return output

class CIWGAN(object):
    def __init__(self, sess, MODE='wgan-gp', DIM=64, BATCH_SIZE =50, LAMBDA = 10, ITERS =200000,
                 checkpoint_dir=None, y_dim=45, FLAGS = None,
                 tags_leave_out = [36, 37, 34, 44, 38, 28, 23, 33, 32],
                cost_dir=None, test_dir=None, sample_dir=None, list_dir=None):

        self.sess = sess
        self.MODE = MODE
        self.DIM = DIM
        self.BATCH_SIZE = BATCH_SIZE
        self.CRITIC_ITERS = 5  # For WGAN and WGAN-GP, number of critic iters per gen iter
        self.LAMBDA = LAMBDA  # Gradient penalty lambda hyperparameter
        self.y_dim = y_dim
        self.ITERS = ITERS
        self.dataset_name = FLAGS.cond + '_' +FLAGS.concat
        self.cond = FLAGS.cond
        self.concat = FLAGS.concat


        self.checkpoint_dir = checkpoint_dir
        self.tags_leave_out = tags_leave_out
        self.cost_dir = cost_dir
        self.test_dir = test_dir
        self.sample_dir = sample_dir
        self.list_dir = list_dir
        self.build_model()
        self.iter = 0
    def build_model(self):

        self.real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

        if y_dim:
            self.y = tf.placeholder(tf.float32, [BATCH_SIZE, y_dim], name='y')
        else:
            self.y = None
        self.prop = tf.placeholder(tf.float32, [BATCH_SIZE], name='prop')

        self.fake_data = self.Generator(self.BATCH_SIZE, self.y, Reuse=None)

        self.disc_real = self.Discriminator(self.real_data, self.y, Reuse=None)
        self.disc_fake = self.Discriminator(self.fake_data, self.y, Reuse=True)

        gen_params = lib.params_with_name('Generator')
        disc_params = lib.params_with_name('Discriminator')

        if MODE=='wgan-gp':
            print('[INFO] WGAN_BP')
            self.gen_cost = -tf.reduce_mean(self.disc_fake)
            self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE, 1],
                minval=0.,
                maxval=1.
            )
            differences = self.fake_data - self.real_data

            interpolates = self.real_data + (alpha * differences)

            gradients = tf.gradients(self.Discriminator(interpolates, self.y, Reuse=True), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.disc_cost += LAMBDA * gradient_penalty

            global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            learning_rate = tf.train.exponential_decay(
                learning_rate=1e-4,
                global_step=global_step,
                decay_steps=5000,
                decay_rate=0.5,
                staircase=True)

            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.gen_cost, var_list=gen_params)

            # 1e-4,
            learning_rate2 = tf.train.exponential_decay(
                learning_rate=1e-4,
                global_step=global_step,
                decay_steps=5000,
                decay_rate=0.5,
                staircase=True)
            '''
            disc_train_op = tf.train.GradientDescentOptimizer(
                learning_rate=1e-4
            ).minimize(disc_cost, var_list=disc_params)

            '''
            self.disc_train_op = tf.train.AdamOptimizer(
                learning_rate=learning_rate2,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.disc_cost, var_list=disc_params)

            self.clip_disc_weights = None
            # Set up the Saver for saving and restoring model checkpoints.
            self.saver = tf.train.Saver()

    def train(self):
        train_gen, dev_gen,_ = lib.BrainPedia.load_ckpt_data(BATCH_SIZE, self.list_dir)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()


        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        gen = self.inf_train_gen(train_gen)

        for iteration in xrange(0, ITERS):
            self.iter = iteration
            start_time = time.time()

            if iteration > 0:
                _, g_loss = self.sess.run([self.gen_train_op, self.gen_cost],feed_dict={self.y: targets})
                print('iter:%d,  d_loss: %.4f,  g_loss: %.4f' % (iteration, _disc_cost, g_loss))
                lib.plot.plot(self.cost_dir + '/train gen cost', g_loss)
            if MODE == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = self.CRITIC_ITERS

            for i in xrange(disc_iters):#scaling_2_v4_10101_0
                _data, targets = next(gen)

                _disc_cost, _ = self.sess.run(
                    [self.disc_cost, self.disc_train_op],
                    feed_dict={self.real_data: _data, self.y: targets}
                )
                if self.clip_disc_weights is not None:
                    _ = self.sess.run(self.clip_disc_weights)

            lib.plot.plot(self.cost_dir + '/train disc cost', _disc_cost)
            lib.plot.plot(self.cost_dir + '/time', time.time() - start_time)

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 200 == 0 and iteration != 0:
                dev_disc_costs = []
                dev_gen_costs = []

                for images, tags in dev_gen():  # image and targets
                    temp_prop = []
                    '''
                    for i in tags:
                        id = list(i).index(1.)
                        temp_prop.append(proportion[id])
                    '''
                    _dev_disc_cost, _dev_g_loss = self.sess.run(
                        [self.disc_cost, self.gen_cost],
                        feed_dict={self.real_data: images, self.y: tags}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                    dev_gen_costs.append(_dev_g_loss)
                lib.plot.plot(self.cost_dir + '/dev disc cost', np.mean(dev_disc_costs))
                lib.plot.plot(self.cost_dir + '/dev gen cost', np.mean(dev_gen_costs))
            '''
            if iteration % 200 == 0 and iteration !=56000:
                # fixed_labels are not written
                fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
                fixed_noise_samples = self.Generator(BATCH_SIZE, noise=fixed_noise, Reuse=True)
                self.generate_image(iteration, fixed_noise_samples)
            '''
            # Write logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush(self.cost_dir)
            lib.plot.tick()

            if np.mod(iteration, 2000) == 0 and iteration != 0:
                print('[INFO] Save checkpoint...')
                self.save(self.checkpoint_dir, iteration)

            if (iteration >=35000):
                self.save_test_img(iteration)

    def LeakyReLU(self, x, alpha=0.2):
        return tf.maximum(alpha * x, x)

    def ReLULayer(self, name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(
            name + '.Linear',
            n_in,
            n_out,
            inputs,
            initialization='he'
        )
        return tf.nn.relu(output)

    def LeakyReLULayer(self, name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(
            name + '.Linear',
            n_in,
            n_out,
            inputs,
            initialization='he'
        )
        return self.LeakyReLU(output)

    def Generator(self, n_samples, y=None, noise=None, Reuse=None):
        g_w = 2
        g_h = 2
        g_d = 2

        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 1, y_dim])
        # 1st concat
        if (self.cond[0] == '1'):
            noise = concat([noise, y], 1)
        noise_shape = noise.get_shape().as_list()
        output = lib.ops.linear.Linear('Generator.Input', noise_shape[-1], g_h * g_w * g_d * 8 * DIM, noise)

        #output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = self.LeakyReLU(output)
        output = tf.reshape(output, [-1, g_d, g_h, g_w, 8 * DIM])
        # NDHWC

        # 2nd concat
        if (self.cond[1] == '1'):
            if (self.concat == '0'):
                output = conv_cond_concat(output, yb)
            else:

                if Reuse:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        w = tf.get_variable('Generator.w1', [2, 2, 2, y_dim, 15],
                                            initializer=tf.truncated_normal_initializer(stddev=0.02))
                        output = concat_2(output, yb, w)
                else:
                    w = tf.get_variable('Generator.w1', [2, 2, 2, y_dim, 15],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)

        output = lib.ops.conv3d.Deconv('Generator.2', output.get_shape().as_list()[-1], [BATCH_SIZE, 3, 4, 4, 4 * DIM],
                                       output)
        #output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 1, 2, 3], output)

        output = self.LeakyReLU(output)

        # 3rd concat
        if (self.cond[2] == '1'):
            if (self.concat == '0'):
                output = conv_cond_concat(output, yb)
            else:
                '''
                yb2_ = lib.ops.linear.Linear('Generator.yb2_', y_dim, 3 * 4 * 4 * 3, y)
                yb2_ = tf.tanh(yb2_)
                yb2_ = tf.reshape(yb2_, [-1, 3, 4, 4, 3])
                output = concat([output, yb2_], 4)

                '''
                if Reuse:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        w = tf.get_variable('Generator.w2', [3, 4, 4, y_dim, 15],
                                            initializer=tf.truncated_normal_initializer(stddev=0.02))

                        output = concat_2(output, yb, w)
                else:
                    w = tf.get_variable('Generator.w2', [3, 4, 4, y_dim, 15],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)

        # def Deconv(name, input_dim,output_shape, inputs,stride = 2):
        output = lib.ops.conv3d.Deconv('Generator.3', output.get_shape().as_list()[-1], [BATCH_SIZE, 6, 7, 8, 2 * DIM],
                                       output)

        #output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 1, 2, 3], output)
        output = self.LeakyReLU(output)
        # 4th concat
        if (self.cond[3] == '1'):
            if (self.concat == '0'):
                output = conv_cond_concat(output, yb)
            else:
                if Reuse:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        w = tf.get_variable('Generator.w3', [6, 7, 8, y_dim, 15],
                                            initializer=tf.truncated_normal_initializer(stddev=0.02))
                        output = concat_2(output, yb, w)
                else:
                    w = tf.get_variable('Generator.w3', [6, 7, 8, y_dim, 15],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)

        output = lib.ops.conv3d.Deconv('Generator.4', output.get_shape().as_list()[-1], [BATCH_SIZE, 12, 13, 16, DIM],
                                       output)
        #output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0, 1, 2, 3], output)

        # 5th concat
        if (self.cond[4] == '1'):
            if (self.concat == '0'):
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

    def Discriminator(self, inputs, y=None, prop=None, Reuse=None):
        # default :"NDHWC"
        output = tf.reshape(inputs, [-1, 23, 26, 31, 1])

        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 1, y_dim])
        # 1st concat
        if (self.cond[4] == '1'):
            if (self.concat == '0'):
                output = conv_cond_concat(output, yb)
            else:

                # yb3_ = lib.ops.linear.Linear('Discriminator.yb3_', y_dim, 11 * 13 * 15 * 10, y)
                if Reuse:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        w = tf.get_variable('Discriminator.w4', [23, 26, 31, y_dim, 15])
                        output = concat_2(output, yb, w)
                else:
                    w = tf.get_variable('Discriminator.w4', [23, 26, 31, y_dim, 15],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)

        output = lib.ops.conv3d.Conv3D('Discriminator.1', output.get_shape().as_list()[-1], DIM, output, stride=2)
        output = self.LeakyReLU(output)

        # 2nd concat
        if (self.cond[3] == '1'):
            if (self.concat == '0'):
                output = conv_cond_concat(output, yb)
            else:
                '''
                yb2_ = lib.ops.linear.Linear('Discriminator.yb2_', y_dim, 6 * 7 * 8 * 3, y)
                yb2_ = tf.reshape(yb2_, [-1,  6, 7, 8, 3])
                yb2_ = tf.tanh(yb2_)
                output = concat([output, yb2_], 4)
                '''
                if Reuse:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=Reuse):
                        w = tf.get_variable('Discriminator.w3', [12, 13, 16, y_dim, 15],
                                            initializer=tf.truncated_normal_initializer(stddev=0.02))
                        output = concat_2(output, yb, w)
                else:
                    w = tf.get_variable('Discriminator.w3', [12, 13, 16, y_dim, 15],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
                    output = concat_2(output, yb, w)

        output = lib.ops.conv3d.Conv3D('Discriminator.2', output.get_shape().as_list()[-1], 2 * DIM, output, stride=2)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
        output = self.LeakyReLU(output)
        # 3rd concat
        # output = conv_cond_concat(output, yb)
        if (self.cond[2] == '1'):
            if (self.concat == '0'):
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

        output = lib.ops.conv3d.Conv3D('Discriminator.3', output.get_shape().as_list()[-1], 4 * DIM, output, stride=2)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
        output = self.LeakyReLU(output)

        if (self.cond[1] == '1'):
            if (self.concat == '0'):
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

        output = lib.ops.conv3d.Conv3D('Discriminator.4', output.get_shape().as_list()[-1], 8 * DIM, output, stride=2)

        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0, 2, 3], output)
        output = self.LeakyReLU(output)

        output = tf.reshape(output, [-1, 2 * 2 * 2 * 8 * DIM])

        if (self.cond[0] == '1'):
            output = concat([output, y], 1)
            output_shape = output.get_shape().as_list()
            output = lib.ops.linear.Linear('Discriminator.5', output_shape[-1], 128, output)

        # add one linear layer into the model
        #output = lib.ops.linear.Linear('Discriminator.6', output_shape[-1], 128, output)

        output = lib.ops.linear.Linear('Discriminator.Output', 128, 1, output)
        output = tf.reshape(output, [-1])
        # weighted
        # output = output* prop
        return output

    '''
    def Generator(self, n_samples, y=None, noise=None, Reuse = None):
        g_w = 2
        g_h = 2
        g_d = 2

        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        # yb = tf.reshape(y, [BATCH_SIZE, 1, 1, y_dim])
        # 1st concat
        # noise = concat([noise, y], 1)
        # noise_shape = noise.get_shape().as_list()

        output = lib.ops.linear.Linear('Generator.Input', 128, g_h * g_w * g_d * 4 * self.DIM, noise)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = self.LeakyReLU(output)
        output = tf.reshape(output, [-1, g_d, g_h, g_w, 4 * DIM])
        # NDHWC

        # 2nd concat

        output = lib.ops.conv3d.Deconv('Generator.2', output.get_shape().as_list()[-1], [self.BATCH_SIZE, 3, 4, 4, 2 * self.DIM],
                                       output)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
        output = self.LeakyReLU(output)

        # 3rd concat
        # output = conv_cond_concat(output, yb)

        # def Deconv(name, input_dim,output_shape, inputs,stride = 2):
        output = lib.ops.conv3d.Deconv('Generator.3', output.get_shape().as_list()[-1], [self.BATCH_SIZE, 6, 7, 8, self.DIM],
                                       output)

        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
        output = self.LeakyReLU(output)
        # 4th concat
        # output = conv_cond_concat(output, yb)
        output = lib.ops.conv3d.Deconv('Generator.5', self.DIM, [BATCH_SIZE, 11, 13, 15, 1], output)

        output = tf.nn.tanh(output)

        output = tf.reshape(output, [-1, OUTPUT_DIM])
        return output

    def Discriminator(self, inputs, y=None, Reuse = None):

        # default :"NDHWC"
        output = tf.reshape(inputs, [-1, 11, 13, 15, 1])
        # yb = tf.reshape(y, [BATCH_SIZE, 1, 1, y_dim])
        # 1st concat
        # output = conv_cond_concat(output, yb)

        output = lib.ops.conv3d.Conv3D('Discriminator.1', output.get_shape().as_list()[-1], self.DIM, output, stride=2)
        output = self.LeakyReLU(output)

        # 2nd concat
        # output = conv_cond_concat(output, yb)

        output = lib.ops.conv3d.Conv3D('Discriminator.2', output.get_shape().as_list()[-1], 2 * self.DIM, output, stride=2)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
        output = self.LeakyReLU(output)
        # 3rd concat
        # output = conv_cond_concat(output, yb)

        output = lib.ops.conv3d.Conv3D('Discriminator.3', output.get_shape().as_list()[-1], 4 * DIM, output, stride=2)
        if MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
        output = self.LeakyReLU(output)
        output = tf.reshape(output, [-1, 2 * 2 * 2 * 4 * DIM])

        # add one linear layer into the model
        output = lib.ops.linear.Linear('Discriminator.Output0', 2 * 2 * 2 * 4 * self.DIM, 128, output)

        output = lib.ops.linear.Linear('Discriminator.Output', 128, 1, output)
        # output = output* prop
        return tf.reshape(output, [-1])
    '''

    def inf_train_gen(self, train_gen):
        while True:
            for images, targets in train_gen():
                yield images, targets

    def generate_image(self, frame, fixed_noise_samples):
        samples = self.sess.run(fixed_noise_samples)
        count = 0
        for i in samples.reshape([BATCH_SIZE, 13, 15, 11]):
            # pkl.dump(i, outputFile)
            temp = lib.upsampling.upsample_vectorized(i)
            img = nibabel.Nifti1Image(temp.get_data(), temp.affine)
            filename = './{}/samples_{}_{}.nii.gz'.format(self.sample_dir, frame, count)
            nibabel.save(img, filename)
            count += 1
            if count == 10:
                break
    def save_test_img(self, frame):
        choose_pool = []
        for i in range(45):
            if (i not in self.tags_leave_out):
                choose_pool.append(i)

        ran = np.random.choice(choose_pool, BATCH_SIZE)
        test_labels = np.zeros((BATCH_SIZE, y_dim), dtype=np.float64)
        for i, label in enumerate(ran):
            test_labels[i, ran[i]] = 1.0

        test_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
        test_noise_samples = self.Generator(BATCH_SIZE, self.y, noise=test_noise, Reuse=True)

        samples = self.sess.run(test_noise_samples, feed_dict={ self.y: test_labels})

        count = 0
        for i in samples.reshape([BATCH_SIZE, 26, 31, 23]):
            temp = lib.upsampling.upsample_vectorized(i)
            img = nibabel.Nifti1Image(temp.get_data(), temp.affine)
            # save labels in test picture names
            id = list(test_labels[count]).index(1.)
            filename = './{}/test_{}_{}_{}.nii.gz'.format(self.test_dir, frame, count, id)
            nibabel.save(img, filename)
            count += 1

    @property
    def model_dir(self):
        return "{}_{}".format(
            self.dataset_name, self.iter)

    def save(self, checkpoint_dir, step):
        model_name = "CIWGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0






