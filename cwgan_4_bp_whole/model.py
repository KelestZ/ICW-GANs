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
OUTPUT_DIM = 2145 # Number of pixels in MNIST (28*28)
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
    x = tf.transpose(x, [0,2,3,1])
    x_shapes = x.get_shape().as_list()
    x = concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    return tf.transpose(x, [0,3,1,2])


class CIWGAN(object):
    def __init__(self, sess, MODE='wgan-gp', DIM=64, BATCH_SIZE =50, sample_num = 50, LAMBDA = 10, ITERS =200000,
                 checkpoint_dir=None, sample_dir=None, y_dim=45
                 ):

        self.sess = sess
        self.MODE = MODE
        self.DIM = DIM
        self.BATCH_SIZE = BATCH_SIZE
        self.CRITIC_ITERS = 5  # For WGAN and WGAN-GP, number of critic iters per gen iter
        self.LAMBDA = LAMBDA  # Gradient penalty lambda hyperparameter
        self.y_dim = y_dim
        self.ITERS = ITERS
        self.data_size
        self.checkpoint_dir = checkpoint_dir

        self.build_model()
    def build_model(self):

        self.real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

        if y_dim:
            self.y = tf.placeholder(tf.float32, [BATCH_SIZE, y_dim], name='y')
        else:
            self.y = None

        self.fake_data = self.Generator(self.BATCH_SIZE, self.y)

        self.disc_real = self.Discriminator(self.real_data, self.y)
        self.disc_fake = self.Discriminator(self.fake_data, self.y)

        gen_params = lib.params_with_name('Generator')
        disc_params = lib.params_with_name('Discriminator')

        if MODE=='wgan-gp':
            # print('[INFO] Add a linear layer')
            gen_cost = -tf.reduce_mean(self.disc_fake)
            disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE, 1],
                minval=0.,
                maxval=1.
            )
            differences = self.fake_data - self.real_data

            interpolates = self.real_data + (alpha * differences)

            gradients = tf.gradients(self.Discriminator(interpolates, self.y), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            disc_cost += LAMBDA * gradient_penalty

            global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            learning_rate = tf.train.exponential_decay(
                learning_rate=0.0002,
                global_step=global_step,
                decay_steps=3000,
                decay_rate=0.5,
                staircase=True)

            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.5,
                beta2=0.9
            ).minimize(gen_cost, var_list=gen_params)
            self.disc_train_op = tf.train.GradientDescentOptimizer(
                learning_rate=1e-4
            ).minimize(disc_cost, var_list=disc_params)

            self.clip_disc_weights = None
            # Set up the Saver for saving and restoring model checkpoints.
            saver = tf.train.Saver()

    def train(self):
        train_gen = lib.BrainPedia.load_BrainPedia_test(BATCH_SIZE)  # load_mnist(BATCH_SIZE)#

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # fixed_labels are not written
        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
        fixed_noise_samples = self.Generator(BATCH_SIZE, noise=fixed_noise)


        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        gen = self.inf_train_gen(train_gen)

        for iteration in xrange(ITERS):
            start_time = time.time()

            if iteration > 0:
                # for i in xrange(3):
                _, g_loss = self.sess.run([self.gen_train_op, self.gen_cost])  # ,feed_dict={y: targets})
                print('iter:%d,  d_loss: %.4f,  g_loss: %.4f' % (iteration, _disc_cost, g_loss))
                lib.plot.plot('train gen cost', g_loss)
            if MODE == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = self.CRITIC_ITERS

            for i in xrange(disc_iters):
                _data, targets = next(gen)
                _disc_cost, _ = self.sess.run.run(
                    [self.disc_cost, self.disc_train_op],
                    feed_dict={self.real_data: _data, self.y: targets}
                )
                if self.clip_disc_weights is not None:
                    _ = self.sess.run.run(self.clip_disc_weights)

            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 0 and iteration != 0:
                self.generate_image(iteration, _data, fixed_noise_samples)

            # Write logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()
            lib.plot.tick()

            if np.mod(iteration, 500) == 2:
                print('[INFO] Save checkpoint...')
                self.save(self.checkpoint_dir, iteration)

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

    def LeakyReLULayer(name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(
            name + '.Linear',
            n_in,
            n_out,
            inputs,
            initialization='he'
        )
        return LeakyReLU(output)

    def Generator(self, n_samples, y=None, noise=None):
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
        # output = conv_cond_concat(output, yb)

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

    def Discriminator(self, inputs, y=None):

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

        return tf.reshape(output, [-1])

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
            filename = './{}/samples_{}_{}.nii.gz'.format('samples', frame, count)
            nibabel.save(img, filename)
            count += 1
            if count == 4:
                break


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
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






