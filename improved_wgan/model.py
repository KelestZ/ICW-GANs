# -*- coding: utf-8 -*-

from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from sklearn.utils import shuffle
from ops import *
from utils import *
import pickle as pkl
import nibabel
import nilearn.masking as masking
from upsampling import *
import tensorflow.contrib.layers as ly

base_dir = '/home/nfs/zpy/BrainPedia/pkl/'
mskFile = open(base_dir + 'msk.pkl', 'rb')

LAMBDA = 10# grand penalty 的参数
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class WGAN(object):
    '''
    def __init__(self, sess, input_height=1, input_width=737, crop=True,
                 batch_size=64, sample_num=64, output_height=1, output_width=737,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.nii', checkpoint_dir=None, sample_dir=None, change=False):
    '''
    def __init__(self, sess, input_height=1, input_width=737, crop=True,
                     batch_size=64, sample_num=64, output_height=1, output_width=737,
                     y_dim=10, z_dim=100, gf_dim=64, df_dim=64,
                     gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='BrainPedia',
                     input_fname_pattern='*.nii', checkpoint_dir=None, sample_dir=None, change=False):

        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]

          # input_f_pattern =什么不重要
          change = True了

          sample_num 是batch size
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.change = change


        # 没有d_bn0
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        if self.change == True:
            self.d_bn3 = batch_norm(name='d_bn3')
            self.d_bn4 = batch_norm(name='d_bn4')
            self.d_bn5 = batch_norm(name='d_bn5')

            self.g_bn3 = batch_norm(name='g_bn3')
            self.g_bn4 = batch_norm(name='g_bn4')
            self.g_bn5 = batch_norm(name='g_bn5')


        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.c_dim = 1#self.data_X[0].shape[-1]

        elif self.dataset_name == 'BrainPedia':

            imageFile = base_dir + 'vec_brain.pkl'
            labelFile = base_dir + 'multi_class_pic_tags.pkl'
            self.data_X, self.data_y = self.load_BrainPedia(imageFile, labelFile)

            self.c_dim = 1

        # self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.input_size = self.input_height*self.input_width*self.c_dim
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size,self.input_height*self.input_width], name='real_images')

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        #self.z_sum = histogram_summary("z", self.z)

        #####从这里开始，代码没有改
        self.G = self.generator(self.z, self.y)
        self.D = self.discriminator(self.inputs, self.y, reuse=False) # , self.D_logits
        self.sampler = self.sampler(self.z, self.y)
        self.D_ = self.discriminator(self.G, self.y, reuse=True) # , self.D_logits


        #self.d_sum = histogram_summary("d", self.D)
        #self.d__sum = histogram_summary("d_", self.D_)
        #self.G_sum = image_summary("G", self.G)
        '''
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        '''
        # improved wgan前半部分的期望
        self.d_loss_real = tf.reduce_mean(self.D)
        self.d_loss_fake = tf.reduce_mean(self.D_)

        self.g_loss = tf.reduce_mean(self.D_)
        self.d_loss = self.d_loss_real - self.d_loss_fake

        #self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        #self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        #self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        #self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

        # Gradient penalty
        # 这不对啊
        self.alpha = tf.random_uniform(
            shape=[],
            minval=0.,
            maxval=1.
        )
        differences = self.G - self.inputs
        self.interpolates = self.inputs + self.alpha * differences
        gradients = tf.gradients(self.discriminator(self.interpolates,self.y),[self.interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1])) #平方误差和

        print('shape',gradients.get_shape().as_list())

        gradient_penalty = tf.reduce_mean(LAMBDA * ((slopes - 1.) ** 2))
        self.d_loss +=  gradient_penalty


    def train(self, config):

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=0.9) \
            .minimize(self.d_loss, var_list=self.d_vars)#, beta1=config.beta1, beta2=0.9
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=0.9) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        #self.g_sum = merge_summary([self.z_sum, self.d__sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        #self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        #self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        if config.dataset == 'mnist' or config.dataset == 'BrainPedia':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist' or config.dataset == 'BrainPedia':
                batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist' or config.dataset == 'BrainPedia':
                    batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
                    print("[INFO] batch image shape ",batch_images.shape)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                if config.dataset == 'mnist' or config.dataset == 'BrainPedia':
                    # Update D network
                    _= self.sess.run([d_optim],feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                   } )
                    #self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _ = self.sess.run([g_optim],feed_dict={
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                   })
                    #self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #_ = self.sess.run([g_optim],feed_dict={self.z: batch_z, self.y: batch_labels})
                    #self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels
                    })
                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )

                        sp = np.reshape(samples,[self.batch_size,self.output_height,self.output_width,1])
                        save_images(sp, image_manifold_size(sp.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    ####  改动的地方  ####
                    elif config.dataset == 'BrainPedia':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                        ct = 0
                        for i in samples.reshape((self.batch_size, -1)):
                            # pkl.dump(i, outputFile)
                            #print('[INFO] pic1 ', i)

                            temp = upsample_vectorized(i)
                            #print('[INFO] pic2 ',temp)
                            # 不加这句话不行
                            #img = nibabel.Nifti1Image(temp.get_data(), temp.affine)

                            filename = './{}/train_{:02d}_{:04d}_{:02d}.nii.gz'.format(config.sample_dir, epoch, idx,
                                                                                       ct)
                            nibabel.save(temp, filename)
                            ct += 1
                        print('[INFO] Save sample images in Epoch', epoch, 'index ', idx, )  # result

                    if np.mod(counter, 500) == 2:
                        self.save(config.checkpoint_dir, counter)

    def load_BrainPedia(self, imageFile, labelFile):
        x = []
        y = []
        count = 0
        print('[INFO] Load BrainPedia dataset...')

        imgpkl = pkl.load(open(imageFile, 'rb'))
        labelpkl = pkl.load(open(labelFile, 'rb'))
        print('[INFO]',len(imgpkl),'images in all')

        for i in labelpkl.keys():
            x.append(imgpkl[i])
            y.append(labelpkl[i])
            count += 1

        # 4维减少成了3维了
        x, y = shuffle(np.array(x).reshape((6573, -1)).astype(np.float64),
                       np.array(y).reshape(6573).astype(np.int))

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float64)

        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        print('[INFO] Finish loading %d images' % count)
        return x, y_vec

    def critic_mlp(self, img, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.reshape(img, [self.batch_size,-1])
            x = concat([x, y], 1)

            # 一层 linear 1024
            h0 = lrelu(self.d_bn1(linear(x, 1024, 'd_h0_lin')))
            h0 = tf.reshape(h0, [self.batch_size, -1])
            h0 = concat([h0, y], 1)

            # 两层 linear  64
            h1 = lrelu( self.d_bn2(linear(h0, 128, 'd_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = concat([h1, y], 1)
            '''
            # 三层 linear 64
            h2 = lrelu( self.d_bn3(linear(h1, self.df_dim, 'd_h2_lin')))
            h2 = tf.reshape(h2, [self.batch_size, -1])
            h2 = concat([h2, y], 1)
            '''
            # 四层 linear 1
            h3 = linear(h1, 1, 'd_h3_lin')
            '''
             with tf.variable_scope('critic') as scope:
                if reuse:
                    scope.reuse_variables()
                img = ly.fully_connected(tf.reshape(
                    img, [self.batch_size, -1]), self.df_dim*4, activation_fn=tf.nn.relu)
                img = ly.fully_connected(img, self.df_dim*2,
                                         activation_fn=tf.nn.relu)
                img = ly.fully_connected(img, self.df_dim,
                                         activation_fn=tf.nn.relu)
                logit = ly.fully_connected(img, 1, activation_fn=None
            '''

            return tf.nn.sigmoid(h3), h3
    def generate_mlp(self,z, y=None):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width

            # 一层 linear
            z = concat([z, y], 1)
            h0 = tf.nn.relu(self.g_bn0(linear(z, 128 , 'g_h0_lin')))
            h0 = tf.reshape(h0, [self.batch_size, -1])
            h0 = concat([h0, y], 1)

            # 两层 linear

            h1 = tf.nn.relu(self.g_bn1(linear(h0, 1024, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = concat([h1, y], 1)
            '''
            # 三层 linear
            h2 = tf.nn.relu(self.g_bn2(linear(h1, self.gf_dim, 'g_h2_lin')))
            h2 = tf.reshape(h2, [self.batch_size, -1])
            h2 = concat([h2, y], 1)
            '''

            # 四层 linear
            h3 = linear(h1,  s_h * s_w * self.c_dim, 'g_h3_lin') #self.batch_size *
            h4 = tf.reshape(h3, [self.batch_size, s_h, s_w, self.c_dim])

            return tf.nn.sigmoid(h4)
    def sampler_mlp(self,z,y= None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width

            # 一层 linear
            z = concat([z, y], 1)
            h0 = tf.nn.relu(self.g_bn0(linear(z, 128, 'g_h0_lin')))
            h0 = tf.reshape(h0, [self.batch_size, -1])
            h0 = concat([h0, y], 1)

            # 两层 linear

            h1 = tf.nn.relu(self.g_bn1(linear(h0, 1024, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = concat([h1, y], 1)
            '''
            # 三层 linear
            h2 = tf.nn.relu(self.g_bn2(linear(h1, self.gf_dim, 'g_h2_lin')))
            h2 = tf.reshape(h2, [self.batch_size, -1])
            h2 = concat([h2, y], 1)
            '''

            # 四层 linear
            h3 = linear(h1, s_h * s_w * self.c_dim, 'g_h3_lin')  # self.batch_size *
            h4 = tf.reshape(h3, [self.batch_size, s_h, s_w, self.c_dim])

            return tf.nn.sigmoid(h4)


    def discriminator(self, image, y=None, reuse=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4

            ###### 根据数据大小加层！层数按照上面的代码来 4卷积，1线性 ######
            # 如果不改会怎么样？
            elif self.change == False:
                img = tf.reshape(image,[self.batch_size, self.input_height, self.input_width, self.c_dim])

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(img, yb)
                # 一层 conv2d
                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)
                # 两层 conv2d + 1 batch_norm
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, y], 1)
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)
                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3)
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)
                # 一层 conv2d （无BN）
                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                # 两层 conv2d
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))

                # 三层 conv2d
                h1 = conv_cond_concat(h1, yb)
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 2 + self.y_dim, name='d_h2_conv')))

                # 四层 conv2d
                h2 = conv_cond_concat(h2, yb)
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 4 + self.y_dim, name='d_h3_conv')))

                # 五层 conv2d
                h3 = conv_cond_concat(h3, yb)
                h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim * 8 + self.y_dim, name='d_h4_conv')))

                # 一层 linear
                h4 = tf.reshape(h4, [self.batch_size, -1])
                h5 = lrelu(self.d_bn5(linear(h4, self.dfc_dim, 'd_h5_lin')))
                h5 = concat([h5, y], 1)

                # 两层 linear
                h6 = linear(h5, 1, 'd_h6_lin')

                return tf.nn.sigmoid(h6), h6

    # 没改完 #
    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])

                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4',
                                                    with_w=True)

                return tf.nn.tanh(h4)

            elif self.change == False:
                s_h, s_w = self.output_height, self.output_width
                #s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                #s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                h3 = tf.nn.tanh(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

                return tf.reshape(h3,[self.batch_size,-1])#sigmoid
            ###### 根据数据大小加层！层数按照上面的代码来 ######
            else:

                s_h, s_w = self.output_height, self.output_width

                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

                # keep in mind: 选择用原z 还是 self.z_ ##########
                z = concat([z, y], 1)

                # 一层 linear  z
                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                # 两层 linear
                h1 = tf.nn.relu(self.g_bn1(linear(h0, s_h16 * s_w16 * self.gf_dim * 8, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h16, s_w16, self.gf_dim * 8])
                h1 = conv_cond_concat(h1, yb)

                # 一层 deconv2d
                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                # 二层 deconv2d
                h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h3')))
                h3 = conv_cond_concat(h3, yb)

                # 三层 deconv2d
                h4 = tf.nn.relu(self.g_bn4(deconv2d(h3, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h4')))
                h4 = conv_cond_concat(h4, yb)

                # 四层 deconv2d 无BN
                h5 = deconv2d(h4, [self.batch_size, s_h, s_w, self.c_dim], name='g_h5')

                #self.test = str(int(self.test)+1)
                return tf.nn.tanh(h5)
    # 没改完  sampler 内部的网络结构要跟generator相同 #
    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            elif self.change == False:
                s_h, s_w = self.output_height, self.output_width
                # s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                # s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                '''
                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                # yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                # z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))

                # h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                # h1 = conv_cond_concat(h1, yb)
                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                # h2 = conv_cond_concat(h2, yb)

                h3 = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')
                '''
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                h3 = tf.nn.tanh(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
                return tf.reshape(h3, [self.batch_size, -1])

            else:
                s_h, s_w = self.output_height, self.output_width

                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

                # keep in mind: 选择用原z 还是 self.z_ ##########
                z = concat([z, y], 1)

                # 一层 linear  z
                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                # 两层 linear
                h1 = tf.nn.relu(self.g_bn1(linear(h0, s_h16 * s_w16 * self.gf_dim * 8, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h16, s_w16, self.gf_dim * 8])
                h1 = conv_cond_concat(h1, yb)

                # 一层 deconv2d
                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                # 二层 deconv2d
                h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h3')))
                h3 = conv_cond_concat(h3, yb)

                # 三层 deconv2d
                h4 = tf.nn.relu(self.g_bn4(deconv2d(h3, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h4')))
                h4 = conv_cond_concat(h4, yb)

                # 四层 deconv2d 无BN
                h5 = deconv2d(h4, [self.batch_size, s_h, s_w, self.c_dim], name='g_h5')

                return tf.nn.tanh(h5)

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.reshape(np.concatenate((trX, teX), axis=0),[70000,-1])
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0
        print('[INFO] y_vec shape:',y_vec.shape)
        print('[INFO] x shape:',X.shape)
        test_x = []
        test_y = []
        x1 = X[0]
        y1 = y_vec[0]
        #return X / 255., y_vec
        print("[INFO] Training y is ", y1)
        #print("[INFO] Save training data X")
        #scipy.misc.imsave('./samples/training_data.png', np.squeeze(x1))
        for i in range(60000):
            test_x.append(x1)
            test_y.append(y1)

        test_x = np.array(test_x).reshape((60000, -1))
        test_y = np.array(test_y).reshape((60000, self.y_dim))

        return 2*((test_x / 255.)-.5), test_y

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

