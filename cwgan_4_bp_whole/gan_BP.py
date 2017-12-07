# coding=utf-8
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




flags = tf.app.flags
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("class_id", "0", "classes")
flags.DEFINE_string("gpu", "3", "gpu_id")

FLAGS = flags.FLAGS
'''
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
'''

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

sample_dir = './37_multi_results/samples'+FLAGS.class_id + '_test'
test_dir = './37_multi_results/test'+FLAGS.class_id + '_test'
cost_dir = './37_multi_results/cost'+FLAGS.class_id + '_test'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(cost_dir):
    os.makedirs(cost_dir)



MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 1 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 2145 # Number of pixels in MNIST (28*28)
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
    x = tf.transpose(x, [0,2,3,1])
    x_shapes = x.get_shape().as_list()
    x = concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    return tf.transpose(x, [0,3,1,2])

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

def Generator(n_samples, y = None, noise = None):
    g_w = 2
    g_h = 2
    g_d = 2

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    # yb = tf.reshape(y, [BATCH_SIZE, 1, 1, y_dim])
    # 1st concat
    #noise = concat([noise, y], 1)
    #noise_shape = noise.get_shape().as_list()

    output = lib.ops.linear.Linear('Generator.Input', 128, g_h * g_w * g_d * 4 * DIM, noise)
    output = LeakyReLU(output)
    output = tf.reshape(output, [-1, g_d, g_h, g_w,  4 * DIM])
    #NDHWC
    # 2nd concat
    #output = conv_cond_concat(output, yb)
    output = lib.ops.conv3d.Deconv('Generator.2', output.get_shape().as_list()[-1], [BATCH_SIZE, 3, 4, 4,  2 * DIM], output)
    output = LeakyReLU(output)
    # 3rd concat
    #output = conv_cond_concat(output, yb)
    output = lib.ops.conv3d.Deconv('Generator.3', output.get_shape().as_list()[-1], [BATCH_SIZE, 6, 7, 8, DIM], output)
    output = LeakyReLU(output)
    #4th concat
    #output = conv_cond_concat(output, yb)
    output = lib.ops.conv3d.Deconv('Generator.5', DIM, [BATCH_SIZE, 11, 13, 15,  1], output)
    output = tf.nn.tanh(output)
    output = tf.reshape(output, [-1, OUTPUT_DIM])
    return output

def Discriminator(inputs, y = None):

    # default :"NDHWC"
    output = tf.reshape(inputs, [-1, 11, 13, 15, 1])
    # yb = tf.reshape(y, [BATCH_SIZE, 1, 1, y_dim])
    # 1st concat
    #output = conv_cond_concat(output, yb)

    output = lib.ops.conv3d.Conv3D('Discriminator.1',output.get_shape().as_list()[-1], DIM, output, stride=2)
    output = LeakyReLU(output)

    # 2nd concat
    #output = conv_cond_concat(output, yb)
    output = lib.ops.conv3d.Conv3D('Discriminator.2', output.get_shape().as_list()[-1], 2*DIM, output, stride=2)
    # 3rd concat
    # output = conv_cond_concat(output, yb)
    output = lib.ops.conv3d.Conv3D('Discriminator.3', output.get_shape().as_list()[-1], 4*DIM,  output, stride=2)
    output = LeakyReLU(output)
    output = tf.reshape(output, [-1, 2*2*2*4*DIM])

    # add one linear layer into the model
    output = lib.ops.linear.Linear('Discriminator.Output0', 2 * 2 * 2 * 4 * DIM, 128, output)

    output = lib.ops.linear.Linear('Discriminator.Output', 128, 1, output)

    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

if y_dim:
    y = tf.placeholder(tf.float32, [BATCH_SIZE, y_dim], name='y')
else:
    y = None

fake_data = Generator(BATCH_SIZE, y)

disc_real = Discriminator(real_data, y)
disc_fake = Discriminator(fake_data, y)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

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
    # print('[INFO] Add a linear layer')
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )

    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)

    gradients = tf.gradients(Discriminator(interpolates, y), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

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

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)

    # 1e-4,
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=0.0001,
        beta1=0.5,
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

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
        learning_rate=1e-4,
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples

train_gen,dev_gen = lib.BrainPedia.load_BrainPedia_test(BATCH_SIZE,FLAGS) # load_mnist(BATCH_SIZE)#

fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
fixed_noise_samples = Generator(BATCH_SIZE, noise=fixed_noise)

def generate_image(frame, true_dist, FLAGS):
    samples = session.run(fixed_noise_samples)#,feed_dict={y:fixed_labels}

    count = 0
    for i in samples.reshape([BATCH_SIZE, 13, 15, 11]):
        # pkl.dump(i, outputFile)
        temp = lib.upsampling.upsample_vectorized(i)
        img = nibabel.Nifti1Image(temp.get_data(), temp.affine)
        filename = './{}/samples{}_{}_{}.nii.gz'.format(sample_dir, FLAGS.class_id, frame, count)
        nibabel.save(img, filename)
        count += 1
        if count == 20:
            break

def save_test_img(frame, FLAGS):
    test_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    test_noise_samples = Generator(BATCH_SIZE, noise=test_noise)

    samples = session.run(test_noise_samples)

    count = 0
    for i in samples.reshape([BATCH_SIZE, 13, 15, 11]):

        temp = lib.upsampling.upsample_vectorized(i)
        img = nibabel.Nifti1Image(temp.get_data(), temp.affine)
        filename = './{}/test_{}_{}_{}.nii.gz'.format(test_dir, FLAGS.class_id, frame, count)
        nibabel.save(img, filename)
        count += 1

def inf_train_gen():

    while True:
        #count = 0
        for images,targets in train_gen():
            #print(count)
            #print(images[0])
            #count += 1
            yield images, targets

# Train loop
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _, g_loss = session.run([gen_train_op,gen_cost])#,feed_dict={y: targets})
            print('iter:%d,  d_loss: %.4f,  g_loss: %.4f'%(iteration, _disc_cost, g_loss))
            lib.plot.plot(cost_dir+'/train gen cost', g_loss)
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS

        for i in xrange(5):
            _data,targets = next(gen)
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data, y: targets}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)
            #print("{INFO}d_loss",_disc_cost)

        if(iteration > 0):
            lib.plot.plot(cost_dir +'/train disc cost', _disc_cost)
            lib.plot.plot(cost_dir +'/time', time.time() - start_time)


        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 0 and iteration != 0:

            dev_disc_costs = []
            dev_gen_costs = []
            
            for images, tags in dev_gen(): # image and targets
                _dev_disc_cost, _dev_g_loss = session.run(
                    [disc_cost,gen_cost],
                    feed_dict={real_data: images, y: tags}
                )
                dev_disc_costs.append(_dev_disc_cost)
                dev_gen_costs.append(_dev_g_loss)
            lib.plot.plot(cost_dir+ '/dev disc cost', np.mean(dev_disc_costs))
            lib.plot.plot(cost_dir + '/dev gen cost', np.mean(dev_gen_costs))

            #if(iteration %400 == 0 and iteration != 0 ):
                #generate_image(iteration, _data, FLAGS)



        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush(cost_dir)
        lib.plot.tick()

        #if (iteration > 3000):#11000
            #save_test_img(iteration, FLAGS)