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
import tflib.ops.deconv2d
import tflib.save_images
import tflib.BrainPedia

import tflib.plot
from six.moves import xrange
import nibabel
import nilearn.masking as masking



MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 1 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
y_dim = 10
filter_size = [1, 32]
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
    g_w =98 #46
    g_h = 1

    if noise is None:
        noise = tf.random_normal([n_samples, 10])

    yb = tf.reshape(y, [BATCH_SIZE,  1, 1, y_dim])
    # 1st concat
    noise = concat([noise, y], 1)
    noise_shape = noise.get_shape().as_list()
    # input_dim（shape[1]）, output_dim(shape[1])

    output = lib.ops.linear.Linear('Generator.Input0', noise_shape[1], g_w, noise)

    output = lib.ops.linear.Linear('Generator.Input', g_w, g_h * g_w * 4 * DIM, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = LeakyReLU(output)
    output = tf.reshape(output, [-1, 4*DIM, 1, g_w])
    # 2nd concat
    #output = conv_cond_concat(output,yb)
    output = lib.ops.deconv2d.Deconv2D('Generator.2', output.get_shape().as_list()[1], [BATCH_SIZE, g_h, 196, 2 * DIM], filter_size, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = LeakyReLU(output)
    #output = output[:, :, :7, :7]
    # 3rd concat
    output = conv_cond_concat(output, yb)
    # 2 * DIM
    output = lib.ops.deconv2d.Deconv2D('Generator.3', output.get_shape().as_list()[1], [BATCH_SIZE, g_h, 392, DIM], filter_size, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, [BATCH_SIZE, g_h, 784, 1], filter_size, output) # 364*50 = 18200
    output = tf.nn.tanh(output)
    #print('g_shape3:', output)

    output = tf.reshape(output, [-1, OUTPUT_DIM])
    return output

def Discriminator(inputs, y = None):
    output = tf.reshape(inputs, [-1, 1, 1, 784])

    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, y_dim])
    # 1st concat
    output = conv_cond_concat(output, yb)

    # 369
    output = lib.ops.conv2d.Conv2D('Discriminator.1', output.get_shape().as_list()[1], DIM, filter_size, output, stride=2)
    output = LeakyReLU(output)
    #print('d_shape1:', output)
    # 2nd concat
    output = conv_cond_concat(output, yb)
    output = lib.ops.conv2d.Conv2D('Discriminator.2', output.get_shape().as_list()[1], 2*DIM, filter_size, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = LeakyReLU(output)
    # 3rd concat
    output = conv_cond_concat(output, yb)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', output.get_shape().as_list()[1], 4*DIM, filter_size, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = LeakyReLU(output)
    output = tf.reshape(output, [-1, 1 * 98 * 4 * DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 1 * 98 * 4 * DIM, 10, output)
    output = lib.ops.linear.Linear('Discriminator.Output0', 10, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

if y_dim:
    y = tf.placeholder(tf.float32, [BATCH_SIZE, y_dim], name='y')
else:
    y = None

fake_data = Generator(BATCH_SIZE,y)

disc_real = Discriminator(real_data,y)
disc_fake = Discriminator(fake_data,y)

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
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates,y), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    learning_rate = tf.train.exponential_decay(
        learning_rate=2e-4,
        global_step=global_step,
        decay_steps=3000,
        decay_rate=0.5,
        staircase=True)

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.GradientDescentOptimizer(
        learning_rate=1e-4
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

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
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

ran = np.random.choice(10, BATCH_SIZE)
fixed_labels = np.zeros((BATCH_SIZE,y_dim), dtype=np.float32)
for i, label in enumerate(ran):
    fixed_labels[i, ran[i]] = 1.0
#print('label:', ran)

fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 10)).astype('float32'))
fixed_noise_samples = Generator(BATCH_SIZE, fixed_labels, noise=fixed_noise)

def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((BATCH_SIZE, 28, 28)),
        'samples_{}.png'.format(frame)
    )

# Dataset iterator
train_gen = lib.BrainPedia.load_mnist(BATCH_SIZE)

def inf_train_gen():

    while True:
        for images, targets in train_gen():
            yield images, targets

# Train loop
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            # for i in xrange(5):
            _,g_loss = session.run([gen_train_op,gen_cost], feed_dict={y: targets})
            print('iter:%d,  d_loss: %.4f,  g_loss: %.4f'%(iteration, _disc_cost, g_loss))
            lib.plot.plot('train gen cost', g_loss)
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS

        for i in xrange(5):
            _data,targets = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data, y: targets}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 200 == 0:
            '''
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            '''
            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()
        lib.plot.tick()