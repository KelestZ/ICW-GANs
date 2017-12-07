#-*- coding : utf-8

import tensorflow as tf
import tflib as lib
import numpy as np
# Have not fixed some initializations.

# he_init = True,weightnorm = None,biases = True,\gain = 1.,mask_type = None,
_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None



def Deconv(name, input_dim, output_shape, inputs, stride = 2,he_init=True,
           weightnorm=None, biases=True, gain=1., mask_type=None):
    with tf.name_scope(name) as scope:
        output_dim = output_shape[-1]
        if output_dim is None:
            output_dim = input_dim / 2
        '''
        return tl.layers.DeConv3dLayer(layer=inputs,
                                    shape = [4, 4, 4, input_dim, output_dim],
                                    output_shape = output_shape,
                                    strides=[1, stride, stride, stride, 1],
                                    W_init = tf.random_normal_initializer(stddev=0.02),
                                    act=tf.identity, name= name)
        '''

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        filter_size = [4, 4, 4]

        fan_in = input_dim * filter_size[0] * filter_size[1] * filter_size[2] * filter_size[2] / (
        stride ** 2)  # filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size[0] * filter_size[1] * filter_size[2]  # filter_size**2

        if he_init:
            filters_stdev = np.sqrt(4. / (fan_in + fan_out))
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2. / (fan_in + fan_out))

        filter_values = uniform(
            filters_stdev,
            (filter_size[0], filter_size[1], filter_size[2], output_dim, input_dim)
        )
        filters = lib.param(
            name + '.Filters',
            filter_values
        )
        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2,3)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2,3]))
                filters = filters * (target_norms / norms)

        result = tf.nn.conv3d_transpose(value=inputs,
                                        filter=filters,
                                        output_shape=output_shape,
                                        strides=[1, 2, 2, 2, 1],
                                        padding='SAME',
                                        name=name,
                                        data_format="NDHWC"
                                        )#
        # data_format="NDHWC"
        '''
        if biases:
            _biases = lib.param(
                name + '.Biases',
                np.zeros(output_dim, dtype='float32')
            )
            result = tf.nn.bias_add(result, _biases)
        '''
        return result


def Conv3D(name, input_dim, output_dim, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True): #filter_size,
    with tf.name_scope(name) as scope:
        if input_dim is None:
            input_dim = output_dim / 2

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        filter_size = [4, 4, 4]

        fan_in = input_dim * filter_size[0] * filter_size[1] * filter_size[2] * filter_size[2] / (
            stride ** 2)  # filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size[0] * filter_size[1] * filter_size[2]  # filter_size**2

        if he_init:
            filters_stdev = np.sqrt(4. / (fan_in + fan_out))
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2. / (fan_in + fan_out))

        filter_values = uniform(
            filters_stdev,
            (filter_size[0], filter_size[1], filter_size[2], input_dim, output_dim )
        )
        filters = lib.param(
            name + '.Filters',
            filter_values
        )


        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2,3)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2,3]))
                filters = filters * (target_norms / norms)

        result = tf.nn.conv3d(input=inputs,
                              filter=filters,
                              strides=[1, stride, stride, stride, 1],
                              padding="SAME",
                              name=name)
        '''
        if biases:
            _biases = lib.param(
                name + '.Biases',
                np.zeros(output_dim, dtype='float32')
            )

            result = tf.nn.bias_add(result, _biases, data_format='NDHWC')
        '''
        return result


def Conv3D_layer(name, input_dim, output_dim, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True):


   layer =  tf.layers.conv3d(inputs,
                             filters = [4,4,4,input_dim,output_dim],
                             kernel_size=[4,4,4],
                             strides=(2, 2, 2),
                             padding='same',
                             data_format='channels_last',
                             dilation_rate=(1, 1, 1),
                             activation=None,
                             use_bias=True,
                             kernel_initializer=None,
                             bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=True,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             trainable=True,
                             name=name,
                             reuse=None)