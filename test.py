def Generator(n_samples, y=None, noise=None):
    d_w = 46
    if noise is None:
        noise = tf.random_normal([n_samples, d_w])

    yb = tf.reshape(y, [BATCH_SIZE, y_dim, 1, 1])
    # 1st concat
    noise = concat([noise, y], 1)

    # input_dim（shape[1]）, output_dim(shape[1])
    output = lib.ops.linear.Linear('Generator.Input', d_w, 1 * d_w * 4 * DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4 * DIM, 1, 46])

    # 2nd concat
    output = conv_cond_concat(output, yb)

    # input_dim（depth）, output_dim (depth), filter_size, inputs,
    # I should change output_din into a complete shape
    # 4 * DIM
    output = lib.ops.deconv2d.Deconv2D('Generator.2', output.get_shape()[1], [BATCH_SIZE, 1, 2 * d_w, 2 * DIM], [1, 32],
                                       output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = output[:, :, :1, :91]

    # 3rd concat
    output = conv_cond_concat(output, yb)
    # 2 * DIM
    output = lib.ops.deconv2d.Deconv2D('Generator.3', output.get_shape()[1], [BATCH_SIZE, 1, 2 * 91, DIM], [1, 32],
                                       output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, [BATCH_SIZE, 1, 4 * 91, 1], [1, 32],
                                       output)  # 364*50 = 18200
    output = tf.nn.sigmoid(output)

    output = tf.reshape(output, [-1, OUTPUT_DIM])
    return output


def Discriminator(inputs, y=None):
    output = tf.reshape(inputs, [-1, 1, 1, 364])

    yb = tf.reshape(y, [BATCH_SIZE, y_dim, 1, 1])
    # 1st concat
    output = conv_cond_concat(output, yb)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 1, DIM, [1, 32], output, stride=2)
    output = LeakyReLU(output)

    # 2nd concat
    output = conv_cond_concat(output, yb)

    # 1，182
    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2 * DIM, [1, 32], output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = LeakyReLU(output)

    # 3rd concat
    output = conv_cond_concat(output, yb)

    # 1，91
    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * DIM, 4 * DIM, [1, 32], output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = LeakyReLU(output)
    # 1，46
    output = tf.reshape(output, [-1, 1 * 46 * 4 * DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 1 * 46 * 4 * DIM, 1, output)

    return tf.reshape(output, [-1])
