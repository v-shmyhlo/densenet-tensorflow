import tensorflow as tf


def composite_function(x, filters, training, name='composite_function'):
  with tf.name_scope(name):
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters, (3, 3), (1, 1), padding='same')

    return x


def bottleneck_composite_function(x,
                                  filters,
                                  training,
                                  name='bottleneck_composite_function'):
  with tf.name_scope(name):
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters * 4, (1, 1), (1, 1), padding='same')

    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters, (3, 3), (1, 1), padding='same')

    return x


def dense_block(x,
                layers,
                growth_rate,
                bottleneck,
                training,
                name='dense_block'):
  with tf.name_scope(name):
    input_depth = x.shape[-1].value

    for i in range(1, layers + 1):
      assert x.shape[-1] == (i - 1) * growth_rate + input_depth

      if bottleneck:
        y = bottleneck_composite_function(
            x,
            growth_rate,
            training=training,
            name='bottleneck_composite_function_{}'.format(i))
      else:
        y = composite_function(
            x,
            growth_rate,
            training=training,
            name='composite_function_{}'.format(i))

      x = tf.concat([x, y], -1)

    return x


def transition_layer(x, compression_factor, training, name='transition_layer'):
  with tf.name_scope(name):
    filters = int(x.shape[-1].value * compression_factor)

    x = tf.layers.batch_normalization(x, training=training)
    x = tf.layers.conv2d(x, filters, (1, 1), (1, 1), padding='same')
    x = tf.layers.average_pooling2d(x, (2, 2), (2, 2), padding='same')

    return x


def input(x, filters, name='input'):
  with tf.name_scope(name):
    x = tf.layers.conv2d(x, filters, 3, 1, padding='same')

    return x


def output(x, name='output'):
  with tf.name_scope(name):
    x = tf.reduce_mean(x, (1, 2), keep_dims=True)
    x = tf.layers.conv2d(x, 1000, 1, 1, padding='same')

    return x


def densenet(x,
             bottleneck=True,
             compression_factor=0.5,
             growth_rate=12,
             training=False):
  with tf.name_scope('densenet'):
    if bottleneck and compression_factor < 1:
      x = input(x, growth_rate * 2)
    else:
      x = input(x, 16)

    x = dense_block(
        x,
        layers=16,
        growth_rate=growth_rate,
        bottleneck=bottleneck,
        training=training,
        name='dense_block_1')
    x = transition_layer(
        x,
        compression_factor=compression_factor,
        training=training,
        name='transition_layer_1')
    x = dense_block(
        x,
        layers=16,
        growth_rate=growth_rate,
        bottleneck=bottleneck,
        training=training,
        name='dense_block_2')
    x = transition_layer(
        x,
        compression_factor=compression_factor,
        training=training,
        name='transition_layer_2')
    x = dense_block(
        x,
        layers=16,
        growth_rate=growth_rate,
        bottleneck=bottleneck,
        training=training,
        name='dense_block_3')

    x = output(x)

    return x
