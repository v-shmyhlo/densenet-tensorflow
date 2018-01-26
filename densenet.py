import tensorflow as tf


def composite_function(x,
                       filters,
                       dropout,
                       initializer,
                       regularizer,
                       training,
                       name='composite_function'):
  with tf.name_scope(name):
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        x,
        filters, (3, 3), (1, 1),
        padding='same',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    x = tf.layers.dropout(x, rate=dropout, training=training)

    return x


def bottleneck_composite_function(x,
                                  filters,
                                  dropout,
                                  initializer,
                                  regularizer,
                                  training,
                                  name='bottleneck_composite_function'):
  with tf.name_scope(name):
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        x,
        filters * 4, (1, 1), (1, 1),
        padding='same',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    x = tf.layers.dropout(x, rate=dropout, training=training)

    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        x,
        filters, (3, 3), (1, 1),
        padding='same',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    x = tf.layers.dropout(x, rate=dropout, training=training)

    return x


def dense_block(x,
                layers,
                growth_rate,
                bottleneck,
                dropout,
                initializer,
                regularizer,
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
            dropout=dropout,
            initializer=initializer,
            regularizer=regularizer,
            training=training,
            name='bottleneck_composite_function_{}'.format(i))
      else:
        y = composite_function(
            x,
            growth_rate,
            dropout=dropout,
            initializer=initializer,
            regularizer=regularizer,
            training=training,
            name='composite_function_{}'.format(i))

      x = tf.concat([x, y], -1)

    return x


def transition_layer(x,
                     compression_factor,
                     dropout,
                     initializer,
                     regularizer,
                     training,
                     name='transition_layer'):
  with tf.name_scope(name):
    filters = int(x.shape[-1].value * compression_factor)

    x = tf.layers.batch_normalization(x, training=training)
    x = tf.layers.conv2d(
        x,
        filters, (1, 1), (1, 1),
        padding='same',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    x = tf.layers.dropout(x, rate=dropout, training=training)
    x = tf.layers.average_pooling2d(x, (2, 2), (2, 2), padding='same')

    return x


def input(x, filters, initializer, regularizer, name='input'):
  with tf.name_scope(name):
    x = tf.layers.conv2d(
        x,
        filters,
        3,
        1,
        padding='same',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    return x


def output(x, initializer, regularizer, name='output'):
  with tf.name_scope(name):
    x = tf.reduce_mean(x, (1, 2), keep_dims=True)
    x = tf.layers.conv2d(
        x,
        1000,
        1,
        1,
        padding='same',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    return x


def densenet(x,
             block_depth,
             growth_rate,
             compression_factor,
             dropout,
             weight_decay,
             bottleneck=True,
             training=False):
  with tf.name_scope('densenet'):
    initializer = tf.contrib.layers.variance_scaling_initializer(
        factor=2.0, mode='FAN_IN', uniform=False)
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    if bottleneck and compression_factor < 1:
      x = input(
          x, growth_rate * 2, initializer=initializer, regularizer=regularizer)
    else:
      x = input(x, 16, initializer=initializer, regularizer=regularizer)

    x = dense_block(
        x,
        layers=block_depth,
        growth_rate=growth_rate,
        bottleneck=bottleneck,
        dropout=dropout,
        initializer=initializer,
        regularizer=regularizer,
        training=training,
        name='dense_block_1')
    x = transition_layer(
        x,
        compression_factor=compression_factor,
        dropout=dropout,
        initializer=initializer,
        regularizer=regularizer,
        training=training,
        name='transition_layer_1')
    x = dense_block(
        x,
        layers=block_depth,
        growth_rate=growth_rate,
        bottleneck=bottleneck,
        dropout=dropout,
        initializer=initializer,
        regularizer=regularizer,
        training=training,
        name='dense_block_2')
    x = transition_layer(
        x,
        compression_factor=compression_factor,
        dropout=dropout,
        initializer=initializer,
        regularizer=regularizer,
        training=training,
        name='transition_layer_2')
    x = dense_block(
        x,
        layers=block_depth,
        growth_rate=growth_rate,
        bottleneck=bottleneck,
        dropout=dropout,
        initializer=initializer,
        regularizer=regularizer,
        training=training,
        name='dense_block_3')

    x = output(x, initializer=initializer, regularizer=regularizer)

    return x
