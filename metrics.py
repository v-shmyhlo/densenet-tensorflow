import tensorflow as tf


def loss(logits, labels):
    logits = tf.squeeze(logits, [1, 2])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

    return tf.metrics.mean(loss)


def accuracy(logits, labels):
    logits = tf.squeeze(logits, [1, 2])
    y_hat = tf.argmax(logits, -1)
    eq = tf.equal(y_hat, labels)

    return tf.metrics.mean(eq)
