import tensorflow as tf


def loss(logits, labels, top_k):
    logits = tf.squeeze(logits, [1, 2])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

    if top_k is not None:
        loss, _ = tf.nn.top_k(loss, top_k)

    return tf.reduce_mean(loss)
