import tensorflow as tf


def loss(logits, labels):
  logits = tf.squeeze(logits, [1, 2])
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)

  return tf.reduce_mean(loss)
