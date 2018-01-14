import tensorflow as tf


def loss(logits, labels):
  logits = tf.squeeze(logits, [1, 2])
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)
  loss = tf.reduce_mean(loss)

  return loss


def accuracy(logits, labels):
  logits = tf.squeeze(logits, [1, 2])
  y_hat = tf.argmax(logits, -1)
  eq = tf.equal(y_hat, labels)

  return tf.reduce_mean(tf.to_float(eq))
