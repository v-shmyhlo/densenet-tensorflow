import os
import argparse
from itertools import count
import termcolor
import pickle
import tensorflow as tf
import densenet
import matplotlib.pyplot as plt
import numpy as np


def success(str):
  return termcolor.colored(str, 'green')


def warning(str):
  return termcolor.colored(str, 'yellow')


def make_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-path', type=str, required=True)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--learning-rate', type=float, default=0.001)
  parser.add_argument('--log-path', type=str, default='./tf_log')
  parser.add_argument('--save-path', type=str, default='./weights')
  return parser


def load_file(path):
  with open(path, 'rb') as f:
    dict = pickle.load(f, encoding='bytes')
    return dict[b'data'], dict[b'labels']


def preprocessing(data, labels):
  data = tf.reshape(data, (3, 32 * 32))
  data = tf.transpose(data)
  data = tf.reshape(data, (32, 32, 3))
  data /= 255

  labels = tf.to_int64(labels)

  return data, labels


def make_dataset(path):
  train_data_batches = []
  train_labels_batches = []

  for i in range(1, 6):
    data, labels = load_file(os.path.join(path, 'data_batch_{}'.format(i)))
    train_data_batches.append(data)
    train_labels_batches.append(labels)

  train_data_batches = np.array(train_data_batches)
  train_labels_batches = np.array(train_labels_batches)
  train_ds = tf.data.Dataset.from_tensor_slices((train_data_batches,
                                                 train_labels_batches))
  train_ds = train_ds.flat_map(
      lambda data, labels: tf.data.Dataset.from_tensor_slices((data, labels)))

  test_data, test_labels = load_file(os.path.join(path, 'test_batch'))
  test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

  train_ds = train_ds.map(preprocessing)
  test_ds = test_ds.map(preprocessing)

  return train_ds, test_ds


def make_loss(logits, labels):
  logits = tf.squeeze(logits, [1, 2])
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)
  loss = tf.reduce_mean(loss)

  return loss


def make_accuracy(logits, labels):
  logits = tf.squeeze(logits, [1, 2])
  y_hat = tf.argmax(logits, -1)
  eq = tf.equal(y_hat, labels)

  return tf.reduce_mean(tf.to_float(eq))


def main():
  args = make_parser().parse_args()

  # epochs = tf.get_variable('epochs', [], tf.int64, trainable=False)
  global_step = tf.get_variable('global_step', [], tf.int64, trainable=False)
  training = tf.get_variable('training', [], tf.bool, trainable=False)

  train_ds, test_ds = make_dataset(args.dataset_path)
  train_ds, test_ds = (train_ds.batch(args.batch_size),
                       test_ds.batch(args.batch_size))

  iter = tf.data.Iterator.from_structure((tf.float32, tf.int64),
                                         ((None, 32, 32, 3), (None)))
  train_init = tf.group(
      [training.assign(True),
       iter.make_initializer(train_ds)])
  test_init = tf.group(
      [training.assign(False),
       iter.make_initializer(test_ds)])

  x, y = iter.get_next()
  logits = densenet.densenet(x, training=training)
  loss = make_loss(logits=logits, labels=y)
  accuracy = make_accuracy(logits=logits, labels=y)
  train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(
      loss, global_step=global_step)

  with tf.name_scope('summary'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  saver = tf.train.Saver()

  with tf.Session() as sess, tf.summary.FileWriter(
      os.path.join(args.log_path, 'test'),
      sess.graph,
  ) as test_writer:
    if tf.train.latest_checkpoint(os.path.dirname(args.save_path)):
      print(warning('Restoring from checkpoint'))
      saver.restore(sess, args.save_path)
    else:
      print(warning('Initializing'))
      sess.run(tf.global_variables_initializer())

    # for epoch in range(sess.run(epochs), args.epochs):
    for epoch in range(args.epochs):
      try:
        sess.run(train_init)

        for _ in count():
          step, _ = sess.run([global_step, train_step])
          print(step, end='\r')

      except tf.errors.OutOfRangeError:
        sess.run(test_init)

        step, summ, l, a = sess.run([global_step, merged, loss, accuracy])

        print(
            success('epoch: {}, step: {}, loss: {}, accuracy: {}'.format(
                epoch, step, l, a * 100)))

        test_writer.add_summary(summ, step)
        test_writer.flush()
        save_path = saver.save(sess, args.save_path, write_meta_graph=False)
        print(warning('model saved: {}'.format(save_path)))


if __name__ == '__main__':
  main()
