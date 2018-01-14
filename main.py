import os
import argparse
from itertools import count
import tensorflow as tf
import densenet
import cifar10_dataset as cifar10
from utils import success, warning, log_args
import metrics


def make_parser():
  # TODO: log arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-path', type=str, required=True)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--learning-rate', type=float, default=0.001)
  parser.add_argument('--log-path', type=str, default='./tf_log')
  parser.add_argument('--save-path', type=str, default='./weights')
  return parser


def main():
  # TODO: add weight initializers

  args = make_parser().parse_args()
  log_args(args)

  global_step = tf.get_variable('global_step', initializer=0, trainable=False)
  training = tf.get_variable('training', initializer=False, trainable=False)

  with tf.name_scope('data_loading'):
    train_ds, test_ds = cifar10.make_dataset(args.dataset_path)
    train_ds, test_ds = (train_ds.batch(args.batch_size),
                         test_ds.batch(args.batch_size))

  iter = tf.data.Iterator.from_structure((tf.float32, tf.int64),
                                         ((None, 32, 32, 3), (None)))
  train_init = tf.group(training.assign(True), iter.make_initializer(train_ds))
  test_init = tf.group(training.assign(False), iter.make_initializer(test_ds))

  x, y = iter.get_next()
  logits = densenet.densenet(x, training=training)
  loss = metrics.loss(logits=logits, labels=y)
  accuracy = metrics.accuracy(logits=logits, labels=y)
  train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(
      loss, global_step=global_step)

  with tf.name_scope('summary'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  saver = tf.train.Saver()

  with tf.Session() as sess, tf.summary.FileWriter(
      os.path.join(args.log_path, 'train'),
      sess.graph,
  ) as train_writer, tf.summary.FileWriter(
      os.path.join(args.log_path, 'test'),
      sess.graph,
  ) as test_writer:
    restore_path = tf.train.latest_checkpoint(args.save_path)
    if restore_path:
      print(warning('Restoring from checkpoint'))
      saver.restore(sess, restore_path)
    else:
      print(warning('Initializing'))
      sess.run(tf.global_variables_initializer())

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
            success(
                'epoch: {}, step: {}, loss: {:.4f}, accuracy: {:.2f}'.format(
                    epoch, step, l, a * 100)))

        test_writer.add_summary(summ, step)
        test_writer.flush()
        save_path = saver.save(
            sess,
            os.path.join(args.save_path, 'model.ckpt'),
            write_meta_graph=False)
        print(warning('model saved: {}'.format(save_path)))


if __name__ == '__main__':
  main()
