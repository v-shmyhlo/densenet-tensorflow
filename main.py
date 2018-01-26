import os
import argparse
from itertools import count
import tensorflow as tf
import densenet
import cifar10_dataset as cifar10
from utils import success, warning, danger, log_args
import metrics
import objectives


def make_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-path', type=str, required=True)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--growth-rate', type=int, default=12)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--compression-factor', type=float, default=0.5)
  parser.add_argument('--learning-rate', type=float, default=1e-3)
  parser.add_argument('--log-path', type=str, default='./tf_log')
  parser.add_argument('--save-path', type=str, default='./weights')
  parser.add_argument('--dropout', type=float, default=0.2)
  parser.add_argument('--block-depth', type=int, default=10)
  parser.add_argument('--weight-decay', type=float, default=1e-4)
  return parser


def main():
  # TODO: should not increment global_step on test set
  # TODO: weight decay

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

  x, y = iter.get_next()
  logits = densenet.densenet(
      x,
      block_depth=args.block_depth,
      growth_rate=args.growth_rate,
      compression_factor=args.compression_factor,
      dropout=args.dropout,
      weight_decay=args.weight_decay,
      training=training)
  loss, update_loss = metrics.loss(logits=logits, labels=y)
  accuracy, update_accuracy = metrics.accuracy(logits=logits, labels=y)

  train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(
      objectives.loss(logits=logits, labels=y),
      global_step=global_step,
  )

  with tf.name_scope('summary'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  saver = tf.train.Saver()

  train_init = tf.group(
      training.assign(True),
      iter.make_initializer(train_ds),
      tf.local_variables_initializer(),
  )
  test_init = tf.group(
      training.assign(False),
      iter.make_initializer(test_ds),
      tf.local_variables_initializer(),
  )

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
      sess.run(train_init)
      try:
        for _ in count():
          _, step = sess.run([(train_step, update_loss, update_accuracy),
                              global_step])
          print(step, end='\r')
      except tf.errors.OutOfRangeError:
        pass

      print(success('epoch: {}, step: {}'.format(epoch, step)))

      l, a, summary = sess.run([loss, accuracy, merged])
      print(
          success('(train) loss: {:.4f}, accuracy: {:.2f}'.format(l, a * 100)))
      train_writer.add_summary(summary, step)
      train_writer.flush()

      sess.run(test_init)
      try:
        for _ in count():
          _, step = sess.run([(update_loss, update_accuracy), global_step])
          print(danger(step), end='\r')
      except tf.errors.OutOfRangeError:
        pass

      l, a, summary = sess.run([loss, accuracy, merged])
      print(
          success('(test) loss: {:.4f}, accuracy: {:.2f}'.format(l, a * 100)))
      test_writer.add_summary(summary, step)
      test_writer.flush()

      save_path = saver.save(
          sess,
          os.path.join(args.save_path, 'model.ckpt'),
          write_meta_graph=False)
      print(warning('model saved: {}'.format(save_path)))


if __name__ == '__main__':
  main()
