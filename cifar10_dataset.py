import os
import pickle
import numpy as np
import tensorflow as tf


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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-path', type=str, required=True)
  args = parser.parse_args()

  train_ds, test_ds = make_dataset(args.dataset_path)
  iter = tf.data.Iterator.from_structure(
      (tf.float32, tf.int64),
      ((32, 32, 3), ()),
  )
  train_init = iter.make_initializer(train_ds),
  test_init = iter.make_initializer(test_ds),
  x, y = iter.get_next()

  with tf.Session() as sess:
    sess.run(train_init)
    img = sess.run(x)
    plt.imshow(img)
    plt.show()

    sess.run(test_init)
    img = sess.run(x)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  main()
