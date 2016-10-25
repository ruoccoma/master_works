from keras.layers import np
import tensorflow as tf


def split_list(data, split_ratio=0.8):
    return np.asarray(data[:int((len(data) * split_ratio))]), np.asarray(data[int((len(data) * split_ratio)):])


def insert_and_remove_last(index, array, element):
    array.insert(index, element)
    del array[-1]
    return array


def l2norm(X):
    norm = tf.sqrt(tf.reduce_sum(tf.pow(X, 2)))
    X /= norm[:, None]
    return X
