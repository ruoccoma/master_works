from keras.layers import np
import tensorflow as tf
import math
import numpy as np
from theano import tensor


def split_list(data, split_ratio=0.8):
	return np.asarray(data[:int((len(data) * split_ratio))]), np.asarray(data[int((len(data) * split_ratio)):])


def insert_and_remove_last(index, array, element):
	array.insert(index, element)
	del array[-1]
	return array


def tf_l2norm(tensor_array):
	norm = tf.sqrt(tf.reduce_sum(tf.pow(tensor_array, 2)))
	tensor_array /= norm
	return tensor_array


def theano_l2norm(X):
	""" Compute L2 norm, row-wise """
	norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
	X /= norm[:, None]
	return X


def l2norm(array):
	norm = math.sqrt(np.sum(([math.pow(x, 2) for x in array])))
	array = [x / norm for x in array]
	return array

