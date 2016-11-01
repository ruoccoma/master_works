from keras.layers import np
import tensorflow as tf
import math
import numpy as np
from theano import tensor
import sys


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


# Print iterations progress
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s%s%s  %s' % (prefix, bar, percents, '%', iteration, '/', total, suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
