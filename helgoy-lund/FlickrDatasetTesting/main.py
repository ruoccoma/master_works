from __future__ import print_function
from model import generate_model, save_model, load
from trainer import train_model

from keras.datasets import cifar10
from keras.layers import Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

nb_classes = 10
batch_size = 32

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = generate_model(X_train.shape[1:], nb_classes)
train_model(model, X_train, X_test, Y_train, Y_test)
save_model(model)

# model = load()


print(model.evaluate(X_train, Y_train, batch_size))
