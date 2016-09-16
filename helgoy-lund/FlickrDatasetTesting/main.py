from __future__ import print_function

from sys import argv

from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import np_utils
from model import generate_model, save_model, load
from trainer import train_model

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

if len(argv) > 1:
	if argv[1] == "load":
		model = load()
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy',
		              optimizer=sgd,
		              metrics=['accuracy'])

else:
	model = generate_model(X_train.shape[1:], nb_classes)
	train_model(model, X_train, X_test, Y_train, Y_test)
	save_model(model)

print(model.evaluate(X_train, Y_train, batch_size))
