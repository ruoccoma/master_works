from __future__ import print_function

from keras.layers import Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.optimizers import SGD

yaml_model_filename = "model.yml"
model_weights_filename = "weights"


def compile_model(model):
	# let's train the model using SGD + momentum (how original).
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


def generate_model(input_shape, nb_classes):
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
	model.add(Activation('relu'))

	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())

	model.add(Dense(512))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	compile_model(model)

	return model


# save/load models in keras http://machinelearningmastery.com/save-load-keras-deep-learning-models/
def load():
	# load YAML and create model
	yaml_file = open(yaml_model_filename, 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	loaded_model = model_from_yaml(loaded_model_yaml)
	# load weights into new model
	loaded_model.load_weights(model_weights_filename)
	print("Loaded model from disk")
	compile_model(loaded_model)
	return loaded_model


# yaml_filename = "model.yaml"
# h5_weights_filename = "model.h5"
def save_model(model):
	# serialize model to YAML
	model_yaml = model.to_yaml()
	with open(yaml_model_filename, "w") as yaml_file:
		yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights(model_weights_filename)
	print("Saved model to disk")
