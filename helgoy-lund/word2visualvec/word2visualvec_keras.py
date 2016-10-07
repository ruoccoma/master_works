import math

import numpy as np
from data_helper import generate_data
from keras.layers import Input, Dense
from keras.models import Model, load_model

#
# fetch data set
data_x, data_y = generate_data(2000)

trainig_test_ratio = 0.8

training_data_x = np.asarray(data_x[:int((len(data_x) * trainig_test_ratio))])
training_data_y = np.asarray(data_y[:int((len(data_y) * trainig_test_ratio))])

test_data_x = np.asarray(data_x[:int(math.ceil(len(data_x) * (1 - trainig_test_ratio)))])
test_data_y = np.asarray(data_y[:int(math.ceil(len(data_y) * (1 - trainig_test_ratio)))])

# from keras.datasets import mnist
# (numpy_train, _), (numpy_test, _) = mnist.load_data()
#
# numpy_train = numpy_train.astype('float32') / 255.
# numpy_test = numpy_test.astype('float32') / 255.
# numpy_train = numpy_train.reshape((len(numpy_train), np.prod(numpy_train.shape[1:])))
# numpy_test = numpy_test.reshape((len(numpy_test), np.prod(numpy_test.shape[1:])))
# print(numpy_train.shape)
# print(numpy_test.shape)



x_dim = 128
y_dim = 2048

# this is the size of our encoded representations
encoding_dim = y_dim  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(x_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(y_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(training_data_x, training_data_y,
                nb_epoch=2,
                batch_size=256,
                shuffle=True,
                validation_data=(test_data_x, test_data_y))

# autoencoder.fit(x_train, x_train,
#                 nb_epoch=10,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
