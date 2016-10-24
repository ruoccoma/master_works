import numpy as np
from data_helper import generate_data
from keras.layers import Input, Dense
from keras.models import Model, model_from_json
from sklearn.metrics import mean_squared_error

from caption_database_helper import db_get_filename_caption_tuple_from_caption_vector
from image_database_helper import fetch_image_vector_pairs


def split_list(data, split_ratio=0.8):
	return np.asarray(data[:int((len(data) * split_ratio))]), np.asarray(data[int((len(data) * split_ratio)):])


# this is the size of our encoded representations
encoding_dim = 2048

input_img = Input(shape=(128,))
encoded_1 = Dense(127, activation='relu')(input_img)
encoded_2 = Dense(64, activation='relu')(encoded_1)
encoded_3 = Dense(2048, activation='relu')(encoded_2)

decoded_1 = Dense(64, activation='relu')(encoded_3)
decoded_2 = Dense(129, activation='relu')(decoded_1)
decoded_3 = Dense(128, activation='sigmoid', name="siste-decoded")(decoded_2)

# this model maps an input to its reconstruction
autoencoder = Model(input=encoded_3(encoded_2(encoded_1(input_img))), output=decoded_3(decoded_2(decoded_1(encoded_3))))

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded_3(encoded_2(encoded_1(input_img))))

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]

# create the decoder model
# decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

data_x, data_y = generate_data()

training_test_ratio = 0.8

training_data_x, test_data_x = split_list(data_x, training_test_ratio)
training_data_y, test_data_y = split_list(data_y, training_test_ratio)

autoencoder.fit(training_data_x, training_data_y,
                nb_epoch=2,
                batch_size=256,
                shuffle=True,
                validation_data=(test_data_x, test_data_y))