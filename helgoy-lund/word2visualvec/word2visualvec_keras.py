import numpy as np
from data_helper import generate_data
from keras.layers import Input, Dense
from keras.models import Model, model_from_json
from sklearn.metrics import mean_squared_error

from caption_database_helper import db_get_filename_caption_tuple_from_vector
from image_database_helper import fetch_image_vector_pairs

SAVE_MODEL = True
LOAD_MODEL = False
MODEL_SUFFIX = "-test"


def split_list(data, split_ratio=0.8):
	return np.asarray(data[:int((len(data) * split_ratio))]), np.asarray(data[int((len(data) * split_ratio)):])


def word2visualvec_main():
	if LOAD_MODEL:
		encoder, decoder, autoencoder = load_model("encoder"), load_model("decoder"), load_model("autoencoder")
	else:
		encoder, decoder = train()

	test_model(encoder)


def save_model_to_file(model, name):
	name += MODEL_SUFFIX
	# serialize model to JSON
	model_json = model.to_json()
	with open("stored_models/" + name + ".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("stored_models/" + name + ".h5")
	print("Saved model \"%s\" to disk" % name)


def load_model(name, optimizer='adadelta', loss='binary_crossentropy'):
	name += MODEL_SUFFIX
	# load json and create model
	json_file = open("stored_models/" + name + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("stored_models/" + name + ".h5")
	print("Loaded model \"%s\" from disk" % name)

	# evaluate loaded model on test data
	loaded_model.compile(optimizer=optimizer, loss=loss)
	return loaded_model


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)


def train():
	data_x, data_y = generate_data()

	training_test_ratio = 0.8

	training_data_x, test_data_x = split_list(data_x, training_test_ratio)
	training_data_y, test_data_y = split_list(data_y, training_test_ratio)

	x_dim = 128
	y_dim = 2048

	# this is our input placeholder
	x_dim_shape = Input(shape=(x_dim,))

	# Simple network:
	# "encoded" is the encoded representation of the input
	encoded = Dense(y_dim, activation='relu')(x_dim_shape)
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(y_dim, activation='sigmoid')(encoded)


	"""
	# Deep network: NOT WORKING
		encoded = Dense(512, activation='relu')(x_dim_shape)
		encoded = Dense(1024, activation='relu')(encoded)
		encoded = Dense(y_dim, activation='relu')(encoded)

		decoded = Dense(1025, activation='relu')(encoded)
		decoded = Dense(513, activation='relu')(decoded)
		decoded = Dense(x_dim, activation='sigmoid')(decoded)
	"""


	# this model maps an input to its reconstruction
	autoencoder = Model(input=x_dim_shape, output=decoded)

	# this model maps an input to its encoded representation
	encoder = Model(input=x_dim_shape, output=encoded)

	# create a placeholder for an encoded (2048-dimensional) input
	encoded_input = Input(shape=(y_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-1]
	# create the decoder model

	# TODO: Fetch decoder. Må kanskje være av typen sequential??
	# decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	autoencoder.fit(training_data_x, training_data_y,
	                nb_epoch=10,
	                batch_size=256,
	                shuffle=True,
	                validation_data=(test_data_x, test_data_y))

	"""
	Det enkleste nettverket når loss: 0.6281 som det beste etter 10 epoker.

	"""

	if SAVE_MODEL:
		save_model_to_file(autoencoder, "autoencoder")
		save_model_to_file(encoder, "encoder")
		# save_model_to_file(decoder, "decoder")

	return encoder, None


def test_model(trained_encoder):
	data_x, data_y = generate_data(10)
	training_test_ratio = 0.8

	train_x, test_x = split_list(data_x, training_test_ratio)
	train_y, test_y = split_list(data_y, training_test_ratio)

	img_num = 0
	testing_data_x = test_x[5:10]

	correct_image_filename, correct_image_caption = db_get_filename_caption_tuple_from_vector(testing_data_x[img_num])
	print("Correct caption:", correct_image_caption)
	print("Correct filename:", correct_image_filename)

	predicted_image_vector = trained_encoder.predict(testing_data_x)[img_num]

	image_vector_pairs = fetch_image_vector_pairs()

	best_image_vector = image_vector_pairs[img_num][1]
	best_image_vector_mse = compare_vectors(predicted_image_vector, image_vector_pairs[img_num][1])
	best_image_vector_name = image_vector_pairs[img_num][0]
	print("Finding closest image vector...")
	for name, image_vector in image_vector_pairs:
		temp_mse = compare_vectors(image_vector, best_image_vector)
		if temp_mse < best_image_vector_mse:
			best_image_vector = image_vector
			best_image_vector_mse = temp_mse
			best_image_vector_name = name
		elif temp_mse == best_image_vector_mse:
			print("Identical")
	print("Best image vector name:", str(best_image_vector_name))


word2visualvec_main()
