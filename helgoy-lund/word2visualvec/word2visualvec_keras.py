#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data_helper import generate_data
from helper import split_list
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error

from caption_database_helper import db_get_filename_caption_tuple_from_vector
from image_database_helper import fetch_image_vector, fetch_image_vector_pairs

# Import models
import feedforward_keras
import autoencoder_keras

# Settings
SAVE_MODEL = True
LOAD_MODEL = True
MODELS = [feedforward_keras, autoencoder_keras]
MODEL = MODELS[0]
MODEL_SUFFIX = ""


def word2visualvec_main():
	if LOAD_MODEL:
		# model = load_model(MODEL.__name__)
		model = load_model("feedforward_keras-e_30")
	else:
		model = MODEL.train()

		if SAVE_MODEL:
			save_model_to_file(model, MODEL.__name__)
		# save_model_to_file(autoencoder, "autoencoder")
		# save_model_to_file(decoder, "decoder")

	test_model(model)


def save_model_to_file(model, name):
	name += MODEL_SUFFIX
	# serialize model to JSON
	model_json = model.to_json()
	with open("stored_models/" + name + ".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("stored_models/" + name + ".h5")
	print("Saved model \"%s\" to disk" % name)


def load_model(name):
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
	loaded_model.compile(optimizer=MODEL.get_optimizer(), loss=MODEL.get_loss())
	return loaded_model


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)


def insert_and_remove_last(list, element):
	list.insert(0, element)
	del list[-1]
	return list


def test_model(model):
	img_num = 0
	all_caption_vectors = fetch_test_captions_vectors()

	correct_caption_vector_list = all_caption_vectors[img_num:img_num + 1]
	correct_caption_vector = correct_caption_vector_list[img_num]

	correct_image_filename, correct_image_caption = db_get_filename_caption_tuple_from_vector(correct_caption_vector)
	correct_image_vector = fetch_image_vector(correct_image_filename)

	predicted_image_vector = model.predict(correct_caption_vector_list)[img_num]

	image_vector_pairs = fetch_image_vector_pairs()
	first_image_vector = image_vector_pairs[img_num][1]
	first_image_filename = image_vector_pairs[img_num][0]

	best_image_vector_mse_list = [0 for i in range(5)]
	best_image_vector_name_list = ["" for i in range(5)]
	best_image_vector_list = [[] for i in range(5)]

	insert_and_remove_last(best_image_vector_mse_list, compare_vectors(predicted_image_vector, first_image_vector))
	insert_and_remove_last(best_image_vector_name_list, first_image_filename)
	insert_and_remove_last(best_image_vector_list, first_image_vector)

	print("Finding closest image vector...")
	for name, image_vector in image_vector_pairs[1:]:
		temp_mse = compare_vectors(predicted_image_vector, image_vector)
		if temp_mse < best_image_vector_mse_list[0]:
			insert_and_remove_last(best_image_vector_mse_list, temp_mse)
			insert_and_remove_last(best_image_vector_name_list, name)
			insert_and_remove_last(best_image_vector_list, image_vector)
	print("")
	print("RESULTS")
	print("")
	print("Correct caption:\t", correct_image_caption)
	print("")
	print("Correct filename:\t", correct_image_filename)
	print("")
	print("Result:")
	for i in range(len(best_image_vector_mse_list)):
		print(i+1, best_image_vector_name_list[i])


def fetch_test_captions_vectors():
	data_x, data_y = generate_data(10)
	training_test_ratio = 0.8
	_, test_x = split_list(data_x, training_test_ratio)
	testing_data_x = test_x[5:10]
	return testing_data_x


word2visualvec_main()
