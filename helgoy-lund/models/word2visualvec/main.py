from random import randint

import numpy
import theano
from sklearn.metrics import mean_squared_error

import multibranch_keras
import settings
from caption_database_helper import fetch_filename_caption_tuple
from embeddings_helper import structure_and_store_embeddings
from image_database_helper import fetch_all_image_vector_pairs
from image_helpers import show_image
from list_helpers import split_list, insert_and_remove_last
from word_averaging import create_caption_vector

theano.config.openmp = True

# Import models
import feedforward_keras
# Settings
LOAD_MODEL = False
PREDICT_NEW = False
MODELS = [multibranch_keras, feedforward_keras]
MODEL = MODELS[0]
MODEL_SUFFIX = "-caption-model-%s-epochs" % MODEL.get_epochs()


def word2visualvec_main():
	if LOAD_MODEL:
		model = load_model(MODEL.__name__)
		prediction_model = MODEL.generate_prediction_model(model)
	else:
		model = MODEL.train()
		save_model_to_file(model, MODEL.__name__)
		prediction_model = MODEL.generate_prediction_model(model)

	if PREDICT_NEW:
		predict(prediction_model)
	else:
		test_model(prediction_model)


def save_model_to_file(model, name):
	name += MODEL_SUFFIX
	model.save_weights("stored_models/" + name + ".h5")
	print("Saved model \"%s\" to disk" % name)


def load_model(name):
	name += MODEL_SUFFIX

	loaded_model = MODEL.get_model()
	loaded_model.load_weights("stored_models/" + name + ".h5")

	print("Loaded model \"%s\" from disk" % name)

	# evaluate loaded model on test datasets
	loaded_model.compile(optimizer=MODEL.get_optimizer(), loss=MODEL.get_loss())
	return loaded_model


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)


def convert_query_to_vector(query):
	return numpy.asarray(create_caption_vector(query))


def convert_captions_to_vectors(queries):
	vectors = []
	for query in queries:
		vector = convert_query_to_vector(query)
		vectors.append(vector)
	return numpy.asarray(vectors)


def predict(model):
	captions = []
	user_provided_caption = " "
	while 1:
		user_provided_caption = input("EXIT WITH EMPTY - Enter caption: ")
		if user_provided_caption == "":
			break
		else:
			captions.append(user_provided_caption)
	samples = convert_captions_to_vectors(captions)
	for i in range(len(samples)):
		correct_caption_vector_list = samples[i:i + 1]

		predicted_image_vector = model.predict(correct_caption_vector_list)[0]

		best_image_vector_name_list = find_n_most_similar(predicted_image_vector)
		print("Result for %s:" % captions[i])
		for i in range(len(best_image_vector_name_list)):
			filename = best_image_vector_name_list[i]
			show_image(settings.IMAGE_DIR + filename, str(i + 1) + "-" + filename)
			print(i + 1, filename)
		print("")


def test_model(model):
	test_size = 1
	all_caption_vectors = fetch_test_captions_vectors()
	numpy.random.shuffle(all_caption_vectors)
	start = randint(0, len(all_caption_vectors) - test_size)
	samples = all_caption_vectors[start:start + test_size]
	print("\nRESULTS")
	for i in range(len(samples)):
		correct_caption_vector_list = samples[i:i + 1]
		correct_caption_vector = correct_caption_vector_list[0]

		correct_image_filename, correct_image_caption = fetch_filename_caption_tuple(
			correct_caption_vector)

		predicted_image_vector = model.predict(correct_caption_vector_list)[0]

		best_image_vector_name_list = find_n_most_similar(predicted_image_vector)

		print("")
		print("Correct caption:\t", correct_image_caption)
		print("")
		print("Correct filename:\t", correct_image_filename)
		print("")
		print("Result:")
		for i in range(len(best_image_vector_name_list)):
			filename = best_image_vector_name_list[i]
			show_image(settings.IMAGE_DIR + filename, str(i + 1) + "-" + filename)
			print(i + 1, filename)
		print("")
		show_image(settings.IMAGE_DIR + correct_image_filename, "QUERY: " + correct_image_caption)


def find_n_most_similar(predicted_image_vector):
	image_vector_pairs = fetch_all_image_vector_pairs()

	first_image_vector = image_vector_pairs[0][1]
	first_image_filename = image_vector_pairs[0][0]
	first_image_mse = compare_vectors(predicted_image_vector, first_image_vector)

	best_image_vector_mse_list = [0 for i in range(5)]
	best_image_vector_name_list = ["" for i in range(5)]
	best_image_vector_list = [[] for i in range(5)]

	best_image_vector_mse_list = insert_and_remove_last(0, best_image_vector_mse_list, first_image_mse)
	best_image_vector_name_list = insert_and_remove_last(0, best_image_vector_name_list, first_image_filename)
	best_image_vector_list = insert_and_remove_last(0, best_image_vector_list, first_image_vector)

	for temp_image_name, temp_image_vector in image_vector_pairs:
		temp_image_mse = compare_vectors(predicted_image_vector, temp_image_vector)
		for index in range(len(best_image_vector_list)):
			if temp_image_mse < best_image_vector_mse_list[index]:
				best_image_vector_mse_list = insert_and_remove_last(index, best_image_vector_mse_list,
				                                                    temp_image_mse)
				best_image_vector_name_list = insert_and_remove_last(index, best_image_vector_name_list,
				                                                     temp_image_name)
				best_image_vector_list = insert_and_remove_last(index, best_image_vector_list, temp_image_vector)
				break
	return best_image_vector_name_list


def fetch_test_captions_vectors():
	data_x, data_y = structure_and_store_embeddings()
	training_test_ratio = 0.8
	_, test_x = split_list(data_x, training_test_ratio)
	return numpy.asarray(data_x)


word2visualvec_main()
