

# Add all project modules to sys.path
import multiprocessing as mp
import os
import sys
import time
from random import randint

import numpy
# Get root dir (parent of parent of main.py)


ROOT_DIR = os.path.dirname((os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir)))) + "/"
sys.path.append(ROOT_DIR)

import settings
from euclidian_distance_architecture import EuclidanDistanceArchitecture
from image_database_helper import fetch_image_vector, fetch_all_image_vector_pairs
from caption_database_helper import fetch_filename_caption_tuple, fetch_all_filename_caption_vector_tuples
from embeddings_helper import structure_and_store_embeddings
from image_helpers import show_image, printProgress
from list_helpers import split_list, find_n_most_similar_images, compare_vectors, find_n_most_similar_images_fast
from word_averaging import create_caption_vector
# Import models

# Settings
LOAD_MODEL = True
PREDICT_NEW = False
ARCHITECTURES = [EuclidanDistanceArchitecture(epochs=50, batch_size=512)]
NEG_TAG = "neg" if settings.CREATE_NEGATIVE_EXAMPLES else "pos"


def word2visualvec_main():
	for ARCHITECTURE in ARCHITECTURES:
		file = open(settings.RESULT_TEXTFILE_PATH, 'a')
		file.write(ARCHITECTURE.get_name() + "\n")
		file.close()
		if LOAD_MODEL:
			load_model(ARCHITECTURE)
			ARCHITECTURE.generate_prediction_model()
		else:
			ARCHITECTURE.train()
			save_model_to_file(ARCHITECTURE.model, ARCHITECTURE)
			ARCHITECTURE.generate_prediction_model()

		if PREDICT_NEW:
			predict(ARCHITECTURE.prediction_model)
		else:
			print("Starting evaluation of model...")
			time_start = time.time()
			r1_avg, r5_avg, r10_avg, r20_avg = evaluate(ARCHITECTURE.prediction_model)
			time_end = time.time()
			# test_model(ARCHITECTURE.prediction_model)
			file = open(settings.RESULT_TEXTFILE_PATH, 'a')
			file.write("RESULTS: (Evaluating time: %s)\n" % s((time_end - time_start) / 60.0))
			file.write("r1:%s,r5:%s,r10:%s,r20:%s\n" % (r1_avg, r5_avg, r10_avg, r20_avg))
			file.close()


def save_model_to_file(model, architecture):
	name = architecture.get_name()
	model.save_weights("stored_models/" + name + ".h5")
	print("Saved model \"%s\" to disk" % name)


def load_model(arc):
	arc.generate_model()
	name = arc.get_name()
	print("Loading model \"%s\" from disk..." % name)
	arc.model.load_weights("stored_models/" + name + ".h5")
	arc.model.compile(optimizer=arc.optimizer, loss=arc.loss)


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

		best_image_vector_name_list, _ = find_n_most_similar_images(predicted_image_vector)
		print("Result for %s:" % captions[i])
		for i in range(len(best_image_vector_name_list)):
			filename = best_image_vector_name_list[i]
			show_image(settings.IMAGE_DIR + filename, str(i + 1) + "-" + filename)
			print(i + 1, filename)
		print("")


def test_model(model):
	test_size = 1
	test_caption_vectors = fetch_test_captions_vectors()
	numpy.random.shuffle(test_caption_vectors)
	start = randint(0, len(test_caption_vectors) - test_size)
	caption_vector_samples = test_caption_vectors[start:start + test_size]
	print("\nRESULTS")
	for i in range(len(caption_vector_samples)):
		caption_vector_list = caption_vector_samples[i:i + 1]
		caption_vector = caption_vector_list[0]

		correct_image_filename, correct_image_caption = fetch_filename_caption_tuple(caption_vector)
		correct_image_vector = fetch_image_vector(correct_image_filename)

		predicted_image_vector = model.predict(caption_vector_list)[0]

		best_image_name_list, best_image_vector_list = find_n_most_similar_images(predicted_image_vector)

		mse_predicted_correct = compare_vectors(correct_image_vector, predicted_image_vector)

		print("")
		print("Correct caption:\t", correct_image_caption)
		print("Correct filename:\t", correct_image_filename)
		print("MSE pred vs. cor:\t", mse_predicted_correct)
		print("")
		print("Result:")
		for i in range(len(best_image_name_list)):
			filename = best_image_name_list[i]
			show_image(settings.IMAGE_DIR + filename, str(i + 1) + "-" + filename)
			mse_pred = compare_vectors(predicted_image_vector, best_image_vector_list[i])
			mse_correct = compare_vectors(correct_image_vector, best_image_vector_list[i])
			print("%s - %s\t Image-vec MSE Pred: %s Correct: %s" % (i + 1, filename, mse_pred, mse_correct))
		print("")
		show_image(settings.IMAGE_DIR + correct_image_filename, "QUERY: " + correct_image_caption)


def totuple(a):
	try:
		return tuple(totuple(i) for i in a)
	except TypeError:
		return a

def evaluate(model):
	r1 = []
	r5 = []
	r10 = []
	r20 = []
	size = 20
	filename_vector_tuples = fetch_all_filename_caption_vector_tuples()
	filename_caption_vector_dictionary = dict()
	total_filname_caption_vector = len(filename_vector_tuples)
	for i in range(total_filname_caption_vector):
		filename, cap_vec = filename_vector_tuples[i]
		filename_caption_vector_dictionary[totuple(cap_vec)] = filename
		printProgress(i, total_filname_caption_vector, prefix="Converting to dictionary")

	caption_vectors = fetch_test_captions_vectors()
	predicted_image_vectors = model.predict(caption_vectors)

	filename_image_vector_pairs = fetch_all_image_vector_pairs()
	pool_formated_list = []
	len_caption_vectors = len(caption_vectors)
	for i in range(len_caption_vectors):
		caption_vector = caption_vectors[i]
		correct_image_filename = filename_caption_vector_dictionary[totuple(caption_vector)]
		predicted_image_vector = predicted_image_vectors[i]
		pool_formated_list.append((predicted_image_vector, correct_image_filename, filename_image_vector_pairs, 20))

	processes = int(mp.cpu_count() * 0.8)
	print("Running on %s processes" % processes)
	pool = mp.Pool(processes=processes)
	print("Starting pool...")
	result = pool.map_async(find_n_most_similar_images_fast, pool_formated_list)
	pool.close()  # No more work

	while not result.ready():
		new_chunks = result._number_left
		print("Chunks left %s" % new_chunks)
		time.sleep(10)

	name_name_lists = result.get()
	total_results = len(name_name_lists)
	for i in range(total_results):
		name, name_list = name_name_lists[i]
		for top_image_index in range(size):
			if name == name_list[top_image_index]:
				if top_image_index < 20:
					r20.append(1)
				if top_image_index < 10:
					r10.append(1)
				if top_image_index < 5:
					r5.append(1)
				if top_image_index == 0:
					r1.append(1)
		printProgress(i, total_results, prefix="Checking recall")

	r1_avg = sum(r1) / len_caption_vectors
	r5_avg = sum(r5) / len_caption_vectors
	r10_avg = sum(r10) / len_caption_vectors
	r20_avg = sum(r20) / len_caption_vectors

	return r1_avg, r5_avg, r10_avg, r20_avg


def fetch_test_captions_vectors():
	data_x, _, _ = structure_and_store_embeddings()
	training_test_ratio = 0.8
	_, test_x = split_list(data_x, training_test_ratio)
	return numpy.asarray(test_x)


word2visualvec_main()
