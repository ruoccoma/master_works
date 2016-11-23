

# Add all project modules to sys.path
import datetime
import os
import sys
import time
from random import randint

import numpy

# Get root dir (parent of parent of main.py)


ROOT_DIR = os.path.dirname((os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir)))) + "/"
sys.path.append(ROOT_DIR)

import settings
from cosine_similarity_architecture import CosineSimilarityArchitecture, FiveLayerCosineSimilarityArchitecture
from euclidian_distance_architecture import EuclidanDistanceArchitecture
from image_database_helper import fetch_image_vector, fetch_all_image_vector_pairs
from caption_database_helper import fetch_filename_caption_tuple, fetch_all_filename_caption_vector_tuples
from embeddings_helper import structure_and_store_embeddings
from image_helpers import show_image, printProgress
from list_helpers import split_list, find_n_most_similar_images, compare_vectors
# from word_averaging import create_caption_vector
from clustering import kmeans_clustering, compare_to_cluster, get_member_ids_dict

# Import models

# Settings
PREDICT_NEW = False
ARCHITECTURES = [FiveLayerCosineSimilarityArchitecture(epochs=100, batch_size=256)]
NEG_TAG = "neg" if settings.CREATE_NEGATIVE_EXAMPLES else "pos"


def train():
	current_time = datetime.datetime.time(datetime.datetime.now())
	print("Current time: %s" % current_time)
	for ARCHITECTURE in ARCHITECTURES:
		print("\nRUNNING NEW ARCHITECTURE: %s\n" % ARCHITECTURE.get_name())
		file = open(settings.RESULT_TEXTFILE_PATH, 'a')
		file.write(ARCHITECTURE.get_name() + "\n")
		file.close()
		if is_saved(ARCHITECTURE):
			load_model(ARCHITECTURE)
			ARCHITECTURE.generate_prediction_model()
		else:
			ARCHITECTURE.train()
			save_model_to_file(ARCHITECTURE.model, ARCHITECTURE)
			ARCHITECTURE.generate_prediction_model()
		ARCHITECTURE = None
		print("\n")


def evaluate():
	current_time = datetime.datetime.time(datetime.datetime.now())
	print("Current time: %s" % current_time)
	for ARCHITECTURE in ARCHITECTURES:
		if is_saved(ARCHITECTURE):
			load_model(ARCHITECTURE)
			ARCHITECTURE.generate_prediction_model()

			if PREDICT_NEW:
				#predict(ARCHITECTURE.prediction_model)
				test_model(ARCHITECTURE.prediction_model)
			else:
				print("Starting evaluation of model...")
				time_start = time.time()
				r1_avg, r5_avg, r10_avg, r20_avg, r100_avg, r1000_avg = evaluate_model(ARCHITECTURE.prediction_model)
				time_end = time.time()

				# test_model(ARCHITECTURE.prediction_model)

				result_header = "RESULTS: (Evaluating time: %s)\n" % ((time_end - time_start) / 60.0)
				recall_results = "r1:%s,r5:%s,r10:%s,r20:%s,r100:%s,r1000:%s\n" % (r1_avg, r5_avg, r10_avg, r20_avg, r100_avg, r1000_avg)

				file = open(settings.RESULT_TEXTFILE_PATH, 'a')
				file.write(result_header)
				file.write(recall_results)
				file.close()

				print(result_header)
				print(recall_results)
			print("\n")
		else:
			print("Architecture not trained")
			print(ARCHITECTURE.get_name())


def save_model_to_file(model, architecture):
	name = architecture.get_name()
	model.save_weights("stored_models/" + name + ".h5")
	print("Saved model \"%s\" to disk" % name)


def is_saved(arc):
	if os.path.isfile("stored_models/" + arc.get_name() + ".h5"):
		return True
	return False


def load_model(arc):
	arc.generate_model()
	name = arc.get_name()
	print("Loading model \"%s\" from disk..." % name)
	arc.model.load_weights("stored_models/" + name + ".h5")
	arc.model.compile(optimizer=arc.optimizer, loss=arc.loss)


# def convert_query_to_vector(query):
# 	return numpy.asarray(create_caption_vector(query))


# def convert_captions_to_vectors(queries):
# 	vectors = []
# 	for query in queries:
# 		vector = convert_query_to_vector(query)
# 		vectors.append(vector)
# 	return numpy.asarray(vectors)


# def predict(model):
# 	captions = []
# 	user_provided_caption = " "
# 	while 1:
# 		user_provided_caption = input("EXIT WITH EMPTY - Enter caption: ")
# 		if user_provided_caption == "":
# 			break
# 		else:
# 			captions.append(user_provided_caption)
# 	samples = convert_captions_to_vectors(captions)
# 	for i in range(len(samples)):
# 		correct_caption_vector_list = samples[i:i + 1]
#
# 		predicted_image_vector = model.predict(correct_caption_vector_list)[0]
#
# 		best_image_vector_name_list, _ = find_n_most_similar_images(predicted_image_vector)
# 		print("Result for %s:" % captions[i])
# 		for i in range(len(best_image_vector_name_list)):
# 			filename = best_image_vector_name_list[i]
# 			show_image(settings.IMAGE_DIR + filename, str(i + 1) + "-" + filename)
# 			print(i + 1, filename)
# 		print("")


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


def evaluate_model(model):
	r1 = []
	r5 = []
	r10 = []
	r20 = []
	r100 = []
	r1000 = []
	size = 1000

	filename_caption_vector_tuples = fetch_all_filename_caption_vector_tuples()
	caption_vector_filename_dictionary = dict()
	total_filname_caption_vector = len(filename_caption_vector_tuples)
	for i in range(total_filname_caption_vector):
		filename, cap_vec = filename_caption_vector_tuples[i]
		tuple_key = totuple(cap_vec)
		caption_vector_filename_dictionary[tuple_key] = filename
		if i % 1000 == 0:
			printProgress(i + 1, total_filname_caption_vector, prefix="Building cap vec -> filename dict", barLength=50)
	print("\n")

	test_caption_vectors = fetch_test_captions_vectors()

	predicted_image_vectors = model.predict(test_caption_vectors)

	filename_caption_vector_tuples = fetch_all_image_vector_pairs()
	filenames = [x[0] for x in filename_caption_vector_tuples]
	image_vectors = [x[1] for x in filename_caption_vector_tuples]

	cluster = kmeans_clustering(image_vectors)
	members_dict = get_member_ids_dict(filenames, image_vectors, cluster)

	# test_filenames = filenames[int(len(filenames)*0.8):]
	# test_caption_vectors = image_vectors[int(len(filenames)*0.8):]

	# predicted_image_vectors = model.predict(numpy.asarray(test_caption_vectors))
	len_test_caption_vectors = len(test_caption_vectors)

	total_results = len(test_caption_vectors)
	for i in range(total_results):
		most_similar, cluster_id = compare_to_cluster([predicted_image_vectors[i]], cluster, 1000, members_dict)
		for top_image_index in range(size):
			test_caption_vector = test_caption_vectors[i]
			filename = caption_vector_filename_dictionary[totuple(test_caption_vector)]
			if filename == most_similar[top_image_index]:
				if top_image_index < 1000:
					r1000.append(1)
				if top_image_index < 100:
					r100.append(1)
				if top_image_index < 20:
					r20.append(1)
				if top_image_index < 10:
					r10.append(1)
				if top_image_index < 5:
					r5.append(1)
				if top_image_index == 0:
					r1.append(1)
		printProgress(i + 1, total_results, prefix="Calculating recall", suffix=str(cluster_id))

	r1_avg = sum(r1) / len_test_caption_vectors
	r5_avg = sum(r5) / len_test_caption_vectors
	r10_avg = sum(r10) / len_test_caption_vectors
	r20_avg = sum(r20) / len_test_caption_vectors
	r100_avg = sum(r100) / len_test_caption_vectors
	r1000_avg = sum(r1000) / len_test_caption_vectors
	return r1_avg, r5_avg, r10_avg, r20_avg, r100_avg, r1000_avg


def fetch_test_captions_vectors():
	data_x, _, _ = structure_and_store_embeddings()
	training_test_ratio = 0.8
	_, test_x = split_list(data_x, training_test_ratio)
	return numpy.asarray(test_x)


def debug():
	for ARCHITECTURE in ARCHITECTURES:
		print("Debugging")
		if is_saved(ARCHITECTURE):
			load_model(ARCHITECTURE)
			ARCHITECTURE.generate_prediction_model()
			model = ARCHITECTURE.model
			#model = Model(input=base_model.input, output=base_model.get_layer("Cosine_layer").output)
			min = 1
			max = 0
			test_caption_vector = fetch_test_captions_vectors()[:1000]
			for i in range(len(test_caption_vector)):
				correct_image_filename, correct_image_caption = fetch_filename_caption_tuple(test_caption_vector[i])
				correct_image_vector = fetch_image_vector(correct_image_filename)
				#caption_vector = numpy.reshape(test_caption_vector[i], (1, 300))
				#image_vector = numpy.reshape(correct_image_vector, (1, 4096))
				caption_vector = numpy.asarray(test_caption_vector[i:i+1])
				image_vector = numpy.asarray([correct_image_vector])
				#print(model.summary())

				cos = model.predict([caption_vector, image_vector])[0][0]
				if cos > max:
					max = cos
				if cos < min:
					min = cos

			print("Min cos: ", min)
			print("Max cos: ", max)


if len(sys.argv) > 1 and sys.argv[1] == "eval":
	evaluate()
elif len(sys.argv) > 1 and sys.argv[1] == "debug":
	debug()
else:
	train()
