from random import randint

import numpy
from sklearn.metrics.pairwise import cosine_similarity

import settings
from caption_database_helper import fetch_filename_caption_tuple, fetch_all_filename_caption_vector_tuples
from embeddings_helper import structure_and_store_embeddings
from image_database_helper import fetch_image_vector, fetch_all_image_vector_pairs
from image_helpers import show_image
from list_helpers import find_n_most_similar_images, compare_vectors, print_progress, totuple, split_list
from word_averaging import create_caption_vector


def predict_text2image_model(model):
	captions = []
	while 1:
		# TODO input not working in python 2
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


def test_text2image_model(model):
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


def evaluate_text2image_model(model):
	r1 = []
	r5 = []
	r10 = []
	r20 = []
	r100 = []
	r1000 = []

	te_ca_caption_vectors = fetch_test_captions_vectors()
	predicted_image_vectors = model.predict(te_ca_caption_vectors)

	tr_im_filename_image_vector_tuples = fetch_all_image_vector_pairs()
	tr_im_filenames = [x[0] for x in tr_im_filename_image_vector_tuples]
	tr_im_image_vectors = [x[1] for x in tr_im_filename_image_vector_tuples]

	tr_ca_caption_vector_tuples = fetch_all_filename_caption_vector_tuples()

	tr_ca_caption_vector_filename_dictionary = build_caption_vector_filename_dict(tr_ca_caption_vector_tuples)

	print("Creating cosine similarity matrix...")
	similarity_matrix = cosine_similarity(predicted_image_vectors, tr_im_image_vectors)
	predicted_images_size = len(predicted_image_vectors)
	total_image_size = len(tr_im_image_vectors)
	for predicted_image_index in range(predicted_images_size):
		similarities = []
		for i in range(total_image_size):
			tr_filename = tr_im_filenames[i]
			similarities.append((tr_filename, similarity_matrix[predicted_image_index][i]))
		similarities.sort(key=lambda s: s[1], reverse=True)

		test_caption_vector = te_ca_caption_vectors[predicted_image_index]
		test_caption_vector_key = totuple(test_caption_vector)
		test_filename = tr_ca_caption_vector_filename_dictionary[test_caption_vector_key]

		for top_image_index in range(1000):
			comparison_filename = similarities[top_image_index][0]
			if test_filename == comparison_filename:
				if top_image_index < 1000:
					r1000.append(1.0)
				if top_image_index < 100:
					r100.append(1.0)
				if top_image_index < 20:
					r20.append(1.0)
				if top_image_index < 10:
					r10.append(1.0)
				if top_image_index < 5:
					r5.append(1.0)
				if top_image_index == 0:
					r1.append(1.0)

		print_progress(predicted_image_index + 1, predicted_images_size, prefix="Calculating recall")

	r1_avg = sum(r1) / predicted_images_size
	r5_avg = sum(r5) / predicted_images_size
	r10_avg = sum(r10) / predicted_images_size
	r20_avg = sum(r20) / predicted_images_size
	r100_avg = sum(r100) / predicted_images_size
	r1000_avg = sum(r1000) / predicted_images_size
	return r1_avg, r5_avg, r10_avg, r20_avg, r100_avg, r1000_avg


def build_caption_vector_filename_dict(filename_caption_vector_tuples):
	caption_vector_filename_dictionary = {}
	total_filname_caption_vector = len(filename_caption_vector_tuples)
	for i in range(total_filname_caption_vector):
		filename, cap_vec = filename_caption_vector_tuples[i]
		tuple_key = totuple(cap_vec)
		caption_vector_filename_dictionary[tuple_key] = filename
		if i % 1000 == 0 or i > total_filname_caption_vector - 5:
			print_progress(i + 1, total_filname_caption_vector, prefix="Building cap vec -> filename dict",
			               barLength=50)
	return caption_vector_filename_dictionary


def fetch_test_captions_vectors():
	data_x, _, _ = structure_and_store_embeddings()
	training_test_ratio = 0.8
	_, test_x = split_list(data_x, training_test_ratio)
	return numpy.asarray(test_x)


def convert_captions_to_vectors(queries):
	vectors = []
	for query in queries:
		vector = convert_query_to_vector(query)
		vectors.append(vector)
	return numpy.asarray(vectors)


def convert_query_to_vector(query):
	return numpy.asarray(create_caption_vector(query))
