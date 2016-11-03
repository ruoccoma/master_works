from random import randint

import numpy

from sklearn.metrics import mean_squared_error

import settings
from caption_database_helper import fetch_all_filename_caption_vector_tuples, fetch_filename_caption_tuple
from embeddings_helper import structure_and_store_embeddings
from image_database_helper import fetch_all_image_names, fetch_image_vector, fetch_all_image_vector_pairs
from image_helpers import show_image
from list_helpers import split_list, insert_and_remove_last


def fetch_test_captions_vectors():
	data_x, _, _ = structure_and_store_embeddings()
	training_test_ratio = 0.8
	_, test_x = split_list(data_x, training_test_ratio)
	return test_x


def fetch_test_image_vectors():
	_, data_y, _ = structure_and_store_embeddings(200)
	training_test_ratio = 0.8
	_, test_y = split_list(data_y, training_test_ratio)
	return test_y


def compare_vectors(v1, v2):
	try:
		error = mean_squared_error(v1, v2)
		return error
	except Exception as e:
		print("All nan!")
		return -1


def test_caption_vectors():
	test_size = 1
	all_caption_vectors = fetch_test_captions_vectors()
	numpy.random.shuffle(all_caption_vectors)
	start = randint(0, len(all_caption_vectors) - test_size)
	samples = all_caption_vectors[start:start + test_size]
	print("\nRESULTS")
	for i in range(len(samples)):
		correct_caption_vector_list = all_caption_vectors[i:i + 1]
		correct_caption_vector = correct_caption_vector_list[0]

		correct_image_filename, correct_image_caption = fetch_filename_caption_tuple(correct_caption_vector)

		caption_vector_pairs = fetch_all_filename_caption_vector_tuples()
		first_caption_vector = caption_vector_pairs[0][1]
		first_caption_filename = caption_vector_pairs[0][0]
		first_caption_mse = compare_vectors(correct_caption_vector, first_caption_vector)

		best_caption_vector_mse_list = [0 for i in range(5)]
		best_caption_vector_name_list = ["" for i in range(5)]
		best_caption_vector_list = [[] for i in range(5)]

		best_caption_vector_mse_list = insert_and_remove_last(0, best_caption_vector_mse_list, first_caption_mse)
		best_caption_vector_name_list = insert_and_remove_last(0, best_caption_vector_name_list, first_caption_filename)
		best_caption_vector_list = insert_and_remove_last(0, best_caption_vector_list, first_caption_vector)

		for temp_image_name, temp_caption_vector in caption_vector_pairs:
			temp_caption_vector_mse = compare_vectors(correct_caption_vector, temp_caption_vector)
			if temp_caption_vector_mse == -1:
				print(temp_image_name)
			for index in range(len(best_caption_vector_list)):
				if temp_caption_vector_mse < best_caption_vector_mse_list[index]:
					best_caption_vector_mse_list = insert_and_remove_last(index, best_caption_vector_mse_list,
					                                                      temp_caption_vector_mse)
					best_caption_vector_name_list = insert_and_remove_last(index, best_caption_vector_name_list,
					                                                       temp_image_name)
					best_caption_vector_list = insert_and_remove_last(index, best_caption_vector_list,
					                                                  temp_caption_vector)
					break
		print("")
		print("Correct caption:\t", correct_image_caption)
		print("")
		print("Correct filename:\t", correct_image_filename)
		print("")
		print("Most similar images(chosen using caption vectors):")
		for i in range(len(best_caption_vector_mse_list)):
			filename = best_caption_vector_name_list[i]
			show_image(settings.IMAGE_DIR + filename, filename)
			print(i + 1, filename)
		print("")


def test_image_vectors():
	test_size = 1
	all_image_names = fetch_all_image_names()
	numpy.random.shuffle(all_image_names)
	start = randint(0, len(all_image_names) - test_size)
	samples = all_image_names[start:start + test_size]
	print("\nRESULTS")
	for i in range(len(samples)):
		correct_image_name = all_image_names[i:i + 1][0]

		correct_image_vector = fetch_image_vector(correct_image_name)

		image_vector_pairs = fetch_all_image_vector_pairs()
		first_image_filename = image_vector_pairs[0][0]
		first_image_vector = image_vector_pairs[0][1]
		first_image_mse = compare_vectors(correct_image_vector, first_image_vector)

		best_image_vector_mse_list = [0 for i in range(5)]
		best_image_vector_name_list = ["" for i in range(5)]
		best_image_vector_list = [[] for i in range(5)]

		best_image_vector_mse_list = insert_and_remove_last(0, best_image_vector_mse_list, first_image_mse)
		best_image_vector_name_list = insert_and_remove_last(0, best_image_vector_name_list, first_image_filename)
		best_image_vector_list = insert_and_remove_last(0, best_image_vector_list, first_image_vector)

		for temp_image_name, temp_image_vector in image_vector_pairs:
			temp_image_vector_mse = compare_vectors(correct_image_vector, temp_image_vector)
			for index in range(len(best_image_vector_list)):
				if temp_image_vector_mse < best_image_vector_mse_list[index]:
					best_image_vector_mse_list = insert_and_remove_last(index, best_image_vector_mse_list,
					                                                      temp_image_vector_mse)
					best_image_vector_name_list = insert_and_remove_last(index, best_image_vector_name_list,
					                                                       temp_image_name)
					best_image_vector_list = insert_and_remove_last(index, best_image_vector_list,
					                                                  temp_image_vector)
					break
		print("")
		print("Correct filename:\t", correct_image_name)
		print("")
		print("Most similar images(chosen using image vectors):")
		for i in range(len(best_image_vector_mse_list)):
			filename = best_image_vector_name_list[i]
			show_image(settings.IMAGE_DIR + filename, filename)
			print(i + 1, filename)
		print("")


# test_caption_vectors()
test_image_vectors()
