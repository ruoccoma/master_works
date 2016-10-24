from random import randint

import numpy
from PIL import Image
from data_helper import generate_data
from helper import split_list, insert_and_remove_last
from sklearn.metrics import mean_squared_error

from caption_database_helper import db_get_filename_caption_tuple_from_caption_vector, \
	fetch_all_filename_caption_vector_tuples


def fetch_test_captions_vectors():
	data_x, data_y = generate_data(200)
	training_test_ratio = 0.8
	_, test_x = split_list(data_x, training_test_ratio)
	return test_x


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)


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

		correct_image_filename, correct_image_caption = db_get_filename_caption_tuple_from_caption_vector(
			correct_caption_vector)

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
			img = Image.open("./cnn/Flicker8k_Dataset/" + filename)
			img.show()
			print(i + 1, filename)
		print("")


test_caption_vectors()
