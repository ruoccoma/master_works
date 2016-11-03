from __future__ import division

import math
import pickle
from random import randint

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from theano import tensor

from image_database_helper import fetch_all_image_vector_pairs


def split_list(data, split_ratio=0.8):
	return np.asarray(data[:int((len(data) * split_ratio))]), np.asarray(data[int((len(data) * split_ratio)):])


def insert_and_remove_last(index, array, element):
	array.insert(index, element)
	del array[-1]
	return array


def tf_l2norm(tensor_array):
	norm = tf.sqrt(tf.reduce_sum(tf.pow(tensor_array, 2)))
	tensor_array /= norm
	return tensor_array


def theano_l2norm(X):
	""" Compute L2 norm, row-wise """
	norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
	X /= norm[:, None]
	return X


def l2norm(array):
	norm = math.sqrt(np.sum(([math.pow(x, 2) for x in array])))
	array = [x / norm for x in array]
	return array


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)


def find_n_most_similar(predicted_image_vector, image_vector_pairs_dictionary, n=5, most_similar=True):
	first_key = image_vector_pairs_dictionary.iterkeys().next()
	first_image_vector = image_vector_pairs_dictionary[first_key]
	first_image_filename = first_key
	first_image_mse = compare_vectors(predicted_image_vector, first_image_vector)

	best_image_vector_mse_list = [0 for i in range(n)]
	best_image_vector_name_list = ["" for i in range(n)]
	# best_image_vector_list = [[] for i in range(n)]

	best_image_vector_mse_list = insert_and_remove_last(0, best_image_vector_mse_list, first_image_mse)
	best_image_vector_name_list = insert_and_remove_last(0, best_image_vector_name_list, first_image_filename)
	# best_image_vector_list = insert_and_remove_last(0, best_image_vector_list, first_image_vector)

	for temp_image_name in image_vector_pairs_dictionary.iterkeys():
		temp_image_vector = image_vector_pairs_dictionary[temp_image_name]
		temp_image_mse = compare_vectors(predicted_image_vector, temp_image_vector)
		for index in range(n):
			if most_similar:
				should_insert = temp_image_mse < best_image_vector_mse_list[index]
			else:
				should_insert = temp_image_mse > best_image_vector_mse_list[index]
			if should_insert:
				best_image_vector_mse_list = insert_and_remove_last(index, best_image_vector_mse_list,
																	temp_image_mse)
				best_image_vector_name_list = insert_and_remove_last(index, best_image_vector_name_list,
																	 temp_image_name)
				# best_image_vector_list = insert_and_remove_last(index, best_image_vector_list, temp_image_vector)
				break
	return best_image_vector_name_list


def generate_sorted_similarity(image_vector_tuple):

	image_filname, image_vector, image_vector_pairs = image_vector_tuple

	total_size = len(image_vector_pairs)
	# Numver of random images to compare
	size = 100
	start = randint(0, total_size - size * 2)
	image_vector_pairs = image_vector_pairs[start:start + size]

	first_image_vector = image_vector_pairs[0][1]
	first_image_filename = image_vector_pairs[0][0]
	first_image_mse = compare_vectors(image_vector, first_image_vector)

	best_image_vector_tuple_list = [("", 0) for i in range(size)]

	best_image_vector_tuple_list = insert_and_remove_last(0, best_image_vector_tuple_list,
	                                                      (first_image_filename, first_image_mse))

	for temp_image_name, temp_image_vector in image_vector_pairs[1:]:
		temp_image_mse = compare_vectors(image_vector, temp_image_vector)
		for index in range(total_images_length):
			should_insert = temp_image_mse > best_image_vector_tuple_list[index][1]
			if should_insert:
				best_image_vector_tuple_list = insert_and_remove_last(index, best_image_vector_tuple_list,
				                                                      (temp_image_name, temp_image_mse))
				break

	return (image_filname, best_image_vector_tuple_list)


import multiprocessing as mp
import time


def make_similarity_dict():
	pool = mp.Pool()
	print("Getting image-vector pairs...")
	image_vector_pairs = fetch_all_image_vector_pairs()

	pool_formated_list = [(image_filname, image_vector, image_vector_pairs) for image_filname, image_vector in
	                      image_vector_pairs]

	print("Starting pool...")
	result = pool.map_async(generate_sorted_similarity, pool_formated_list)
	pool.close()  # No more work

	while not result.ready():
		new_chunks = result._number_left
		print("Chunks left %s" % new_chunks)
		time.sleep(10)

	map_res = result.get()
	return dict(map_res)


if __name__ == "__main__":
	SAVE_NEW = True
	if SAVE_NEW:
		a = make_similarity_dict()
		try:
			pickle_file = open("similarity-dict.p", 'wb')
			pickle.dump(a, pickle_file, protocol=2)
			pickle_file.close()
			print("Saved dictionary to file.")
		except Exception as e:
			print(a)
			print(e)
	else:
		pickle_file = open("similarity-dict.p", 'rb')
		dataset = pickle.load(pickle_file)
		pickle_file.close()
