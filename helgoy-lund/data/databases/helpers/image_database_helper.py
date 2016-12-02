import sqlite_wrapper as wrapper
# from helpers.list_helpers import l2norm
import numpy as np
import math
import copy
from sklearn.preprocessing import normalize


def store_image_vector_to_db(image_name, vector):
	wrapper.db_insert_image_vector(image_name, vector)


def fetch_all_image_names():
	return [x[0] for x in wrapper.db_keys_images()]


def fetch_image_vector(image_name):
	return wrapper.db_get_image_vector(image_name)[0]


def fetch_all_image_vector_pairs():
	return wrapper.db_all_filename_img_vec_pairs()


def fetch_filename_from_image_vector(image_vector):
	return wrapper.db_get_filename_from_image_vector(image_vector)


def normalize_abs_image_vectors():
	tr_im_image_vector_tuple = wrapper.db_all_filename_img_vec_pairs()
	for filename, vector in tr_im_image_vector_tuple:
		temp_vector = np.copy(vector)
		vector = l2norm(vector)
		# vector = np.absolute(vector)
		vector_sum = 0
		temp_vector_sum = 0
		for i in range(len(vector)):
			sum += temp_vector[i]

			if vector[i] != temp_vector[i]:
				print(vector[i], temp_vector[i])
			# print(vector[i], "\t", temp_vector[i])
		print(sum)
		return



def l2norm(array):
	norm = math.sqrt(np.sum(([math.pow(x, 2) for x in array])))
	array = [x / norm for x in array]
	return array

if __name__ == "__main__":
	# normalize_abs_image_vectors()
	v1 = np.asarray([1, 2])
	temp_v1 = np.copy(v1)
	v1 = l2norm(v1)
	print(temp_v1, v1)