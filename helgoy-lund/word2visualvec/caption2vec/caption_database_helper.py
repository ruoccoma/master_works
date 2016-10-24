import numpy as np

from sqliteDatabase import db_get_caption_vectors, db_insert_caption_vector, db_get_filename_caption_tuple_from_vector, \
	db_get_caption_table_size


def insert_caption_vector_into_db(image_name, caption_vector):
	db_insert_caption_vector(image_name, caption_vector)


def get_caption_vectors_for_image(image_name):
	vectors = db_get_caption_vectors(image_name)
	vector_list = [i[0] for i in vectors]
	return vector_list

def fetch_all_caption_vectors():
	return db_get_all_caption_vectors()

def fetch_image_name_for_vecgor(vector):
	return db_get_filename_caption_tuple_from_caption_vector(vector)


def fetch_caption_count():
	return db_get_caption_table_size()



def fetch_all_filename_caption_vector_tuples():
	return db_all_filename_caption_vector_tuple()


def testing():
	image_name = "image1"
	vector1 = np.asarray([1, 2, 3])
	caption1 = "This is a caption"
	db_insert_caption_vector(image_name, caption1, vector1)
	print(get_caption_vectors_for_image(image_name))
	for vector in get_caption_vectors_for_image(image_name):
		print(vector)