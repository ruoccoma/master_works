import numpy as np

import sqlite_wrapper as db_wrapper


def save_caption_vector(image_name, caption_text, caption_vector):
	db_wrapper.db_insert_caption_vector(image_name, caption_text, caption_vector)


def save_caption_vector_list(tuple_list):
	print("Storing captions in database...")
	db_wrapper.db_insert_caption_vector_list(tuple_list)


def fetch_caption_vectors_for_image_name(image_name):
	vectors = db_wrapper.db_get_caption_vectors(image_name)
	vector_list = [i[0] for i in vectors]
	return vector_list


def fetch_all_caption_vectors():
	return db_wrapper.db_fetch_all_caption_vectors()


def fetch_filename_caption_tuple(caption_vector):
	return db_wrapper.db_get_filename_caption_tuple_from_caption_vector(caption_vector)


def fetch_caption_count():
	return db_wrapper.db_get_caption_table_size()


def fetch_all_filename_caption_vector_tuples():
	return db_wrapper.db_all_filename_caption_vector_tuple()


def fetch_all_caption_rows():
	return db_wrapper.db_all_caption_rows()


def fetch_filenames_from_cation_vector(caption_vector):
	return db_wrapper.db_get_filenames_from_caption_vector(caption_vector)


if __name__ == "__main__":
	caps = fetch_caption_vectors_for_image_name("1000092795.jpg")
	for cap in caps:
		print len(cap)
