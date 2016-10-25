import numpy as np

import sqlite_wrapper as db_wrapper


def save_caption_vector(image_name, caption_text, caption_vector):
	db_wrapper.db_insert_caption_vector(image_name, caption_text, caption_vector)


def fetch_caption_vectors_for_image_name(image_name):
	vectors = db_wrapper.db_get_caption_vectors(image_name)
	vector_list = [i[0] for i in vectors]
	return vector_list


def fetch_all_caption_vectors():
	return db_wrapper.db_get_all_caption_vectors()


def fetch_filename_caption_tuple(caption_vector):
	return db_wrapper.db_get_filename_caption_tuple_from_caption_vector(caption_vector)


def fetch_caption_count():
	return db_wrapper.db_get_caption_table_size()


def fetch_all_filename_caption_vector_tuples():
	return db_wrapper.db_all_filename_caption_vector_tuple()
