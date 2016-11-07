import sqlite_wrapper as wrapper


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