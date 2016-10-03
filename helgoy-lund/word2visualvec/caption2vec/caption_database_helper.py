import numpy as np

from sqliteDatabase import db_get_caption_vectors, db_insert_caption_vector


def insert_caption_vector_into_db(image_name, caption_vector):
	db_insert_caption_vector(image_name, caption_vector)


def get_caption_vectors_for_image(image_name):
	vectors = db_get_caption_vectors(image_name)
	vector_list = [i[0] for i in vectors]
	return vector_list


def testing():
	image_name = "image1"
	vector1 = np.asarray([1, 2, 3])
	caption1 = "This is a caption"
	db_insert_caption_vector(image_name, caption1, vector1)
	print(get_caption_vectors_for_image(image_name))
	for vector in get_caption_vectors_for_image(image_name):
		print(vector)

testing()