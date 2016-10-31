# encoding=utf8

import os.path
import pickle
import settings

from caption_database_helper import fetch_caption_vectors_for_image_name, fetch_caption_count
from image_database_helper import fetch_all_image_names, fetch_image_vector


def structure_and_store_embeddings(size=-1):
	if embedding_exists(size):
		return load_embeddings(size)
	else:
		sorted_caption_vector_data = []
		sorted_image_data = []
		if size > 0:
			all_image_names = fetch_all_image_names()[:size]
		else:
			all_image_names = fetch_all_image_names()
		num_images = len(all_image_names)

		validate_database(num_images)

		print("Generating datasets for %s images" % num_images)
		counter = 1
		for image_name in all_image_names:
			image_vector = fetch_image_vector(image_name)
			for caption_vector in fetch_caption_vectors_for_image_name(image_name):
				sorted_image_data.append(image_vector)
				sorted_caption_vector_data.append(caption_vector)
			if counter % 100 == 0:
				print("%s/%s" % (counter, num_images))
			counter += 1
		print("Finished generating %s training example" % len(sorted_caption_vector_data))
		dataset = [sorted_caption_vector_data, sorted_image_data]

		save_embeddings(dataset, size)

		return dataset


def save_embeddings(dataset, size):
	pickle_file = open(find_filepath(size), 'wb')
	pickle.dump(dataset, pickle_file, protocol=2)
	pickle_file.close()


def find_filepath(size):
	return settings.STORED_EMBEDDINGS_DIR + get_filename(size)


def embedding_exists(size):
	return os.path.isfile(find_filepath(size))


def load_embeddings(size):
	print("Loaded datasets from local storage")
	pickle_file = open(find_filepath(size), 'rb')
	dataset = pickle.load(pickle_file)
	pickle_file.close()
	return dataset


def validate_database(num_images):
	if num_images == 0:
		raise IOError('No images in databases')
	elif fetch_caption_count() == 0:
		raise IOError('No captions in databases')


def get_filename(size):
	return "%sdataset-%s.picklefile" % (settings.STORED_EMBEDDINGS_PREFIX, size)

