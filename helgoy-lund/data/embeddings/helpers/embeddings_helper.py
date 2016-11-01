# encoding=utf8

import os.path
import pickle
import settings

from list_helpers import printProgress
from caption_database_helper import fetch_caption_count, fetch_all_filename_caption_vector_tuples
from image_database_helper import fetch_all_image_names, fetch_all_image_vector_pairs


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

		print("Generating compatible dataset...")
		image_name_image_vector_dict = {key: value for (key, value) in fetch_all_image_vector_pairs()}

		image_name_caption_vector_dict = dict()

		name_cap_vec_tuples = fetch_all_filename_caption_vector_tuples()
		for (name, cap_vec) in name_cap_vec_tuples:
			if name in image_name_caption_vector_dict:
				image_name_caption_vector_dict[name].append(cap_vec)
			else:
				image_name_caption_vector_dict[name] = [cap_vec]

		counter = 1
		for image_name in all_image_names:
			image_vector = image_name_image_vector_dict[image_name]
			caption_vectors = image_name_caption_vector_dict[image_name]
			for caption_vector in caption_vectors:
				sorted_image_data.append(image_vector)
				sorted_caption_vector_data.append(caption_vector)
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
	return "%s-%s.picklefile" % (settings.STORED_EMBEDDINGS_PREFIX, size)


if __name__ == "__main__":
	structure_and_store_embeddings(99)
