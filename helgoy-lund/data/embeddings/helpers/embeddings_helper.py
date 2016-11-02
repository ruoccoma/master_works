# encoding=utf8

import os.path
import pickle
from random import randint

import settings

from caption_database_helper import fetch_caption_count, fetch_all_filename_caption_vector_tuples
from image_database_helper import fetch_all_image_names, fetch_all_image_vector_pairs
from image_helpers import show_image, printProgress
from list_helpers import find_n_most_similar


def structure_and_store_embeddings(size=-1):
	if embedding_exists(size):
		return load_embeddings(size)
	else:
		print("Generating compatible dataset...")
		all_image_names, image_name_caption_vector_dict = create_dictionaries(size)
		print("Generating positive training data")
		pos_sorted_caption_vector_data, pos_sorted_image_data, pos_similarity = get_examples(all_image_names, image_name_caption_vector_dict)
		print("\nGenerating negative training data")
		neg_sorted_caption_vector_data, neg_sorted_image_data, neg_similarity = get_examples(all_image_names, image_name_caption_vector_dict, False)
		image_caption = pos_sorted_caption_vector_data + neg_sorted_caption_vector_data
		image_data = pos_sorted_image_data + neg_sorted_image_data
		similarity = pos_similarity + neg_similarity
		dataset = [image_caption, image_data, similarity]

		print("Finished generating %s training example" % len(image_caption))
		save_embeddings(dataset, size)

		return dataset


def create_dictionaries(size):
	if size > 0:
		all_image_names = fetch_all_image_names()[:size]
	else:
		all_image_names = fetch_all_image_names()
	num_images = len(all_image_names)
	validate_database(num_images)
	image_name_caption_vector_dict = dict()
	name_cap_vec_tuples = fetch_all_filename_caption_vector_tuples()
	for (name, cap_vec) in name_cap_vec_tuples:
		if name in image_name_caption_vector_dict:
			image_name_caption_vector_dict[name].append(cap_vec)
		else:
			image_name_caption_vector_dict[name] = [cap_vec]
	return all_image_names, image_name_caption_vector_dict


def get_examples(all_image_names, image_name_caption_vector_dict, positive=True):
	sorted_caption_vector_data = []
	sorted_image_data = []
	image_name_image_vector_dict = {key: value for (key, value) in fetch_all_image_vector_pairs()}
	all_image_names_total = len(all_image_names)
	for i in range(all_image_names_total):
		image_name = all_image_names[i]
		image_vector = image_name_image_vector_dict[image_name]
		if positive:
			caption_vectors = image_name_caption_vector_dict[image_name]
		else:
			chose_dissimilar_from_size = 10
			dissimilar_image_name = find_n_most_similar(image_vector, image_name_image_vector_dict, chose_dissimilar_from_size, False)[randint(0, chose_dissimilar_from_size - 1)]
			caption_vectors = image_name_caption_vector_dict[dissimilar_image_name]
			#show_image(settings.IMAGE_DIR + image_name, "Image: ", image_name)
			#show_image(settings.IMAGE_DIR + dissimilar_image_name, "Dissimilar: ", dissimilar_image_name)
			#print("Most different: ", find_n_most_similar(image_name_image_vector_dict[dissimilar_image_name], 1, False)[0])
		for caption_vector in caption_vectors:
			sorted_image_data.append(image_vector)
			sorted_caption_vector_data.append(caption_vector)
		printProgress(i, all_image_names_total, prefix='Generating data:', suffix='Complete', barLength=50)

	return sorted_caption_vector_data, sorted_image_data, [1 if positive else 0 for x in range(len(sorted_caption_vector_data))]


def save_embeddings(dataset, size):
	pickle_file = open(find_filepath(size), 'wb')
	pickle.dump(dataset, pickle_file, protocol=2)
	pickle_file.close()


def find_filepath(size):
	return settings.STORED_EMBEDDINGS_DIR + get_filename(size)


def embedding_exists(size):
	return os.path.isfile(find_filepath(size))


def load_embeddings(size):
	print("Loaded compatible dataset from local storage")
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
	structure_and_store_embeddings(10)
