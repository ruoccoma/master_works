# encoding=utf8

import os.path
import pickle

import settings
from caption_database_helper import fetch_caption_count, fetch_all_filename_caption_vector_tuples
from image_database_helper import fetch_all_image_names, fetch_all_image_vector_pairs
from image_helpers import printProgress
from io_helper import load_pickle_file, save_pickle_file, check_pickle_file


def structure_and_store_embeddings(size=-1):
	if embedding_exists(size):
		return load_embeddings(size)
	else:
		print("Generating compatible dataset...")
		all_image_names, image_name_caption_vector_dict = create_dictionaries(size)
		print("Generating positive training data")
		pos_sorted_caption_vector_data, pos_sorted_image_data, pos_similarity = get_examples(all_image_names, image_name_caption_vector_dict)
		image_caption = pos_sorted_caption_vector_data
		image_data = pos_sorted_image_data
		similarity = pos_similarity
		if settings.CREATE_NEGATIVE_EXAMPLES:
			print("\nGenerating negative training data")
			neg_sorted_caption_vector_data, neg_sorted_image_data, neg_similarity = get_examples(all_image_names, image_name_caption_vector_dict, False)
			image_caption += neg_sorted_caption_vector_data
			image_data += neg_sorted_image_data
			similarity += neg_similarity

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


def get_similarity_dictionary():
	pickle_file = open(settings.ROOT_DIR + "helpers/similarity-dict.p", 'rb')
	dataset = pickle.load(pickle_file)
	pickle_file.close()
	return dataset


def get_examples(all_image_names, image_name_caption_vector_dict, positive=True):
	sorted_caption_vector_data = []
	sorted_image_data = []
	similiarity_dict = dict()
	if not positive:
		similiarity_dict = get_similarity_dictionary()
	image_name_image_vector_dict = {key: value for (key, value) in fetch_all_image_vector_pairs()}
	all_image_names_total = len(all_image_names)
	for i in range(all_image_names_total):
		image_name = all_image_names[i]
		image_vector = image_name_image_vector_dict[image_name]
		if positive:
			caption_vectors = image_name_caption_vector_dict[image_name]
		else:
			dissimilar_image_name = similiarity_dict[image_name][0][0]
			caption_vectors = image_name_caption_vector_dict[dissimilar_image_name]
		for caption_vector in caption_vectors:
			sorted_image_data.append(image_vector)
			sorted_caption_vector_data.append(caption_vector)
		printProgress(i + 1, all_image_names_total, prefix='Generating data:', suffix='Complete', barLength=50)

	return sorted_caption_vector_data, sorted_image_data, [1.0 if positive else 0.0 for x in range(len(sorted_caption_vector_data))]


def save_embeddings(dataset_to_store, size):
	filepath = find_stored_embeddings_filepath(size)
	save_pickle_file(dataset_to_store, filepath)


def load_embeddings(size):
	print("Loaded compatible dataset from local storage: %s" % get_stored_embeddings_filename(size))
	filepath = find_stored_embeddings_filepath(size)
	return load_pickle_file(filepath)


def find_stored_embeddings_filepath(size):
	return settings.STORED_EMBEDDINGS_DIR + get_stored_embeddings_filename(size)


def embedding_exists(size):
	filepath = find_stored_embeddings_filepath(size)
	return check_pickle_file(filepath)


def validate_database(num_images):
	if num_images == 0:
		raise IOError('No images in databases')
	elif fetch_caption_count() == 0:
		raise IOError('No captions in databases')


def get_stored_embeddings_filename(size):
	if size == -1:
		size = "all"
	return "%s-%s.picklefile" % (settings.STORED_EMBEDDINGS_NAME, size)


if __name__ == "__main__":
	from image_database_helper import fetch_filename_from_image_vector
	from caption_database_helper import fetch_filenames_from_cation_vector, fetch_filename_caption_tuple

	dataset = structure_and_store_embeddings(5)
	caption_vectors = dataset[0]
	img_vecs = dataset[1]
	sims = dataset[2]
	print("%s \t %s \t %s \t %s" % (
		"filename from cap", "filename from img", "similarity",
		"text caption"))
	for i in range(len(caption_vectors)):
		cap_vec = caption_vectors[i]
		img_vec = img_vecs[i]
		sim = sims[i]

		print("%s \t %s \t %s \t %s" % (
			fetch_filenames_from_cation_vector(cap_vec), fetch_filename_from_image_vector(img_vec), sim,
			fetch_filename_caption_tuple(cap_vec)[1]))
