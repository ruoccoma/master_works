# encoding=utf8

import os.path
import pickle

from caption_database_helper import fetch_caption_vectors_for_image_name, fetch_caption_count
from image_database_helper import fetch_all_image_names, fetch_image_vector


def generate_data(size=-1):
	if os.path.isfile(get_filename(size)):
		print("Loaded datasets from local storage")
		pickle_file = open(get_filename(size), 'rb')
		dataset = pickle.load(pickle_file)
		return dataset
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
		pickle_file = open(get_filename(size), 'wb')
		pickle.dump(dataset, pickle_file)
		return dataset


def validate_database(num_images):
	if num_images == 0:
		raise IOError('No images in databases')
	elif fetch_caption_count() == 0:
		raise IOError('No captions in databases')


def get_filename(size):
	return "dataset-" + str(size) + ".picklefile"
