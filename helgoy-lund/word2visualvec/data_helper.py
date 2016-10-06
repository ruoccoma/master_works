# encoding=utf8

import os.path
import pickle

from caption_database_helper import get_caption_vectors_for_image
from image_database_helper import fetch_all_image_names, fetch_image_vector


def generate_data(size=-1):
	if os.path.isfile(get_filename(size)):
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
		print("Generating data for %s images" % num_images)
		counter = 1
		for image_name in all_image_names:
			image_vector = fetch_image_vector(image_name)
			for caption_vector in get_caption_vectors_for_image(image_name):
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


def get_filename(size):
	return "dataset-" + str(size) + ".picklefile"