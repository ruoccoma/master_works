"""
Dataset loading
"""
from collections import OrderedDict

import numpy

import settings
from image_database_helper import fetch_all_image_vector_pairs
from list_helpers import split_list


def load_dataset():
	all_data = {"caps": [], "ims": []}

	text_caption_file = open(settings.WORD_FILEPATH)
	all_lines = text_caption_file.readlines()
	text_caption_file.close()

	caption_dictionary = {}
	for caption_line in all_lines:
		splitted_line = caption_line.split("#")
		filename = splitted_line[0]
		caption = splitted_line[1][2:].strip()
		if filename in caption_dictionary:
			caption_dictionary[filename].append(caption)
		else:
			caption_dictionary[filename] = [caption]

	image_vector_pairs = fetch_all_image_vector_pairs()

	for filename, vector in image_vector_pairs:
		for caption in caption_dictionary[filename]:
			all_data["ims"].append(vector)
			all_data["caps"].append(caption)

	all_data["ims"] = numpy.asarray(all_data["ims"])
	train_dict = {}
	test_dict = {}

	train_ims, test_ims = split_list(all_data["ims"], 0.8)
	train_caps, test_caps = split_list(all_data["caps"], 0.8)

	train_caps = numpy.asarray(train_caps)
	test_caps = numpy.asarray(test_caps)

	train_dict["ims"] = train_ims
	train_dict["caps"] = train_caps
	test_dict["ims"] = test_ims
	test_dict["caps"] = test_caps

	dataset = {"train": train_dict, "test": test_dict}
	return dataset


def build_dictionary(text):
	wordcount = OrderedDict()
	for cc in text:
		words = cc.split()
		for w in words:
			if w not in wordcount:
				wordcount[w] = 0
			wordcount[w] += 1
	words = wordcount.keys()
	freqs = wordcount.values()
	sorted_idx = numpy.argsort(freqs)[::-1]

	worddict = OrderedDict()
	for idx, sidx in enumerate(sorted_idx):
		worddict[words[sidx]] = idx + 2  # 0: <eos>, 1: <unk>

	return worddict
