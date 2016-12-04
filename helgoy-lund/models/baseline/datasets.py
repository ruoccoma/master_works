"""
Dataset loading
"""
from collections import OrderedDict

import numpy
import settings
from image_database_helper import fetch_all_image_vector_pairs
from io_helper import save_pickle_file, check_pickle_file, load_pickle_file


def load_dataset():
	dataset = {"caps": [], "ims": []}

	text_caption_file = open(settings.WORD_FILEPATH)
	all_lines = text_caption_file.readlines()
	text_caption_file.close()

	"""
	1306145560_1e37081b91.jpg#1	A black dog swims in the water .
	"""
	caption_dictionary = {}
	for caption_line in all_lines:
		splitted_line = caption_line.split("#")
		filename = splitted_line[0]
		caption = splitted_line[1][2:].strip()
		if filename in caption_dictionary:
			caption_dictionary[filename].append(caption)
		else:
			caption_dictionary[filename] = [caption]

	image_vector_pairs = fetch_all_image_vector_pairs()  # List with (filename, array)

	for filename, vector in image_vector_pairs:
		for caption in caption_dictionary[filename]:
			dataset["ims"].append(vector)
			dataset["caps"].append(caption)

	dataset["ims"] = numpy.asarray(dataset["ims"])

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
