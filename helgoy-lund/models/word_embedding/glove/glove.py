#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import settings
from word_database_helper import save_word_vector_tuple


def run_glove():
	with open(settings.ROOT_DIR + "models/word_embedding/glove/glove.6B.300d.txt") as word_embedding:
		print("Saving word embeddings to database...")
		words = []
		for line in word_embedding.readlines():
			line = (line.strip()).split(" ")
			word = line.pop(0)
			line = map(float, line)
			words.append((word, np.asarray(line)))
		save_word_vector_tuple(words)


if __name__ == "__main__":
	run_glove()
