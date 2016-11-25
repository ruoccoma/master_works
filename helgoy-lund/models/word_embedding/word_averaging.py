#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np  # Make sure that numpy is imported

import settings
from caption_database_helper import save_caption_vector, save_caption_vector_list
from list_helpers import print_progress
from word_database_helper import fetch_all_word_vectors, fetch_word_vector


def getWordVectors():
	return fetch_all_word_vectors()


def generate_and_store_caption_vectors(filepath):
	sentence_file = open(filepath)
	lines = sentence_file.readlines()
	sentence_file.close()

	sentences = extract_sentences(lines)

	mean_vectors = convert_sentences(np.asarray(sentences), settings.WORD_EMBEDDING_DIMENSION)

	store_caption_vector(filepath, mean_vectors)


def extract_sentences(lines):
	sentences = []
	for line in lines:
		sentence = []
		for x in ((((line.split(".jpg#")[1])[1:]).strip()).split()):
			if x != "." or x != ",":
				sentence.append(x.lower())

		sentences.append(sentence)
	return sentences


def store_caption_vector(filepath, caption_vector):
	with open(filepath) as f:
		lineNumber = 0
		captions = []
		for line in f.readlines():
			image_name = line.split("#")[0]
			caption_text = ((line.split("#")[1])[1:]).strip()
			captions.append((image_name, caption_text, caption_vector[lineNumber]))
			lineNumber += 1
		save_caption_vector_list(captions)


def convert_sentence_to_vector(words, num_features, dictionary):
	# Function to average all of the word vectors in a given
	# paragraph
	#
	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,), dtype="float32")
	#
	nwords = 0.
	#
	# Loop over each word in the review and, if it is in the model's
	# vocaublary, add its feature vector to the total
	for word in words:
		# word_vector = fetch_word_vector(word, None)
		if word in dictionary:
			nwords += 1.
			featureVec = np.add(featureVec, dictionary[word])
	#
	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec, nwords)
	return featureVec


def convert_sentences(sentences, num_features):
	# Given a set of sentences (each one a list of words), calculate
	# the average feature vector for each one and return a 2D numpy array
	#
	# Initialize a counter
	counter = 0
	#
	# Preallocate a 2D numpy array, for speed
	len_sentences = len(sentences)
	sentenceFeatureVecs = np.zeros((len_sentences, num_features), dtype="float32")


	word_vectors = fetch_all_word_vectors()
	print("Building word-vec dict")
	dictionary = {key: value for (key, value) in word_vectors}
	print("Building word-vec dict complete.")

	# Loop through the reviews
	for sentence in sentences:
		if counter % 1000 == 0:
			print_progress(counter, len_sentences, prefix='Convert sentences:', suffix='Complete', barLength=50)

		# Call the function (defined above) that makes average feature vectors
		sentenceFeatureVecs[counter] = convert_sentence_to_vector(sentence, num_features, dictionary)
		#
		# Increment the counter
		counter = counter + 1
	print()
	return sentenceFeatureVecs


def create_caption_vector(sentence):
	words = [sentence.split()]
	mean_vectors = convert_sentences(np.asarray(words), settings.WORD_EMBEDDING_DIMENSION)
	return mean_vectors[0]


if __name__ == "__main__":
	generate_and_store_caption_vectors(settings.WORD_FILEPATH)
