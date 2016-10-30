import pickle
import sys

import numpy as np  # Make sure that numpy is imported

from caption_database_helper import save_caption_vector
from preprosessing import tokenize
import settings

def getWordVectors(filepath):
	return pickle.load(open(filepath, "rb"))


def getSentences(filepath):
	with open(filepath) as f:
		sentences = []
		for line in f.readlines():
			sentence = []
			for x in ((((line.split("#")[1])[1:]).strip()).split()):
				if x != "." or x != ",":
					sentence.append(x.lower())
			tokenized_sentence = tokenize(sentence)
			sentences.append(tokenized_sentence)

		word_vectors = getWordVectors("%sword_embeddings-%s" % (settings.WORD_EMBEDDING_DIR, settings.WORD_EMBEDDING_DIMENSION))
		mean_vectors = getAvgFeatureVecs(np.asarray(sentences), word_vectors, settings.WORD_EMBEDDING_DIMENSION)

	insertIntoDB(filepath, mean_vectors)


def insertIntoDB(filepath, mean_vectors):
	with open(filepath) as f:
		lineNumber = 0
		for line in f.readlines():
			image_name = line.split("#")[0]
			caption_text = ((line.split("#")[1])[1:]).strip()
			save_caption_vector(image_name, caption_text, mean_vectors[lineNumber])
			if lineNumber % 1000. == 0.:
				print("Inserted %d of %d" % (lineNumber, len(mean_vectors)))
			lineNumber += 1


def makeFeatureVec(words, model, num_features):
	# Function to average all of the word vectors in a given
	# paragraph
	#
	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,), dtype="float32")
	#
	nwords = 0.
	#
	# Index2word is a list that contains the names of the words in
	# the model's vocabulary. Convert it to a set, for speed
	index2word_set = set(model.keys())
	#
	# Loop over each word in the review and, if it is in the model's
	# vocaublary, add its feature vector to the total
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1.
			featureVec = np.add(featureVec, model[word])
	#
	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec, nwords)
	return featureVec


def getAvgFeatureVecs(sentences, model, num_features):
	# Given a set of sentences (each one a list of words), calculate
	# the average feature vector for each one and return a 2D numpy array
	#
	# Initialize a counter
	counter = 0.
	#
	# Preallocate a 2D numpy array, for speed
	len_sentences = len(sentences)
	sentenceFeatureVecs = np.zeros((len_sentences, num_features), dtype="float32")
	#
	# Loop through the reviews
	for sentence in sentences:
		#
		# Print a status message every 1000th sentence
		if counter % 1000. == 0.:
			print("Review %d of %d" % (counter, len_sentences))
		#
		# Call the function (defined above) that makes average feature vectors
		sentenceFeatureVecs[counter] = makeFeatureVec(sentence, model, \
		                                              num_features)
		#
		# Increment the counter
		counter = counter + 1.
	return sentenceFeatureVecs

getSentences(settings.WORD_FILEPATH)
