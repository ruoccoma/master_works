import pickle
import sys

import numpy as np  # Make sure that numpy is imported

from caption_database_helper import db_insert_caption_vector


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
			sentences.append(sentence)

		mean_vectors = getAvgFeatureVecs(np.asarray(sentences), getWordVectors(WORD_VECTOR_FILEPATH), 128)

	insertIntoDB(filepath, mean_vectors)


def insertIntoDB(filepath, mean_vectors):
	with open(filepath) as f:
		lineNumber = 0
		for line in f.readlines():
			db_insert_caption_vector(line.split("#")[0], ((line.split("#")[1])[1:]).strip(), mean_vectors[lineNumber])
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
	sentenceFeatureVecs = np.zeros((len(sentences), num_features), dtype="float32")
	#
	# Loop through the reviews
	for sentence in sentences:
		#
		# Print a status message every 1000th sentence
		if counter % 1000. == 0.:
			print("Review %d of %d" % (counter, len(sentences)))
		#
		# Call the function (defined above) that makes average feature vectors
		sentenceFeatureVecs[counter] = makeFeatureVec(sentence, model, \
		                                              num_features)
		#
		# Increment the counter
		counter = counter + 1.
	return sentenceFeatureVecs


FILEPATH = ""
for path in sys.path:
	if path.endswith("master_works"):
		FILEPATH = path
		break

WORD_VECTOR_FILEPATH = FILEPATH + "/helgoy-lund/word2vec/word_embeddings-128"
DATA_FILEPATH = FILEPATH + "/helgoy-lund/word2visualvec/datasets/Flickr8k/Flickr8k.token.txt"

getSentences(DATA_FILEPATH)
