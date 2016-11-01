import os

ROOT_DIR = os.path.dirname((os.path.abspath(__file__)))
DB_FILE_PATH = ROOT_DIR + "/data/databases/sqlite/database.db"

CNN_NAME = "inception"
WORD_EMBEDDING_METHOD = "word2vec"
DATASET = "Flickr30k"
RES_DIR = ROOT_DIR + "/res/"
IMAGE_DIR = ROOT_DIR + "/data/datasets/" + DATASET + "/images/"

# Word2Vec
WORD_EMBEDDING_DIMENSION = 300
WORD_EMBEDDING_DIR = ROOT_DIR + "/models/word2vec/embeddings/"
WORD_FILEPATH = ROOT_DIR + "/data/datasets/Flickr30k/flickr30k/results_20130124.token"

# Stored embeddings
STORED_EMBEDDINGS_DIR = ROOT_DIR + "/data/embeddings/stored-embeddings/"
STORED_EMBEDDINGS_PREFIX = "%s%s-%s-%s-" % (WORD_EMBEDDING_DIMENSION, WORD_EMBEDDING_METHOD, CNN_NAME, DATASET)
