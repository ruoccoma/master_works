import os

ROOT_DIR = os.path.dirname((os.path.abspath(__file__)))
DB_FILE_PATH = ROOT_DIR + "/data/databases/sqlite/database.db"

CNN_NAME = "inception"
WORD_EMBEDDING_METHOD = "word2vec"
IMAGE_DIR = ROOT_DIR + "/data/datasets/Flickr_8k/images/"

# Word2Vec
WORD_EMBEDDING_DIMENSION = 128
WORD_EMBEDDING_DIR = ROOT_DIR + "/models/word2vec/embeddings/"
WORD_FILEPATH = ROOT_DIR + "/data/datasets/Flickr_8k/Flickr8k.token.txt"

# Stored embeddings
STORED_EMBEDDINGS_DIR = ROOT_DIR + "/data/embeddings/stored-embeddings/"
STORED_EMBEDDINGS_PREFIX = "%s%s-%s-" % (WORD_EMBEDDING_DIMENSION, WORD_EMBEDDING_METHOD, CNN_NAME)