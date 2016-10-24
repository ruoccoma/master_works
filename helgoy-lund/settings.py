import os

ROOT_DIR = os.path.dirname((os.path.abspath(__file__)))
DB_FILE_PATH = ROOT_DIR + "/data/databases/sqlite/database.db"

CNN_NAME = "inception"
WORD_EMBEDDING_METHOD = "word2vec"
WORD_EMBEDDING_DIMENSION = 128

# Stored embeddings
STORED_EMBEDDINGS_DIR = ROOT_DIR + "/data/embeddings/stored-embeddings/"
STORED_EMBEDDINGS_PREFIX = "%s%s-%s-" % (WORD_EMBEDDING_DIMENSION, WORD_EMBEDDING_METHOD, CNN_NAME)