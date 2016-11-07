import os
import sys


def add_modules_to_sys_path(root_dir):
	sys.path.append(root_dir + "data/databases/helpers")
	sys.path.append(root_dir + "data/embeddings/helpers")
	sys.path.append(root_dir + "data/embeddings/plotting")
	sys.path.append(root_dir + "data/embeddings/testing")
	sys.path.append(root_dir + "helpers")
	sys.path.append(root_dir + "models/word2vec")
	sys.path.append(root_dir + "models/word2visualvec")

ROOT_DIR = os.path.dirname((os.path.abspath(__file__))) + "/"

add_modules_to_sys_path(ROOT_DIR)

DB_FILE_PATH = ROOT_DIR + "/data/databases/sqlite/database.db"

CNN_NAME = "inception"
WORD_EMBEDDING_METHOD = "word2vec"
DATASET = "Flickr8k"
RES_DIR = ROOT_DIR + "res/"
IMAGE_DIR = ROOT_DIR + "data/datasets/" + DATASET + "/images/"
CREATE_NEGATIVE_EXAMPLES = False

# Word2Vec
WORD_EMBEDDING_DIMENSION = 300
WORD_EMBEDDING_DIR = ROOT_DIR + "models/word2vec/embeddings/"
WORD_FILEPATH = ROOT_DIR + "data/datasets/Flickr30k/flickr30k/results_20130124.token"

# Stored embeddings
STORED_EMBEDDINGS_DIR = ROOT_DIR + "data/embeddings/stored-embeddings/"
NEG_TAG = "neg" if CREATE_NEGATIVE_EXAMPLES else ""
STORED_EMBEDDINGS_PREFIX = "%s%s-%s-%s-%s" % (WORD_EMBEDDING_DIMENSION, WORD_EMBEDDING_METHOD, CNN_NAME, DATASET, NEG_TAG)

