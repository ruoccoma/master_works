import os
import sys


def add_modules_to_sys_path(root_dir):
	sys.path.append(root_dir + "data/databases/helpers")
	sys.path.append(root_dir + "data/embeddings/helpers")
	sys.path.append(root_dir + "data/embeddings/plotting")
	sys.path.append(root_dir + "data/embeddings/testing")
	sys.path.append(root_dir + "helpers")
	sys.path.append(root_dir + "models/word_embedding")
	sys.path.append(root_dir + "models/word_embedding/word2vec")
	sys.path.append(root_dir + "models/word_embedding/glove")
	sys.path.append(root_dir + "models/image_embedding")
	sys.path.append(root_dir + "models/image_embedding/inception")
	sys.path.append(root_dir + "models/image_embedding/vgg")
	sys.path.append(root_dir + "models/word2visualvec")
	sys.path.append(root_dir + "models/word2visualvec/old_architectures")
	sys.path.append(root_dir + "models/word2visualvec/results")

ROOT_DIR = os.path.dirname((os.path.abspath(__file__))) + "/"

add_modules_to_sys_path(ROOT_DIR)

# vgg | inception
IMAGE_EMBEDDING_METHOD = "inception"

# word2vec | glove
WORD_EMBEDDING_METHOD = "word2vec"

# Flickr8k | Flickr30k
DATASET = "Flickr8k"

RES_DIR = ROOT_DIR + "res/"
IMAGE_DIR = ROOT_DIR + "data/datasets/" + DATASET + "/images/"
CREATE_NEGATIVE_EXAMPLES = False

DB_SUFFIX = "%s-%s-%s" % (IMAGE_EMBEDDING_METHOD, WORD_EMBEDDING_METHOD, DATASET)
DB_FILE_PATH = ROOT_DIR + "/data/databases/sqlite/database-%s.db" % DB_SUFFIX

# Word2Vec
WORD_EMBEDDING_DIMENSION = 300
WORD_EMBEDDING_DIR = ROOT_DIR + "models/word2vec/embeddings/"
WORD_FILEPATH = ROOT_DIR + "data/datasets/Flickr30k/flickr30k/results_20130124.token"

# Stored embeddings
STORED_EMBEDDINGS_DIR = ROOT_DIR + "data/embeddings/stored-embeddings/"
NEG_TAG = "neg" if CREATE_NEGATIVE_EXAMPLES else ""
STORED_EMBEDDINGS_NAME = "%s-%s" % (DB_SUFFIX, NEG_TAG)

RESULT_TEXTFILE_PATH = ROOT_DIR + "models/word2visualvec/results/results.txt"

