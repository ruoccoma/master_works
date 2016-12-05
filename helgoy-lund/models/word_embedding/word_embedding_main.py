import settings
from glove.glove import run_glove
from word2vec.word2vec_basic import run_word2vec

if settings.WORD_EMBEDDING_METHOD == "word2vec":
	# TODO Wrap word2vec in method to run here. Global variable data_index needs to be outside
	# run_word2vec()
	print("Wrap word2vec in method to run here")
elif settings.WORD_EMBEDDING_METHOD == "glove":
	run_glove()
