import settings
from glove.glove import run_glove
from word2vec.word2vec_basic import run_word2vec

if settings.WORD_EMBEDDING_METHOD == "word2vec":
	run_word2vec()
elif settings.WORD_EMBEDDING_METHOD == "glove":
	run_glove()