import numpy as np
from keras.engine import Input, Model
from keras.layers import Dense, Lambda, Dropout, BatchNormalization, Embedding, Conv1D, MaxPooling1D, Flatten, LSTM
from euclidian_distance_architecture import EuclidianDistanceArchitecture
from embeddings_helper import structure_and_store_embeddings
import settings


class LSTMEmbeddingArchitecture(EuclidianDistanceArchitecture):

	def train(self):
		caption_vectors, image_vectors, similarities = structure_and_store_embeddings()

		caption_vectors = np.asarray(caption_vectors)
		image_vectors = np.asarray(image_vectors)

		self.generate_model()

		self.model.compile(optimizer=self.optimizer, loss=self.loss)

		# plot(self.model, 'results/%s.png' % self.get_name())
		self.model.fit([caption_vectors, image_vectors],
		               similarities,
		               batch_size=self.batch_size,
		               nb_epoch=self.epochs,
		               callbacks=self.callbacks,
		               shuffle=True,
		               validation_split=self.validation_split)

	@staticmethod
	def get_caption_model():
		embedding_layer = Embedding(20323 + 1,
									settings.WORD_EMBEDDING_DIMENSION,
									input_length=82)

		caption_embedding = embedding_layer()
		caption_model = LSTM(100)(caption_embedding)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(2048, activation='relu')(caption_model)
		return embedding_layer, caption_model
