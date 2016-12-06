import numpy as np
import settings
from embeddings_helper import structure_and_store_embeddings
from keras.engine import Input
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from contrastive_loss_architecture import ContrastiveLossArchitecture
from sqlite_wrapper import update_database_connection


class LSTMEmbeddingArchitecture(ContrastiveLossArchitecture):

	def __init__(self,
	             epochs=100,
	             batch_size=256,
	             validation_split=0.2,
	             optimizer="adam",
				 image_embedding=settings.IMAGE_EMBEDDING_METHOD):
		super(LSTMEmbeddingArchitecture, self).__init__(epochs=epochs, batch_size=batch_size, validation_split=validation_split, optimizer=optimizer, word_embedding="sequence", image_embedding=image_embedding)

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
		caption_inputs = Input(shape=(82,), name="Caption_input")
		caption_embedding = Embedding(20323 + 1,
									  settings.WORD_EMBEDDING_DIMENSION,
									  input_length=82)(caption_inputs)
		caption_model = LSTM(300)(caption_embedding)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(settings.IMAGE_EMBEDDING_DIMENSIONS, activation='relu')(caption_model)
		return caption_inputs, caption_model


class TwoLSTMEmbeddingArchitecture(LSTMEmbeddingArchitecture):
	@staticmethod
	def get_caption_model():
		caption_inputs = Input(shape=(82,), name="Caption_input")
		caption_embedding = Embedding(20323 + 1,
									  settings.WORD_EMBEDDING_DIMENSION,
									  input_length=82)(caption_inputs)
		caption_model = LSTM(500)(caption_embedding)
		caption_model = LSTM(500)(caption_model)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(settings.IMAGE_EMBEDDING_DIMENSIONS, activation='relu')(caption_model)
		return caption_inputs, caption_model

class FourHiddenLSTMEmbeddingArchitecture(LSTMEmbeddingArchitecture):
	@staticmethod
	def get_caption_model():
		caption_inputs = Input(shape=(82,), name="Caption_input")
		caption_embedding = Embedding(20323 + 1,
									  settings.WORD_EMBEDDING_DIMENSION,
									  input_length=82)(caption_inputs)
		caption_model = LSTM(500)(caption_embedding)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(2048, activation='relu')(caption_model)
		caption_model = Dense(2048, activation='relu')(caption_model)
		caption_model = Dense(settings.IMAGE_EMBEDDING_DIMENSIONS, activation='relu')(caption_model)
		return caption_inputs, caption_model


class BiLSTMEmbeddingArchitecture(LSTMEmbeddingArchitecture):
	@staticmethod
	def get_caption_model():
		caption_inputs = Input(shape=(82,), name="Caption_input")
		caption_embedding = Embedding(20323 + 1,
									  settings.WORD_EMBEDDING_DIMENSION,
									  input_length=82)(caption_inputs)
		caption_model = Bidirectional(LSTM(500))(caption_embedding)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(settings.IMAGE_EMBEDDING_DIMENSIONS, activation='relu')(caption_model)
		return caption_inputs, caption_model


class TwoBiLSTMEmbeddingArchitecture(LSTMEmbeddingArchitecture):
	@staticmethod
	def get_caption_model():
		caption_inputs = Input(shape=(82,), name="Caption_input")
		caption_embedding = Embedding(20323 + 1,
									  settings.WORD_EMBEDDING_DIMENSION,
									  input_length=82)(caption_inputs)
		caption_model = Bidirectional(LSTM(300))(caption_embedding)
		caption_model = Bidirectional(LSTM(500))(caption_model)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(settings.IMAGE_EMBEDDING_DIMENSIONS, activation='relu')(caption_model)
		return caption_inputs, caption_model
