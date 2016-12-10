import numpy
import numpy as np
import settings
from caption_database_helper import fetch_all_filename_caption_vector_tuples
from contrastive_loss_architecture import ContrastiveLossArchitecture
from embeddings_helper import structure_and_store_embeddings
from image_database_helper import fetch_all_image_vector_pairs
from keras.engine import Input
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from list_helpers import print_progress, totuple, split_list
from sklearn.metrics.pairwise import cosine_similarity


class LSTMEmbeddingArchitecture(ContrastiveLossArchitecture):
	def __init__(self,
				 epochs=100,
				 batch_size=256,
				 validation_split=0.2,
				 optimizer="adam",
				 image_embedding=settings.IMAGE_EMBEDDING_METHOD):
		super(LSTMEmbeddingArchitecture, self).__init__(epochs=epochs, batch_size=batch_size,
														validation_split=validation_split, optimizer=optimizer,
														word_embedding="sequence", image_embedding=image_embedding)

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

	def evaluate(self):
		te_ca_caption_vectors = fetch_test_captions_vectors()
		predicted_image_vectors = self.prediction_model.predict(te_ca_caption_vectors)

		tr_im_filename_image_vector_tuples = fetch_all_image_vector_pairs()
		tr_im_filenames = [x[0] for x in tr_im_filename_image_vector_tuples]
		tr_im_image_vectors = [x[1] for x in tr_im_filename_image_vector_tuples]

		similarity_matrix = cosine_similarity(predicted_image_vectors, tr_im_image_vectors)

		tr_ca_caption_vector_tuples = fetch_all_filename_caption_vector_tuples()

		tr_ca_caption_vector_filename_dictionary = build_caption_vector_filename_dict(tr_ca_caption_vector_tuples)

		print("Creating cosine similarity matrix...")
		predicted_images_size = len(predicted_image_vectors)
		total_image_size = len(tr_im_image_vectors)
		for predicted_image_index in range(predicted_images_size):
			similarities = []
			for i in range(total_image_size):
				tr_filename = tr_im_filenames[i]
				similarities.append((tr_filename, similarity_matrix[predicted_image_index][i]))
			similarities.sort(key=lambda s: s[1], reverse=True)

			test_caption_vector = te_ca_caption_vectors[predicted_image_index]
			test_caption_vector_key = totuple(test_caption_vector)
			test_filename = tr_ca_caption_vector_filename_dictionary[test_caption_vector_key]

			print("Q: %s" % test_filename)

			for top_image_index in range(1000):
				comparison_filename = similarities[top_image_index][0]
				if top_image_index < 5:
					print("Rank %s: %s, %s" % (
						top_image_index, similarities[top_image_index][0], similarities[top_image_index][1]))
			print_progress(predicted_image_index + 1, predicted_images_size, prefix="Calculating recall")
		return 0, 0, 0, 0, 0, 0

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


def fetch_test_captions_vectors():
	data_x, _, _ = structure_and_store_embeddings()
	# training_test_ratio = 0.8
	# _, test_x = split_list(data_x, training_test_ratio)
	training_test_ratio = 0.2
	test_x, _ = split_list(data_x, training_test_ratio)
	return numpy.asarray(test_x)


def build_caption_vector_filename_dict(filename_caption_vector_tuples):
	caption_vector_filename_dictionary = {}
	total_filname_caption_vector = len(filename_caption_vector_tuples)
	for i in range(total_filname_caption_vector):
		filename, cap_vec = filename_caption_vector_tuples[i]
		tuple_key = totuple(cap_vec)
		caption_vector_filename_dictionary[tuple_key] = filename
		if i % 1000 == 0 or i > total_filname_caption_vector - 5:
			print_progress(i + 1, total_filname_caption_vector, prefix="Building cap vec -> filename dict",
						   barLength=50)
	return caption_vector_filename_dictionary
