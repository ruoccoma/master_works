from keras.engine import Input, Model
from keras.layers import Dense, Lambda, merge
# from keras.utils.visualize_util import plot
import tensorflow as tf
import numpy as np

from abstract_text_to_image_architecture import AbstractTextToImageArchitecture
from list_helpers import tf_l2norm
from embeddings_helper import structure_and_store_embeddings
import settings
from sqlite_wrapper import update_database_connection


def contrastive_loss(_, predict):
	s, im = tf.split(1, 2, predict)
	s2 = tf.expand_dims(tf.transpose(s, [0, 1]), 1)
	im2 = tf.expand_dims(tf.transpose(im, [0, 1]), 0)
	diff = im2 - s2
	maximum = tf.maximum(diff, 0.0)
	tensor_pow = tf.square(maximum)
	errors = tf.reduce_sum(tensor_pow, 2)
	diagonal = tf.diag_part(errors)
	cost_s = tf.maximum(0.05 - errors + diagonal, 0.0)
	cost_im = tf.maximum(0.05 - errors + tf.reshape(diagonal, (-1, 1)), 0.0)
	cost_tot = cost_s + cost_im
	zero_diag = tf.mul(diagonal, 0.0)
	cost_tot_diag = tf.matrix_set_diag(cost_tot, zero_diag)
	tot_sum = tf.reduce_sum(cost_tot_diag)
	return tot_sum


class ContrastiveLossArchitecture(AbstractTextToImageArchitecture):
	def __init__(self,
	             epochs=100,
	             batch_size=256,
	             validation_split=0.2,
	             optimizer="adam",
				 word_embedding=settings.WORD_EMBEDDING_METHOD,
				 image_embedding=settings.IMAGE_EMBEDDING_METHOD):
		super(ContrastiveLossArchitecture, self).__init__()
		self.epochs = epochs
		self.batch_size = batch_size
		self.validation_split = validation_split
		self.optimizer = optimizer
		self.loss = contrastive_loss
		self.word_embedding = word_embedding
		self.image_embedding = image_embedding

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

	def generate_model(self):
		image_inputs = Input(shape=(settings.IMAGE_EMBEDDING_DIMENSIONS,), name="Image_input")

		caption_inputs, caption_model = self.get_caption_model()

		merge_layer = merge([caption_model, image_inputs], mode='concat', name="Merge_layer")

		self.model = Model(input=[caption_inputs, image_inputs], output=merge_layer)

	@staticmethod
	def get_caption_model():
		caption_inputs = Input(shape=(settings.WORD_EMBEDDING_DIMENSION,), name="Caption_input")
		caption_model = Lambda(lambda x: tf_l2norm(x), name="Normalize_caption_vector")(caption_inputs)
		caption_model = Lambda(lambda x: abs(x), name="Caption Abs")(caption_model)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(settings.IMAGE_EMBEDDING_DIMENSIONS, activation='relu')(caption_model)
		return caption_inputs, caption_model

	def generate_prediction_model(self):
		weights = self.model.get_weights()
		caption_inputs, caption_model = self.get_caption_model()

		caption_model = Model(input=caption_inputs, output=caption_model)
		caption_model.set_weights(weights)
		caption_model.compile(optimizer=self.optimizer, loss=self.loss)

		self.prediction_model = caption_model


class LargeContrastive(ContrastiveLossArchitecture):
	@staticmethod
	def get_caption_model():
		caption_inputs = Input(shape=(settings.WORD_EMBEDDING_DIMENSION,), name="Caption_input")
		caption_model = Lambda(lambda x: tf_l2norm(x), name="Normalize_caption_vector")(caption_inputs)
		caption_model = Lambda(lambda x: abs(x), name="Caption Abs")(caption_model)
		caption_model = Dense(500, activation='relu')(caption_model)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(2048, activation='relu')(caption_model)
		caption_model = Dense(settings.IMAGE_EMBEDDING_DIMENSIONS, activation='relu')(caption_model)
		return caption_inputs, caption_model


class Word2visualVecContrastive(ContrastiveLossArchitecture):
	@staticmethod
	def get_caption_model():
		caption_inputs = Input(shape=(settings.WORD_EMBEDDING_DIMENSION,), name="Caption_input")
		caption_model = Lambda(lambda x: tf_l2norm(x), name="Normalize_caption_vector")(caption_inputs)
		caption_model = Lambda(lambda x: abs(x), name="Caption Abs")(caption_model)
		caption_model = Dense(500, activation='relu')(caption_model)
		caption_model = Dense(1000, activation='relu')(caption_model)
		caption_model = Dense(settings.IMAGE_EMBEDDING_DIMENSIONS, activation='relu')(caption_model)
		return caption_inputs, caption_model
