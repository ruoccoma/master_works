import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Dense, Lambda, merge
# from keras.utils.visualize_util import plot
import tensorflow as tf

from abstract_word2visualvec_architecture import AbstractWord2VisualVecArchitecture
from embeddings_helper import structure_and_store_embeddings
from list_helpers import tf_l2norm


def contrastive_loss(y_true, y_pred):
	x, y = tf.split(1, 2, y_pred)
	print(tf.shape(x))
	print(tf.shape(y))
	d = tf.reduce_sum(tf.square(x - y), 1)
	d_sqrt = tf.sqrt(d)

	margin = 1

	loss = y_true * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - y_true) * d
	return 0.5 * loss


class ContrastiveLossArchitecture(AbstractWord2VisualVecArchitecture):
	def __init__(self,
	             epochs=50,
	             batch_size=128,
	             validation_split=0.2,
	             optimizer="adam"):
		super(ContrastiveLossArchitecture, self).__init__()
		self.epochs = epochs
		self.batch_size = batch_size
		self.validation_split = validation_split
		self.optimizer = optimizer
		self.loss = contrastive_loss

	def train(self):
		caption_vectors, image_vectors, similarities = structure_and_store_embeddings()

		caption_vectors = np.asarray(caption_vectors)
		image_vectors = np.asarray(image_vectors)

		self.generate_model()

		self.model.compile(optimizer=self.optimizer, loss=self.loss)
		self.model.summary()
		# plot(self.model, 'results/%s.png' % self.get_name())
		self.model.fit([caption_vectors, image_vectors],
		               similarities,
		               batch_size=self.batch_size,
		               nb_epoch=self.epochs,
		               callbacks=self.callbacks,
		               shuffle=True,
		               validation_split=self.validation_split)

	def generate_model(self):
		image_inputs = Input(shape=(4096,), name="Image_input")
		image_model = Lambda(lambda x: abs(x), name="Image Abs")(image_inputs)

		caption_inputs, caption_model = self.get_caption_model()

		norm_image_model = Lambda(lambda x: K.l2_normalize(x, axis=1), name="Image_norm")(image_model)
		norm_caption_model = Lambda(lambda x: K.l2_normalize(x, axis=1), name="Caption_norm")(caption_model)

		distance = merge([norm_caption_model, norm_image_model], mode='concat', name="Merge_layer")

		self.model = Model(input=[caption_inputs, image_inputs], output=distance)

	@staticmethod
	def get_caption_model():
		caption_inputs = Input(shape=(300,), name="Caption_input")
		caption_model = Lambda(lambda x: tf_l2norm(x), name="Normalize_caption_vector")(caption_inputs)
		caption_model = Lambda(lambda x: abs(x), name="Caption Abs")(caption_model)
		caption_model = Dense(500, activation='relu')(caption_model)
		caption_model = Dense(800, activation='relu')(caption_model)
		caption_model = Dense(1024, activation='relu')(caption_model)
		caption_model = Dense(4096, activation='relu')(caption_model)
		return caption_inputs, caption_model

	def generate_prediction_model(self):
		weights = self.model.get_weights()
		caption_inputs, caption_model = self.get_caption_model()

		caption_model = Model(input=caption_inputs, output=caption_model)
		caption_model.set_weights(weights)
		caption_model.compile(optimizer=self.optimizer, loss=self.loss)

		self.prediction_model = caption_model
