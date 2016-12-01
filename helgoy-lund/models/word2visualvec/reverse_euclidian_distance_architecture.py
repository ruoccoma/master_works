import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Dense, Lambda, Dropout, BatchNormalization
# from keras.utils.visualize_util import plot

from abstract_word2visualvec_architecture import AbstractWord2VisualVecArchitecture
from embeddings_helper import structure_and_store_embeddings
from list_helpers import tf_l2norm


def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return shape1[0], 1


# vgg_w2v r1000: 0.386
class ReversedEuclidanDistanceArchitecture(AbstractWord2VisualVecArchitecture):
	def __init__(self,
				 epochs=100,
				 batch_size=256,
				 validation_split=0.2,
				 optimizer="adam"):
		super(ReversedEuclidanDistanceArchitecture, self).__init__()
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

		# plot(self.model, 'results/%s.png' % self.get_name())
		self.model.fit([image_vectors, caption_vectors],
					   similarities,
					   batch_size=self.batch_size,
					   nb_epoch=self.epochs,
					   callbacks=self.callbacks,
					   shuffle=True,
					   validation_split=self.validation_split)

	def generate_model(self):
		caption_inputs = Input(shape=(300,), name="Caption_input")
		caption_model = Lambda(lambda x: abs(x), name="Caption Abs")(caption_inputs)

		image_inputs, image_model = self.get_image_model()

		distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([caption_model, image_model])
		self.model = Model(input=[image_inputs, caption_inputs], output=distance)

	@staticmethod
	def get_image_model():
		image_input = Input(shape=(4096,), name="Image_input")
		image_model = Lambda(lambda x: abs(x), name="Image Abs")(image_input)
		image_model = Dense(1024, activation='relu')(image_model)
		image_model = Dense(800, activation='relu')(image_model)
		image_model = Dense(500, activation='relu')(image_model)
		image_model = Dense(300, activation='relu')(image_model)
		return image_input, image_model

	def generate_prediction_model(self):
		weights = self.model.get_weights()
		image_inputs, image_model = self.get_image_model()

		image_model = Model(input=image_inputs, output=image_model)
		image_model.set_weights(weights)
		image_model.compile(optimizer=self.optimizer, loss=self.loss)

		self.prediction_model = image_model
