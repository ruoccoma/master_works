import numpy as np
from keras import callbacks
from keras.engine import Input, Model
from keras.layers import Dense, Lambda, Dropout
from keras.layers import Merge
from keras.utils.visualize_util import plot
from keras import backend as K


from embeddings_helper import structure_and_store_embeddings
from list_helpers import theano_l2norm, tf_l2norm

remote = callbacks.RemoteMonitor(root='http://127.0.0.1:9000')


def get_optimizer():
	return optimizer


def get_loss():
	return loss


def get_epochs():
	return epochs


def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def hinge_onehot(y_true, y_pred):
	y_true = y_true*2 - 1
	y_pred = y_pred*2 - 1

	return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)


# hyperparams
epochs = 100
batch_size = 256
validation_split = 0.2
optimizer = "adam"
loss = hinge_onehot


def train():
	caption_vectors, image_vectors, similarities = structure_and_store_embeddings()

	caption_vectors = np.asarray(caption_vectors)
	image_vectors = np.asarray(image_vectors)

	merged_model = get_model()

	merged_model.compile(optimizer=optimizer, loss=loss)

	plot(merged_model, 'merged-euclid-model.png')
	merged_model.fit([caption_vectors, image_vectors], similarities, batch_size=batch_size, nb_epoch=epochs,
	                 callbacks=[remote],
					 shuffle=True,
	                 validation_split=validation_split)

	return merged_model


def get_model():
	image_inputs = Input(shape=(2048,), name="Image_input")
	image_model = Lambda(lambda x: abs(x), name="Image Abs")(image_inputs)

	caption_inputs, caption_model = get_caption_model()

	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([caption_model, image_model])
	model = Model(input=[caption_inputs, image_inputs], output=distance)

	return model


def get_caption_model():
	caption_inputs = Input(shape=(300,), name="Caption_input")
	caption_model = Lambda(lambda x: tf_l2norm(x), name="Normalize_caption_vector")(caption_inputs)
	caption_model = Lambda(lambda x: abs(x), name="Caption Abs")(caption_model)
	caption_model = Dense(500, activation='relu')(caption_model)
	caption_model = Dropout(0.2)(caption_model)
	caption_model = Dense(800, activation='relu')(caption_model)
	caption_model = Dropout(0.2)(caption_model)
	caption_model = Dense(1024, activation='relu')(caption_model)
	caption_model = Dropout(0.2)(caption_model)
	caption_model = Dense(2048, activation='relu')(caption_model)
	return caption_inputs, caption_model


def generate_prediction_model(model):
	weights = model.get_weights()
	caption_inputs, caption_model = get_caption_model()

	caption_model = Model(input=caption_inputs, output=caption_model)
	caption_model.set_weights(weights)
	caption_model.compile(optimizer=optimizer, loss=loss)

	return caption_model
