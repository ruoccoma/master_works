import numpy as np
import theano
import theano.tensor as tensor
from keras import callbacks
from keras.engine import Input, Model
from keras.layers import Dense, Lambda
from keras.layers import Merge
from keras.models import Sequential
from keras.utils.visualize_util import plot
from theano.tensor.extra_ops import fill_diagonal

from embeddings_helper import structure_and_store_embeddings
from list_helpers import tf_l2norm, theano_l2norm

remote = callbacks.RemoteMonitor(root='http://127.0.0.1:9000')


def get_optimizer():
	return optimizer


def get_loss():
	return loss


def contrastive_loss(labels, predict):
	"""For a minibatch of sentence and image embeddings, compute the pairwise contrastive loss"""
	margin = 0.05
	res = theano.tensor.split(predict, [2048, 2048], 2, axis=-1)
	s = res[0]
	im = res[1]
	im2 = im.dimshuffle(('x', 0, 1))
	s2 = s.dimshuffle((0, 'x', 1))
	errors = tensor.pow(tensor.maximum(0, im2 - s2), 2).sum(axis=2)
	diagonal = errors.diagonal()
	# compare every diagonal score to scores in its column (all contrastive images for each sentence)
	cost_s = tensor.maximum(0, margin - errors + diagonal)
	# all contrastive sentences for each image
	cost_im = tensor.maximum(0, margin - errors + diagonal.reshape((-1, 1)))
	cost_tot = cost_s + cost_im
	cost_tot = fill_diagonal(cost_tot, 0)
	return cost_tot.sum()


# hyperparams
epochs = 10
batch_size = 256
validation_split = 0.2
optimizer = "adam"
loss = contrastive_loss


def train():
	caption_vectors, image_vectors = structure_and_store_embeddings()
	similarities = np.ones(len(caption_vectors))

	caption_vectors = np.asarray(caption_vectors)
	image_vectors = np.asarray(image_vectors)

	merged_model = get_model()

	merged_model.compile(optimizer=optimizer, loss=loss)

	plot(merged_model, to_file='model.png')
	merged_model.fit([caption_vectors, image_vectors], similarities, batch_size=batch_size, nb_epoch=epochs,
	                 callbacks=[remote],
	                 validation_split=0.2)

	return merged_model


def get_model():
	image_inputs = Input(shape=(2048,), name="Image_input")
	caption_inputs, caption_model = get_caption_model()
	image_model = Lambda(lambda x: abs(x), name="Image Abs")(image_inputs)
	merged_model = Merge(mode="concat", dot_axes=1, name="Merge: cosine distance")([caption_model, image_model])
	# cos_distance = Reshape((1,))(cos_distance)
	# cos_similarity = Lambda(lambda x: 1 - x, name="cos_sim")(cos_distance)
	merged_model = Model(input=[caption_inputs, image_inputs], output=[merged_model])
	# merged_model.summary()
	return merged_model


def get_caption_model():
	caption_inputs = Input(shape=(300,), name="Caption_input")
	caption_model = Lambda(lambda x: theano_l2norm(x), name="Normalize_caption_vector")(caption_inputs)
	caption_model = Lambda(lambda x: abs(x), name="Caption Abs")(caption_model)
	caption_model = Dense(400, activation='relu')(caption_model)
	caption_model = Dense(800, activation='relu')(caption_model)
	caption_model = Dense(1024, activation='relu')(caption_model)
	caption_model = Dense(2048, activation='relu')(caption_model)
	return caption_inputs, caption_model


def train_sequential():
	caption_vectors, image_vectors = structure_and_store_embeddings()
	similarities = [1.0 for x in range(len(caption_vectors))]

	caption_inputs = Input(shape=(300,))
	image_inputs = Input(shape=(2048,))

	caption_model = Sequential()
	caption_model.add(Dense(400, activation='relu', input_dim=caption_inputs))
	caption_model.add(Dense(800, activation='relu'))
	caption_model.add(Dense(1024, activation='relu'))
	caption_model.add(Lambda(lambda x: tf_l2norm(x)))
	caption_model.add(Lambda(lambda x: abs(x)))

	image_model = Sequential()
	image_model.add(Lambda(lambda x: abs(x)))

	merge = Merge([caption_model, image_model], mode="cos")
	merged_model = Model(input=[caption_inputs, image_inputs], output=[merge])

	merged_model.compile(optimizer=optimizer, loss=loss)
	merged_model.fit([caption_vectors, image_vectors], similarities)


def generate_prediction_model(model):
	weights = model.get_weights()
	caption_inputs, caption_model = get_caption_model()

	caption_model = Model(input=caption_inputs, output=caption_model)
	caption_model.set_weights(weights)
	caption_model.compile(optimizer=optimizer, loss=loss)

	return caption_model
