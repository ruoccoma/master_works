import tensorflow as tf
from keras.layers import Embedding, GRU, Merge
from keras.layers import Input, Dense
from keras.layers.core import Lambda, Masking
from keras.models import Model

import datasource
from datasets import build_dictionary
from datasets import load_dataset


def tf_l2norm(tensor_array):
	norm = tf.sqrt(tf.reduce_sum(tf.pow(tensor_array, 2)))
	tensor_array /= norm
	return tensor_array


def contrastive_loss_keras(_, predict):
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


# main trainer
def train(params):
	# GRID SEARCH
	global model_config
	model_config['margin'] = params['margin'] if 'margin' in params else model_config['margin']
	model_config['output_dim'] = params['output_dim'] if 'output_dim' in params else model_config['output_dim']
	model_config['max_cap_length'] = params['max_cap_length'] if 'max_cap_length' in params else model_config[
		'max_cap_length']
	model_config['optimizer'] = params['optimizer'] if 'optimizer' in params else model_config['optimizer'],
	model_config['dim_word'] = params['dim_word'] if 'dim_word' in params else model_config['dim_word']

	# Load training and development sets
	print('Loading dataset')
	dataset = load_dataset()

	train = dataset["train"]
	test = dataset["test"]

	# Create dictionary
	print('Creating dictionary')

	worddict = build_dictionary(train['caps'])
	print('Dictionary size: ' + str(len(worddict)))
	model_config['worddict'] = len(worddict)

	print('Loading data')
	train_iter = datasource.Datasource(train, batch_size=model_config['batch_size'], worddict=worddict)

	print("Image model loading")
	# # this returns a tensor of emb_image
	image_input = Input(shape=(model_config['dim_cnn'],), name='image_input')
	image_model = Dense(model_config['output_dim'], name="image_dense_layer")(image_input)
	image_model = Lambda(lambda x: tf_l2norm(x))(image_model)
	emb_image = Lambda(lambda x: abs(x))(image_model)

	print("Text model loading")
	# this returns a tensor of emb_cap
	cap_input = Input(shape=(model_config['max_cap_length'],), dtype='float32', name='cap_input')
	X = Masking(mask_value=0, input_shape=(model_config['max_cap_length'], model_config['output_dim']))(cap_input)
	X = Embedding(output_dim=model_config['dim_word'], input_dim=model_config['worddict'],
	              input_length=model_config['max_cap_length'])(cap_input)
	X = GRU(output_dim=model_config['output_dim'], return_sequences=False)(X)
	X = Lambda(lambda x: tf_l2norm(x))(X)
	emb_cap = Lambda(lambda x: abs(x))(X)

	print("loading the joined model")
	merged = Merge(mode='concat')([emb_cap, emb_image])
	model = Model(input=[cap_input, image_input], output=[merged])

	print("compiling the model")
	model.compile(optimizer=model_config['optimizer'][0], loss=contrastive_loss_keras)

	training_data = train_iter.all()

	print("Fitting model...")
	model.fit([training_data[0], training_data[1]], training_data[0], validation_split=0.2, nb_epoch=300)

	model.save_weights('my_model_weights.h5')

	def eval_model():
		print('evaluating model...')
		weights = model.get_weights()
		emb_w = weights[0]
		im_w = weights[10]
		im_b = weights[11]
		gru_weights = weights[1:10]

		test_model_im = Model(input=image_input, output=emb_image)
		test_model_im.set_weights([im_w, im_b])
		test_model_im.compile(optimizer='adam', loss=contrastive_loss_keras)
		test_model_cap = Model(input=cap_input, output=emb_cap)
		test_model_cap.set_weights([emb_w] + gru_weights)
		test_model_cap.compile(optimizer='adam', loss=contrastive_loss_keras)

		test_cap, test_im = test_iter.all()
		all_caps = numpy.zeros(shape=(len(test_cap), model_config['max_cap_length']))
		all_images = numpy.zeros(shape=(len(test_cap), model_config['dim_cnn']))
		pred_cap = test_model_cap.predict(test_cap)
		pred_im = test_model_im.predict(test_im)

	eval_model()

model_config = {}

def trainer(config):
	global model_config
	model_config = config
	train(model_config)
