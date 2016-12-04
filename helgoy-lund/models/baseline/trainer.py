import sys

import numpy
import tensorflow as tf
from keras import callbacks
from keras.layers import Embedding, GRU, Merge
from keras.layers import Input, Dense
from keras.layers.core import Lambda, Masking
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

import datasource
import settings
from custom_callback import WriteToFileCallback
from datasets import build_dictionary
from datasets import load_dataset
from image_database_helper import fetch_all_image_vector_pairs, print_progress
from io_helper import save_pickle_file, load_pickle_file, check_pickle_file


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


def train(params, eval_mode=False):
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
	dataset = load_dataset(evaluate_mode=eval_mode)

	train = dataset["train"]
	test = dataset["test"]

	# Create dictionary
	print('Creating dictionary')

	worddict = build_dictionary(train['caps'])
	print('Dictionary size: ' + str(len(worddict)))
	model_config['worddict'] = len(worddict)

	print('Loading data')
	train_iter = datasource.Datasource(train, batch_size=model_config['batch_size'], worddict=worddict,
	                                   max_cap_lengh=model_config['max_cap_length'])

	test_iter = datasource.Datasource(test, batch_size=model_config['batch_size'], worddict=worddict,
	                                  max_cap_lengh=model_config['max_cap_length'], eval_mode=True)

	model, image_input, emb_image, cap_input, emb_cap = generate_model(model_config)

	training_data = train_iter.all()

	train_caps = training_data[0]
	train_ims = training_data[1]

	result_file = open(settings.RESULT_TEXTFILE_PATH, "a")
	result_file.write("BASELINE\n")
	result_file.close()

	if eval_mode:
		model.load_weights("my_model_weights.h5")
		eval_model(model, image_input, emb_image, cap_input, emb_cap, test_iter)
	else:
		print("Fitting model...")
		custom_callback = WriteToFileCallback(settings.RESULT_TEXTFILE_PATH)
		early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)
		model.fit([train_caps, train_ims], train_caps, validation_split=0.2, nb_epoch=300,
		          callbacks=[custom_callback, early_stopping])

		model.save_weights('my_model_weights.h5')


def generate_model(model_conf):
	print("Image model loading")
	# # this returns a tensor of emb_image
	image_input = Input(shape=(model_conf['dim_cnn'],), name='image_input')
	image_model = Dense(model_conf['output_dim'], name="image_dense_layer")(image_input)
	image_model = Lambda(lambda x: tf_l2norm(x))(image_model)
	emb_image = Lambda(lambda x: abs(x))(image_model)

	print("Text model loading")
	# this returns a tensor of emb_cap
	cap_input = Input(shape=(model_conf['max_cap_length'],), dtype='float32', name='cap_input')
	X = Masking(mask_value=0, input_shape=(model_conf['max_cap_length'], model_conf['output_dim']))(cap_input)
	X = Embedding(output_dim=model_conf['dim_word'], input_dim=model_conf['worddict'],
	              input_length=model_conf['max_cap_length'])(cap_input)
	X = GRU(output_dim=model_conf['output_dim'], return_sequences=False)(X)
	X = Lambda(lambda x: tf_l2norm(x))(X)
	emb_cap = Lambda(lambda x: abs(x))(X)

	print("loading the joined model")
	merged = Merge(mode='concat')([emb_cap, emb_image])
	model = Model(input=[cap_input, image_input], output=[merged])

	print("compiling the model")
	model.compile(optimizer=model_conf['optimizer'][0], loss=contrastive_loss_keras)
	return model, image_input, emb_image, cap_input, emb_cap


def generate_dataset_embeddings(image_model):
	image_embeddings_filename = "%s-baseline-image-embeddings.pickle" % settings.DATASET
	if check_pickle_file(image_embeddings_filename):
		return load_pickle_file(image_embeddings_filename)

	image_vector_pairs = fetch_all_image_vector_pairs()
	filename_embedding_tuples = []
	tot_images = len(image_vector_pairs)
	for i in range(tot_images):
		filename, vector = image_vector_pairs[i]
		embedding = image_model.predict(numpy.array([vector]))[0]
		filename_embedding_tuples.append((filename, embedding))
		print_progress(i + 1, tot_images, prefix="Generating 1024 dim image embeddings")

	save_pickle_file(filename_embedding_tuples, image_embeddings_filename)
	return filename_embedding_tuples


def eval_model(model, image_input, emb_image, cap_input, emb_cap, train_iter):
	print('evaluating model...')

	weights = model.get_weights()
	emb_w = weights[0]
	im_w = weights[10]
	im_b = weights[11]
	gru_weights = weights[1:10]

	image_model = Model(input=image_input, output=emb_image)
	image_model.set_weights([im_w, im_b])
	image_model.compile(optimizer='adam', loss=contrastive_loss_keras)

	caption_model = Model(input=cap_input, output=emb_cap)
	caption_model.set_weights([emb_w] + gru_weights)
	caption_model.compile(optimizer='adam', loss=contrastive_loss_keras)

	test_data = train_iter.all()
	test_caps = test_data[0]
	test_ims = test_data[1]
	test_filenames = test_data[2]

	print("Predicting test captions...")
	pred_cap = caption_model.predict(numpy.array(test_caps))
	filename_image_embedding_tuples = generate_dataset_embeddings(image_model)
	print (evaluate(test_filenames, pred_cap, filename_image_embedding_tuples))


def evaluate(test_filenames, pred_cap, filename_image_embedding_tuples):
	r1 = []
	r5 = []
	r10 = []
	r20 = []
	r100 = []
	r1000 = []

	# te_ca_caption_vectors = fetch_test_captions_vectors()
	predicted_image_vectors = pred_cap

	tr_im_filename_image_vector_tuples = filename_image_embedding_tuples
	tr_im_filenames = [x[0] for x in tr_im_filename_image_vector_tuples]
	tr_im_image_vectors = [x[1] for x in tr_im_filename_image_vector_tuples]

	similarity_matrix = cosine_similarity(predicted_image_vectors, tr_im_image_vectors)

	print("Creating cosine similarity matrix...")
	predicted_images_size = len(predicted_image_vectors)
	total_image_size = len(tr_im_image_vectors)
	for predicted_image_index in range(predicted_images_size):
		similarities = []
		for i in range(total_image_size):
			tr_filename = tr_im_filenames[i]
			similarities.append((tr_filename, similarity_matrix[predicted_image_index][i]))
		similarities.sort(key=lambda s: s[1], reverse=True)

		test_filename = test_filenames[predicted_image_index]

		for top_image_index in range(1000):
			comparison_filename = similarities[top_image_index][0]
			if test_filename == comparison_filename:
				if top_image_index < 1000:
					r1000.append(1.0)
				if top_image_index < 100:
					r100.append(1.0)
				if top_image_index < 20:
					r20.append(1.0)
				if top_image_index < 10:
					r10.append(1.0)
				if top_image_index < 5:
					r5.append(1.0)
				if top_image_index == 0:
					r1.append(1.0)
				break

		print_progress(predicted_image_index + 1, predicted_images_size, prefix="Calculating recall")

	r1_avg = sum(r1) / predicted_images_size
	r5_avg = sum(r5) / predicted_images_size
	r10_avg = sum(r10) / predicted_images_size
	r20_avg = sum(r20) / predicted_images_size
	r100_avg = sum(r100) / predicted_images_size
	r1000_avg = sum(r1000) / predicted_images_size
	return r1_avg, r5_avg, r10_avg, r20_avg, r100_avg, r1000_avg


model_config = {}


def trainer(config):
	global model_config
	model_config = config

	if "eval" in sys.argv:
		train(model_config, eval_mode=True)
	else:
		train(model_config)
