import theano
import theano.tensor as tensor
import tensorflow as tf
from theano.tensor.extra_ops import fill_diagonal


def theano_contrastive_loss(labels, predict):
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

def tensorflow_contrastive_loss_converted(labels, predict):
	"""For a minibatch of sentence and image embeddings, compute the pairwise contrastive loss"""
	margin = 0.05
	res = tf.split(0, 2, predict)
	s = res[0]
	im = res[1]
	im2 = im.dimshuffle(('x', 0, 1))
	s2 = s.dimshuffle((0, 'x', 1))
	errors = tf.pow(tf.maximum(0, im2 - s2), 2).sum(axis=2)
	diagonal = errors.diagonal()
	# compare every diagonal score to scores in its column (all contrastive images for each sentence)
	cost_s = tf.maximum(0, margin - errors + diagonal)
	# all contrastive sentences for each image
	cost_im = tf.maximum(0, margin - errors + diagonal.reshape((-1, 1)))
	cost_tot = cost_s + cost_im
	cost_tot = fill_diagonal(cost_tot, 0)
	return cost_tot.sum()


def tensorflow_contrastive_loss(lables, predict):
	margin = 0.2
	label = 1


	res = tf.split(0, 2, predict)
	s = res[0]
	im = res[1]

	d = tf.reduce_sum(tf.square(s - im), 0)
	d_sqrt = tf.sqrt(d)

	loss = label * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - label) * d

	loss = 0.5 * tf.reduce_mean(loss)

	return loss

if __name__ == "__main__":
	test_list = tf.constant([[1, 1, 2, 2], [3, 3, 4, 4]])


	with tf.Session() as sess:
		#res = tensorflow_contrastive_loss(None, test_list)
		for i in test_list:
			res = tf.reshape(test_list, [2, 2])
			print(sess.run(res))
