import theano
import theano.tensor as tensor
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
