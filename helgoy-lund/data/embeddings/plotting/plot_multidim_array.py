#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from sklearn.manifold import TSNE

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from image_database_helper import fetch_image_vector_pairs
from caption_database_helper import fetch_all_caption_vectors

# Random state.
RS = 20150101

def plot_image_vectors():
	# We first reorder the datasets points according to the handwritten numbers.
	X = numpy.vstack([x[1] for x in fetch_image_vector_pairs()])
	y = numpy.hstack([x[0] for x in fetch_image_vector_pairs()])

	digits_proj = TSNE(random_state=RS).fit_transform(X)

	scatter(digits_proj, y)
	plt.savefig('images/digits_tsne-generated.png', dpi=120)

def plot_word_averagings():
	# We first reorder the datasets points according to the handwritten numbers.
	X = numpy.vstack([x[0] for x in fetch_all_caption_vectors()[:1000]])
	y = numpy.hstack([x[0] for x in fetch_image_vector_pairs()[:10]])

	digits_proj = TSNE(random_state=RS).fit_transform(X)

	scatter(digits_proj, y)
	plt.savefig('images/word-averaging_tsne-generated.png', dpi=120)

def scatter(x, colors):
	palette = []
	for color in range(0, len(x), 5):
		palette.append([color for i in range(5)])
	# We create a scatter plot.
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette)
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')

	return f, ax, sc

plot_word_averagings()