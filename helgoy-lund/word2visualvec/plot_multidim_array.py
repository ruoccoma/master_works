#!/usr/bin/env python
# -*- coding: utf-8 -*-
# That's an impressive list of imports.

# We import sklearn.

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.

# Random state.
RS = 20150101

# We'll use matplotlib for graphics.

# We import seaborn to make nice plots.
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

### PLOTTING: REMOVE ABOVE

import numpy


def plot():
	# We first reorder the data points according to the handwritten numbers.
	X = numpy.vstack([x[1] for x in fetch_image_vector_pairs()])
	y = numpy.hstack([x[0] for x in fetch_image_vector_pairs()])

	digits_proj = TSNE(random_state=RS).fit_transform(X)

	scatter(digits_proj, y)
	plt.savefig('images/digits_tsne-generated.png', dpi=120)


def scatter(x, colors):
	# We choose a color palette with seaborn.
	palette = numpy.array(sns.color_palette("hls", 10))

	# We create a scatter plot.
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40)
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')

	return f, ax, sc


plot()
