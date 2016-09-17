import numpy
from PIL import Image


def image2RBKpixelarray(filepath):
	im = Image.open(filepath)
	image_map = list(im.getdata())
	image_map = numpy.array(image_map)
	RGB = [[i[0] for i in image_map], [i[1] for i in image_map], [i[2] for i in image_map]]
	return numpy.asarray(RGB)