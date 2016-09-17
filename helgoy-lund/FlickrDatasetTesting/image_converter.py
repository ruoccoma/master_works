import numpy
import pickle
from PIL import Image
from os import listdir
from os.path import isfile, join


def fetch_all_imagepaths():
	dirpath = "Flicker8k_Dataset"
	return [f for f in listdir(dirpath) if isfile(join(dirpath, f))]


def image2RGBpixelarray(filepath):
	im = Image.open(filepath)
	image_map = list(im.getdata())
	image_map = numpy.array(image_map)
	RGB = [[i[0] for i in image_map], [i[1] for i in image_map], [i[2] for i in image_map]]
	return numpy.asarray(RGB)


def generate_imgRBG_img_dict():
	img_dict = {}
	imgs = fetch_all_imagepaths()
	num_images = str(len(imgs))
	counter = 1
	for img in imgs:
		img_dict[img] = image2RGBpixelarray("Flicker8k_Dataset/" + img)
		print("Images converted: " + str(counter) + "/" + num_images)
		counter += 1

	return img_dict


def save_img_dict(img_dict):
	pickle.dump(img_dict, open("image_RGB_img_dict-" + str(len(img_dict)), "wb"))


def load_img_dict(file):
	return pickle.load(open(file, "rb"))


img_dict = generate_imgRBG_img_dict()

save_img_dict(img_dict)
