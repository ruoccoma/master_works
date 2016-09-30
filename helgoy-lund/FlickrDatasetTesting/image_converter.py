import numpy
# numpy.set_printoptions(threshold=numpy.inf)

from PIL import Image
import os
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
import random
import pickle
import time
import datetime

from sqliteDatabase import db_insert_image_vector, db_get_image_vector, db_keys_images


def fetch_all_imagepaths():
	dirpath = "Flicker8k_Dataset"
	return [f for f in listdir(dirpath) if isfile(join(dirpath, f))]


def image2RGBpixelarray2(filename):
	im = Image.open("Flicker8k_Dataset/" + filename)
	image_map = list(im.getdata())
	image_map = numpy.array(image_map)
	RGB = [[i[0] for i in image_map], [i[1] for i in image_map], [i[2] for i in image_map]]
	return filename, numpy.asarray(RGB)


def image2RGBpixelarray(filename):
	img = Image.open("Flicker8k_Dataset/" + filename)
	rgb = numpy.array(img)
	red = numpy.zeros(shape=(rgb.shape[0], rgb.shape[1]), dtype=numpy.int)
	green = numpy.zeros(shape=(rgb.shape[0], rgb.shape[1]), dtype=numpy.int)
	blue = numpy.zeros(shape=(rgb.shape[0], rgb.shape[1]), dtype=numpy.int)
	for i in range(len(rgb)):
		for j in range(len(rgb[i])):
			red[i][j] = rgb[i][j][0]
			green[i][j] = rgb[i][j][1]
			blue[i][j] = rgb[i][j][2]

	return filename, numpy.asarray([red, green, blue])


def generate_imgRBG_img_dict():
	img_dict = {}
	imgs = fetch_all_imagepaths()
	num_images = str(len(imgs))
	counter = 1
	for img in imgs:
		img_dict[img] = image2RGBpixelarray(img)
		print("Images converted: " + str(counter) + "/" + num_images)
		counter += 1

	return img_dict


def save_img_dict(img_dict_to_save):
	now = datetime.datetime.now()
	filename = str(now.hour) + str(now.minute) + "-image_RGB_img_dict-" + str(len(img_dict_to_save)) + ".txt"
	with open(filename, 'wb') as pickle_file:
		pickle.dump(img_dict_to_save, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_img_dict(file):
	with open(file, 'rb') as pickle_load:
		return pickle.load(pickle_load)


def convert_result_list_to_dict(result):
	result_length = str(len(result))
	img_dict = {}
	counter = 1
	for name_array_tuple in result:
		img_dict[name_array_tuple[0]] = name_array_tuple[1]
		if counter % 250 == 0:
			print("Converting result to dictionary\t" + str(counter) + "/" + result_length)
		counter += 1
	return img_dict


def write_result_to_file(results):
	now = datetime.datetime.now()
	filename = str(now.hour) + str(now.minute) + "-image_RGB_img_dict-" + str(len(results)) + ".txt"
	text_file = open(filename, "w")
	for result_tuple in results:
		text_file.write(result_tuple[0])
		text_file.write(",")
		numpy.savetxt(filename, result_tuple[1])
		text_file.write("\n")

	text_file.close()


def write_result_to_database(results):
	for result_tuple in results:
		db_insert_image_vector(result_tuple[0], result_tuple[1])


def generate_and_store_images():
	if __name__ == '__main__':
		pool = Pool(os.cpu_count())  # process per core
		result = pool.map_async(image2RGBpixelarray, fetch_all_imagepaths())
		pool.close()
		old_remaining = -1
		while True:
			if result.ready():
				break

			remaining = result._number_left

			if old_remaining == remaining:
				continue

			old_remaining = remaining
			print("Waiting for", remaining, "tasks to complete...")
			time.sleep(1)

		result = result.get()
		# img_dict = convert_result_list_to_dict(result)
		# write_result_to_file(result)
		write_result_to_database(result)

	# save_img_dict(img_dict)


def create_image():
	"""
	Only for testing purposes.

	:return:
	void. Saves img to current dir.
	"""
	now = datetime.datetime.now()
	# old = time.time()
	rgb_array = db_get_image_vector("997722733_0cb5439472.jpg")
	# all_images = db_keys()
	# new = time.time()
	# print("Time to get from DB: " + str(new - old))
	# print("All images length: " + str(len(all_images)))
	red = rgb_array[0][0]
	green = rgb_array[0][1]
	blue = rgb_array[0][2]

	rgb_tuples = numpy.zeros(shape=red.shape, dtype='i,i,i')
	# rgb_tuples = [(0,0,0) for i in range(red.shape[0])]
	for i in range(len(red)):
		for j in range(len(red[0])):
			rgb_tuples[i][j] = (int(red[i][j]), int(green[i][j]), int(blue[i][j]))

	rgb_tuples_list = []
	for i in range(len(rgb_tuples)):
		for j in range(len(rgb_tuples[0])):
			rgb_tuples_list.append(tuple(rgb_tuples[i, j]))
	im2 = Image.new("RGB", (red.shape[1], red.shape[0] ))
	im2.putdata(rgb_tuples_list)
	im2.save(str(now.hour) + str(now.minute) + ".jpg")

create_image()
#
# generate_and_store_images()

# img_dict = load_img_dict("async-image_RGB_img_dict-8090")
# img_dict = load_img_dict("test-image_RGB_img_dict-10")
# img_dict = load_img_dict("1210 - image_RGB_img_dict-1000")
# print(img_dict)
