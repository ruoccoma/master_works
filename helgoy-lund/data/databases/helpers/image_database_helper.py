import sys

import numpy as np

import os
import sys

file_par_dir = os.path.join(__file__, os.pardir)
file_par_par_dir = os.path.join(file_par_dir, os.pardir)
file_par_par_par_dir = os.path.join(file_par_par_dir, os.pardir)
ROOT_DIR = os.path.dirname((os.path.abspath(file_par_par_par_dir))) + "/"
sys.path.append(ROOT_DIR)
import settings

import sqlite_wrapper as wrapper


def store_image_vector_to_db(image_name, vector):
	wrapper.db_insert_image_vector(image_name, vector)


def fetch_all_image_names():
	return [x[0] for x in wrapper.db_keys_images()]


def fetch_image_vector(image_name):
	return wrapper.db_get_image_vector(image_name)[0]


def fetch_all_image_vector_pairs():
	return wrapper.db_all_filename_img_vec_pairs()


def fetch_filename_from_image_vector(image_vector):
	return wrapper.db_get_filename_from_image_vector(image_vector)


def update_image_vectors(filename_image_vector_tuples):
	return wrapper.db_insert_image_vector_list(filename_image_vector_tuples)


def normalize_abs_image_vectors():
	print("Loading database...")
	tr_im_image_vector_tuples = wrapper.db_all_filename_img_vec_pairs()
	print("Loaded database.")
	for i in range(len(tr_im_image_vector_tuples)):
		filename, vector = tr_im_image_vector_tuples[i]
		tr_im_image_vector_tuples[i] = np.asarray(l2norm(vector)), filename
		print_progress(i, len(tr_im_image_vector_tuples), prefix="Normilizing all images")

	update_image_vectors(tr_im_image_vector_tuples)


def fiddle():
	print("Loading database...")
	tr_im_image_vector_tuples = wrapper.db_all_filename_img_vec_pairs()
	print("Loaded database.")
	print("First element:,", tr_im_image_vector_tuples[0][0], tr_im_image_vector_tuples[0][1])

if __name__ == "__main__":
	if "fiddle" in sys.argv:
		fiddle()
	elif "norm" in sys.argv:
		normalize_abs_image_vectors()




"""

Transitive dependence

"""


def l2norm(array):
	norm = math.sqrt(np.sum(([math.pow(x, 2) for x in array])))
	array = [x / norm for x in array]
	return array


def print_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=30):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		barLength   - Optional  : character length of bar (Int)
	"""
	formatStr = "{0:." + str(decimals) + "f}"
	percents = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s%s%s  %s' % (prefix, bar, percents, '%', iteration, '/', total, suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()