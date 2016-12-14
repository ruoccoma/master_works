'''
Author: Igor Lapshun
'''
import os
import sys

import datetime

ROOT_DIR = os.path.dirname((os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir)))) + "/"
sys.path.append(ROOT_DIR)
import settings


# This is the configurations/ meta parameters for this model
# (currently set to optimal upon cross valiation).
config = {
	'model_cnn': 'vgg19_weights.h5',
	# 'data': 'data/coco',
	'save_dir': 'anypath',
	'dim_cnn': settings.IMAGE_EMBEDDING_DIMENSIONS,
	'optimizer': 'adam',
	'batch_size': 128,
	'epoch': 300,
	'output_dim': 1024,
	'dim_word': 300,
	'lrate': 0.05,
	'max_cap_length': 82,
	'cnn': '10crop',
	'margin': 0.05
}

if __name__ == '__main__':
	import trainer

	current_time = datetime.datetime.time(datetime.datetime.now())
	print ("Current time: %s" % current_time)

	if settings.DATASET == "Flickr8k":
		config["max_cap_length"] = 50
	elif settings.DATASET == "Flickr30k":
		config["max_cap_length"] = 82

	trainer.trainer(config)
