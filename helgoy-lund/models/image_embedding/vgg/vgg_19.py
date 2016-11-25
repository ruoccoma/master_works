from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from os import listdir
from os.path import isfile, join
from list_helpers import l2norm, print_progress
from image_database_helper import store_image_vector_to_db
import settings

LAYER = 'fc1'

def get_model():
	base_model = VGG19(weights='imagenet', include_top=True)
	return Model(input=base_model.input, output=base_model.get_layer(LAYER).output)


def predict(model, img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return model.predict(x)


def fetch_all_imagepaths():
	dirpath = settings.IMAGE_DIR
	return [f for f in listdir(dirpath) if isfile(join(dirpath, f))]


def run_vgg():
	vgg = get_model()
	image_paths = fetch_all_imagepaths()
	num_images = len(image_paths)
	for i in range(num_images):
		image_path = settings.IMAGE_DIR + "/" + image_paths[i]
		image_vector = predict(vgg, image_path)[0]
		norm_image_vector = np.asarray(l2norm(image_vector))
		store_image_vector_to_db(image_paths[i], norm_image_vector)
		print_progress(i, num_images)

if __name__ == "__main__":
	run_vgg()
