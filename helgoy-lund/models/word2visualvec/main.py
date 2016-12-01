import datetime
import os
import sys
import time

from text2image_evaluation import predict_text2image_model, test_text2image_model, evaluate_text2image_model

ROOT_DIR = os.path.dirname((os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir)))) + "/"
sys.path.append(ROOT_DIR)
import settings

import io_helper
io_helper.create_missing_folders()

from reverse_euclidian_distance_architecture import ReversedEuclidanDistanceArchitecture
from euclidian_distance_architecture import FiveLayerEuclidianDistance, ThreeLayerEuclidianDistance, \
	SixLayerBatchNormEuclidianDistance, SixLayerEuclidianDistance

ARCHITECTURES = [ReversedEuclidanDistanceArchitecture(),
				 ThreeLayerEuclidianDistance(),
				 FiveLayerEuclidianDistance(),
				 SixLayerEuclidianDistance(),
				 SixLayerBatchNormEuclidianDistance()]


NEG_TAG = "neg" if settings.CREATE_NEGATIVE_EXAMPLES else "pos"


def main():
	current_time = datetime.datetime.time(datetime.datetime.now())
	print("Current time: %s" % current_time)
	for ARCHITECTURE in ARCHITECTURES:
		print(ARCHITECTURE.get_name())
		train(ARCHITECTURE)
		if "eval" in sys.argv:
			evaluate(ARCHITECTURE)
		if "caption_query" in sys.argv:
			caption_query(ARCHITECTURE)
		if "sample_image_query" in sys.argv:
			sample_image_query(ARCHITECTURE)


def train(architecture):
	result_file = open(settings.RESULT_TEXTFILE_PATH, 'a')
	result_file.write(architecture.get_name() + "\n")
	result_file.close()
	if is_saved(architecture):
		print("Architecture already trained")
	else:
		print("Training architecture...")
		architecture.train()
		print(architecture.model.summary())
		save_model_to_file(architecture.model, architecture)


def evaluate(architecture):
	result_file = open(settings.RESULT_TEXTFILE_PATH, 'a')
	result_file.write(architecture.get_name() + "\n")
	result_file.close()

	load_model(architecture)
	architecture.generate_prediction_model()

	print("Starting evaluation of model...")
	time_start = time.time()
	r1_avg, r5_avg, r10_avg, r20_avg, r100_avg, r1000_avg = architecture.evaluate()
	time_end = time.time()

	# test_model(ARCHITECTURE.prediction_model)

	result_header = "RESULTS: (Evaluating time: %s)\n" % ((time_end - time_start) / 60.0)
	recall_results = "r1:%s,r5:%s,r10:%s,r20:%s,r100:%s,r1000:%s\n" % \
	(r1_avg, r5_avg, r10_avg, r20_avg, r100_avg, r1000_avg)

	result_file = open(settings.RESULT_TEXTFILE_PATH, 'a')
	result_file.write(result_header)
	result_file.write(recall_results)
	result_file.close()

	print(result_header)
	print(recall_results)
	print("\n")


def caption_query(architecture):
	if is_saved(architecture):
		load_model(architecture)
		architecture.generate_prediction_model()
		architecture.predict()
	else:
		print("Architecture not trained")
		print(architecture.get_name())


def sample_image_query(architecture):
	if is_saved(architecture):
		load_model(architecture)
		architecture.generate_prediction_model()
		architecture.test()
	else:
		print("Architecture not trained")
		print(architecture.get_name())


def save_model_to_file(model, architecture):
	name = architecture.get_name()
	model.save_weights("stored_models/" + name + ".h5")
	print("Saved model \"%s\" to disk" % name)


def is_saved(arc):
	if os.path.isfile("stored_models/" + arc.get_name() + ".h5"):
		return True
	return False


def load_model(arc):
	arc.generate_model()
	name = arc.get_name()
	arc.model.load_weights("stored_models/" + name + ".h5")
	arc.model.compile(optimizer=arc.optimizer, loss=arc.loss)


main()
