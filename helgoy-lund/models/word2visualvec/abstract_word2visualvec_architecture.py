from abc import abstractmethod, ABCMeta

import keras.backend.tensorflow_backend as ker
from keras import callbacks

from custom_callback import WriteToFileCallback

import settings


class AbstractWord2VisualVecArchitecture:
	__metaclass__ = ABCMeta

	remote = callbacks.RemoteMonitor(root='http://127.0.0.1:9000')
	custom_callback = WriteToFileCallback(settings.RESULT_TEXTFILE_PATH)
	early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)

	def __init__(self,
	             epochs=50,
	             batch_size=128,
	             validation_split=0.2,
	             optimizer="adam",
	             loss="categorical_crossentropy"):
		# hyperparams
		self.epochs = epochs
		self.batch_size = batch_size
		self.validation_split = validation_split
		self.optimizer = optimizer
		self.loss = loss
		self.callbacks = [self.custom_callback, self.early_stopping]
		self.model = None
		self.prediction_model = None

		config1 = ker.tf.ConfigProto()
		config1.gpu_options.allow_growth = True
		ker.set_session(ker.tf.Session(config=config1))

	@abstractmethod
	def train(self):
		pass

	def get_architecture_name(self):
		return type(self).__name__

	def get_name(self):
		return self.get_architecture_name() + "-" + self.get_parameter_string() + "-" + settings.DB_SUFFIX

	@abstractmethod
	def generate_prediction_model(self):
		pass

	@abstractmethod
	def generate_model(self):
		pass

	def get_parameter_string(self):
		return "%s-%s-%s-%s" % (self.epochs, self.batch_size, self.optimizer, self.loss.__name__)
