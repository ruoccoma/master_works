import keras


class WriteToFileCallback(keras.callbacks.Callback):
	def __init__(self, filename="training-epochs-results-DEFAULT.txt"):
		super().__init__()
		self.filename = filename

	def on_epoch_end(self, epoch, logs={}):
		file = open(self.filename, 'a')
		file.write("%s," % epoch)
		for k, v in logs.items():
			file.write("%s,%s," % (k, v))
		file.write("\n")
		file.close()
