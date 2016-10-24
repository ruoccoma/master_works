from keras.layers import np


def split_list(data, split_ratio=0.8):
	return np.asarray(data[:int((len(data) * split_ratio))]), np.asarray(data[int((len(data) * split_ratio)):])


def insert_and_remove_last(index, array, element):
	array.insert(index, element)
	del array[-1]
	return array
