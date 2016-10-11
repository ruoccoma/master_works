from keras.layers import np


def split_list(data, split_ratio=0.8):
    return np.asarray(data[:int((len(data) * split_ratio))]), np.asarray(data[int((len(data) * split_ratio)):])
