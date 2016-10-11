from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, np

from data_helper import generate_data
from helper import split_list

# hyperparams
optimizer = "adadelta"
loss = "binary_crossentropy"


def get_optimizer():
    return optimizer


def get_loss():
    return loss


def train():
    data_x, data_y = generate_data()

    training_test_ratio = 0.8

    training_data_x, test_data_x = split_list(data_x, training_test_ratio)
    training_data_y, test_data_y = split_list(data_y, training_test_ratio)

    x_dim = 128
    y_dim = 2048

    # this is our input placeholder
    x_dim_shape = Input(shape=(x_dim,))

    # Simple network:
    # "encoded" is the encoded representation of the input
    encoded = Dense(y_dim, activation='relu')(x_dim_shape)
    encoded = Dense(2048, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(y_dim, activation='relu')(encoded)
    decoded = Dense(y_dim, activation='sigmoid')(decoded)

    """
    # Deep network: NOT WORKING
        encoded = Dense(512, activation='relu')(x_dim_shape)
        encoded = Dense(1024, activation='relu')(encoded)
        encoded = Dense(y_dim, activation='relu')(encoded)

        decoded = Dense(1025, activation='relu')(encoded)
        decoded = Dense(513, activation='relu')(decoded)
        decoded = Dense(x_dim, activation='sigmoid')(decoded)
    """

    # this model maps an input to its reconstruction
    autoencoder = Model(input=x_dim_shape, output=decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input=x_dim_shape, output=encoded)

    # create a placeholder for an encoded (2048-dimensional) input
    encoded_input = Input(shape=(y_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model

    # TODO: Fetch decoder. Maa kanskje vare av typen sequential??
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.compile(optimizer=optimizer, loss=loss)

    autoencoder.fit(training_data_x, training_data_y,
                    nb_epoch=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(test_data_x, test_data_y))

    """
    Det enkleste nettverket naar loss: 0.6281 som det beste etter 10 epoker.

    """

    return encoder
