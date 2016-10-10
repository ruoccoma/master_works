import math

import numpy as np
from data_helper import generate_data
from keras.layers import Input, Dense
from keras.models import Model, model_from_json
from caption_database_helper import db_get_filename_caption_tuple_from_vector
from image_database_helper import fetch_image_vector_pairs
from sklearn.metrics import mean_squared_error

SAVE_MODEL = False
MODEL_SUFFIX = ""
LOAD_MODEL = False


def word2visualvec_main():
    if LOAD_MODEL:
        encoder, decoder = load_model("encoder"), load_model("decoder")
    else:
        train()


def save_model_to_file(model, name):
    name += "-" + MODEL_SUFFIX
    # serialize model to JSON
    model_json = model.to_json()
    with open("stored_models/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("stored_models/" + name + ".h5")
    print("Saved model \"%s\" to disk" % name)


def load_model(name, optimizer='adadelta', loss='binary_crossentropy'):
    name += "-" + MODEL_SUFFIX
    # load json and create model
    json_file = open("stored_models/" + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("stored_models/" + name + ".h5")
    print("Loaded model \"%s\"from disk" % name)

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=optimizer, loss=loss)
    return loaded_model


def compare_vectors(v1, v2):
    return mean_squared_error(v1, v2)


def train():
    data_x, data_y = generate_data()

    training_test_ratio = 0.8

    training_data_x = np.asarray(data_x[:int((len(data_x) * training_test_ratio))])
    training_data_y = np.asarray(data_y[:int((len(data_y) * training_test_ratio))])

    test_data_x = np.asarray(data_x[:int(math.ceil(len(data_x) * (1 - training_test_ratio)))])
    test_data_y = np.asarray(data_y[:int(math.ceil(len(data_y) * (1 - training_test_ratio)))])

    x_dim = 128
    y_dim = 2048

    # this is the size of our encoded representations
    encoding_dim = y_dim  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(x_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(y_dim, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input=input_img, output=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(training_data_x, training_data_y,
                    nb_epoch=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(test_data_x, test_data_y))

    if (SAVE_MODEL):
        save_model_to_file(autoencoder, "autoencoder")
        save_model_to_file(encoder, "encoder")
        save_model_to_file(decoder, "decoder")

    return encoder, decoder


def loadModel(training_data_x, loaded_encoder):
    img_num = 0
    testing_data_x = training_data_x[:200]
    filename, caption = db_get_filename_caption_tuple_from_vector(testing_data_x[img_num])
    predicted_image_vector = loaded_encoder.predict(testing_data_x)[img_num]
    image_vector_paris = fetch_image_vector_pairs()
    best_vector = image_vector_paris[img_num][1]
    best_vector_mse = compare_vectors(predicted_image_vector, image_vector_paris[img_num][1])
    best_vector_name = image_vector_paris[img_num][0]
    for name, image_vector in image_vector_paris:
        temp_mse = compare_vectors(image_vector, best_vector)
        if temp_mse < best_vector_mse:
            best_vector = image_vector
            best_vector_mse = temp_mse
            best_vector_name = name
    print("Best image vector name:", str(best_vector_name))
    print("Correct caption:", caption)
    print("Correct filename:", filename)


word2visualvec_main()
