#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error

from data_helper import generate_data
from caption_database_helper import db_get_filename_caption_tuple_from_vector
from image_database_helper import fetch_image_vector_pairs
from helper import split_list

# Import models
import feedforward_keras
import autoencoder_keras

# Settings
SAVE_MODEL = True
LOAD_MODEL = True
MODELS = [feedforward_keras, autoencoder_keras]
MODEL = MODELS[1]
MODEL_SUFFIX = ""


def word2visualvec_main():
    if LOAD_MODEL:
        model = load_model(MODEL.__name__)
        #model = load_model("feedforward-adagrad-Epochs:30-Batch:128-Optimizer:adadelta")
    else:
        model = MODEL.train()

        if SAVE_MODEL:
            save_model_to_file(model, MODEL.__name__)
            # save_model_to_file(autoencoder, "autoencoder")
            # save_model_to_file(decoder, "decoder")

    test_model(model)


def save_model_to_file(model, name):
    name += MODEL_SUFFIX
    # serialize model to JSON
    model_json = model.to_json()
    with open("stored_models/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("stored_models/" + name + ".h5")
    print("Saved model \"%s\" to disk" % name)


def load_model(name):
    name += MODEL_SUFFIX
    # load json and create model
    json_file = open("stored_models/" + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("stored_models/" + name + ".h5")
    print("Loaded model \"%s\" from disk" % name)

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=MODEL.get_optimizer(), loss=MODEL.get_loss())
    return loaded_model


def compare_vectors(v1, v2):
    return mean_squared_error(v1, v2)


def test_model(model):
    data_x, data_y = generate_data(10)
    training_test_ratio = 0.8

    train_x, test_x = split_list(data_x, training_test_ratio)
    train_y, test_y = split_list(data_y, training_test_ratio)

    img_num = 0
    testing_data_x = test_x[5:10]

    correct_image_filename, correct_image_caption = db_get_filename_caption_tuple_from_vector(testing_data_x[img_num])
    print("Correct caption:", correct_image_caption)
    print("Correct filename:", correct_image_filename)

    predicted_image_vector = model.predict(testing_data_x)[img_num]

    image_vector_pairs = fetch_image_vector_pairs()

    best_image_vector = image_vector_pairs[img_num][1]
    best_image_vector_mse = compare_vectors(predicted_image_vector, image_vector_pairs[img_num][1])
    best_image_vector_name = image_vector_pairs[img_num][0]
    print("Finding closest image vector...")
    for name, image_vector in image_vector_pairs:
        temp_mse = compare_vectors(image_vector, best_image_vector)
        if temp_mse < best_image_vector_mse:
            best_image_vector = image_vector
            best_image_vector_mse = temp_mse
            best_image_vector_name = name
        elif temp_mse == best_image_vector_mse:
            print("Identical")
    print("Best image vector name:", str(best_image_vector_name))


word2visualvec_main()
