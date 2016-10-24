from data_helper import generate_data
from list_helpers import split_list
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense

# hyperparams
epochs = 30
batch_size = 32
validation_split = 0.2
optimizer = "adadelta"
loss = "categorical_crossentropy"


def get_optimizer():
    return optimizer


def get_loss():
    return loss


def train():
    data_x, data_y = generate_data()

    training_test_ratio = 1

    training_data_x, test_data_x = split_list(data_x, training_test_ratio)
    training_data_y, test_data_y = split_list(data_y, training_test_ratio)

    # this returns a tensor
    inputs = Input(shape=(128,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(512, activation='relu')(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2048, activation='softmax')(x)

    # this creates a model that includes
    # the Input layer and three Dense layers
    model = Model(input=inputs, output=predictions)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    model.fit(training_data_x, training_data_y, nb_epoch=epochs, validation_split=validation_split, shuffle=True,
              batch_size=batch_size)  # starts training

    return model
