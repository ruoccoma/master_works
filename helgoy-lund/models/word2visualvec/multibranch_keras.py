from embeddings_helper import structure_and_store_embeddings
from keras.models import Sequential
from list_helpers import split_list, tf_l2norm
from keras.engine import Input, Model
from keras.layers import Dense, Lambda, Merge
import numpy as np

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
    caption_vectors, image_vectors = structure_and_store_embeddings(10)
    similarities = np.ones(len(caption_vectors))

    caption_inputs = Input(shape=(300,))
    image_inputs = Input(shape=(2048,))

    caption_model = Dense(400, activation='relu')(caption_inputs)
    caption_model = Dense(800, activation='relu')(caption_model)
    caption_model = Dense(1024, activation='relu')(caption_model)
    caption_model = Dense(2048, activation='relu')(caption_model)

    caption_model = Lambda(lambda x: tf_l2norm(x))(caption_model)
    caption_model = Lambda(lambda x: abs(x), name="Caption Abs")(caption_model)

    image_model = Lambda(lambda x: abs(x), name="Image Abs")(image_inputs)



    merge = Merge(mode="cos", dot_axes=1)([caption_model, image_model])
    merged_model = Model(input=[caption_inputs, image_inputs], output=[merge])

    merged_model.summary()

    merged_model.compile(optimizer=optimizer, loss=loss)
    merged_model.fit([caption_vectors, image_vectors], similarities)


def train_sequential():
    caption_vectors, image_vectors= structure_and_store_embeddings()
    similarities = [1.0 for x in range(len(caption_vectors))]

    caption_inputs = Input(shape=(300,))
    image_inputs = Input(shape=(2048,))

    caption_model = Sequential()
    caption_model.add(Dense(400, activation='relu', input_dim=caption_inputs))
    caption_model.add(Dense(800, activation='relu'))
    caption_model.add(Dense(1024, activation='relu'))
    caption_model.add(Lambda(lambda x: tf_l2norm(x)))
    caption_model.add(Lambda(lambda x: abs(x)))

    image_model = Sequential()
    image_model.add(Lambda(lambda x: abs(x)))

    merge = Merge([caption_model, image_model], mode="cos")
    merged_model = Model(input=[caption_inputs, image_inputs], output=[merge])

    merged_model.compile(optimizer=optimizer, loss=loss)
    merged_model.fit([caption_vectors, image_vectors], similarities)