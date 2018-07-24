import numpy as np
from pkl import pickle_load
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda
from keras.layers import MaxPooling2D, Dropout
from keras.utils import print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K


def keras_model(image_x, image_y):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(image_x, image_y, 1)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.0001), loss="mse")
    filepath = "model_adam_mse.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    return model, callbacks_list


def loadFromPickle():
    preprocessed = np.array(pickle_load("preprocessed"))
    labels = np.array(pickle_load("labels"))
    return preprocessed, labels


def augmentData(preprocessed, labels):
    preprocessed = np.append(preprocessed, preprocessed[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return preprocessed, labels

if __name__ == "__main__":
    preprocessed, labels = loadFromPickle()
    preprocessed, labels = augmentData(preprocessed, labels)
    preprocessed, labels = shuffle(preprocessed, labels)
    train_x, test_x, train_y, test_y = train_test_split(preprocessed, labels, random_state=0, test_size=.1)
    train_x = train_x.reshape(train_x.shape[0], 100, 100, 1)
    test_x = test_x.reshape(test_x.shape[0], 100, 100, 1)
    model, callbacks_list = keras_model(100, 100)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5, batch_size=64, callbacks=callbacks_list)
    
    print("Summary:")
    print_summary(model)

    model.save('model_adam_mse.h5')

    K.clear_session()

