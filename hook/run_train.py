import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import layers
import os
import pickle


def run():
    data_root = '/content/100_1'
    fname = '/content/100_1/0/img_11090.jpg'

    X = []
    y = []
    for i in range(10):
        target_dir_fnames = os.listdir(os.path.join(data_root, f'{i}'))
        target = list(
            map(lambda image: np.array(PIL.Image.open(os.path.join(data_root, f'{i}', image)).resize((28, 28))),
                target_dir_fnames))

        X += target
        y += [i] * len(target)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=88)

    X_train = X_train / 255.
    X_val = X_val.astype('float32') / 255.

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)

    model = build_model_1()

    EPOCHS = 50
    BATCH_SIZE = 8

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=2
    )

    ## Save pickle
    with open("model.pckl", "wb") as fw:
        pickle.dump(model, fw)




def build_model_1():
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='leaky_relu', input_shape=(28, 28, 1)),
        layers.MaxPool2D(2),
        layers.Conv2D(64, 3, activation='leaky_relu'),
        layers.MaxPool2D(2),
        layers.Flatten(),
        layers.Dropout(0.8),
        layers.Dense(258, activation='leaky_relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )

    return model
