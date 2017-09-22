from keras import layers
from keras import models


def base_mlp(input_dim, nb_classes, loss, optimizer='adam',
             dropout=0.15, metrics=['accuracy']):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_dim))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(nb_classes))
    model.add(layers.Activation('softmax'))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    print(model.summary())
    return model




def base_cnn(input_shape, nb_classes, loss='categorical_crossentropy', optimizer='adam', dropout=0.15, metrics=['accuracy']):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    print(model.summary())
    return model
