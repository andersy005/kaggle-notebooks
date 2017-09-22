import numpy as np
from keras.utils.np_utils import to_categorical
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('seaborn')



def data_info(x_train, y_train, x_test, y_test):
    """Print Dataset information"""
    print("....Shape information......\n")
    print("Training set         (shape) : {}".format(x_train.shape))
    print("Training set labels  (shape) : {}".format(y_train.shape))
    print("Test set             (shape) : {}".format(x_test.shape))
    print("Test set labels      (shape) : {}".format(y_test.shape))


def one_hot_encode(labels):
    """ Function that converts labels to categorical values"""
    return to_categorical(labels)




def training(model, x_train, y_train, epochs=10, batch_size=64):
    print("Training......\n\n")
    start = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.1)
    stop = time.time()
    total = stop - start

    print("\n\nTraining took {} secs".format(round(total, 3)))
    return history


def plot_history(history):

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc_values = history.history['acc']
    val_acc_values = history.history['val_acc']

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
