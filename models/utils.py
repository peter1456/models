from tensorflow.keras.datasets.cifar10 import load_data as ld
import numpy as np
from os.path import isfile
import pickle

def load_data():

    # if isfile('data/data.pickle'):
    #     with open('data/data.pickle', 'r') as f:
    #         data = pickle.load(f)
    #         return data[0], data[1]
    # else:

    (x_train, y_train_), (x_test, y_test_) = ld()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # x_train.shape
    y_train = one_hot(y_train_, num_classes)
    y_test = one_hot(y_test_, num_classes)

        # with open('data/data.pickle', 'wb') as f:
        #     pickle.dump(((x_train, y_train), (x_test, y_test)), f)

    return (x_train, y_train), (x_test, y_test)


# one-hot encoding of labels
num_classes = 10;
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def plot(history):
    # Plot training & validation accuracy values
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


