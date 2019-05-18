import tensorflow as tf
import keras
from tensorflow.keras.datasets.cifar10 import load_data as ld
from tensorflow.keras.backend import get_session
from tensorflow import RunMetadata
from tensorflow.profiler import profile
from tensorflow.profiler import ProfileOptionBuilder
from matplotlib.pyplot import plot, title, ylabel, xlabel, legend, show, figure
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from os.path import isfile
import pickle
from tensorflow.keras.utils import to_categorical

def load_data(k):
    num_classes = 10

    # if isfile('data/data.pickle'):
    #     with open('data/data.pickle', 'r') as f:
    #         data = pickle.load(f)
    #         return data[0], data[1]
    # else:

    (x_train, y_train), (x_test, y_test) = ld()


    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255
    # x_train.shape

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

        # with open('data/data.pickle', 'wb') as f:
        #     pickle.dump(((x_train, y_train), (x_test, y_test)), f)

    n = np.shape(x_train)[0]
    x_train, x_val = [x_train[0:int(k*n), :, :, :], x_train[int(k*n):, :, :, :]]
    y_train, y_val = [y_train[0:int(k*n), :], y_train[int(k*n):, :]]

    # standardize instead of divide by 255, also flip the image horizontally
    trainDatagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True)
    testDatagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True) # Not flipping them in validation and test
    trainDatagen.fit(x_train)
    testDatagen.fit(x_train)

    ValIter = testDatagen.flow(x_val, y_val)
    TestIter = testDatagen.flow(x_test, y_test)

    return (x_train, y_train), ValIter, TestIter, testDatagen

def plot_accuracy(history):
    # Plot training & validation accuracy values
    f = figure()
    plot(history['acc'])
    plot(history['val_acc'])
    title('Model accuracy')
    ylabel('Accuracy')
    xlabel('Epoch')
    legend(['Train', 'Validation'], loc='upper right')
    return f

def plot_loss(history):
    # Plot training & validation loss values
    f = figure()
    plot(history['loss'])
    plot(history['val_loss'])
    title('Model loss')
    ylabel('Loss')
    xlabel('Epoch')
    legend(['Train', 'Validation'], loc='upper right')
    return f

# https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
def get_flops(model):
    """ Provide the number of flops for a tf.keras model """
    run_meta = RunMetadata()
    opts = ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = profile(graph=get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.
