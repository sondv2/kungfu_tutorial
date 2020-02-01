from __future__ import print_function
import argparse

import numpy
import tensorflow as tf
from keras.utils import np_utils
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback
import load_data

import argparse
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import math
import tensorflow as tf
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback
from kungfu.tensorflow.optimizers import (SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer,
                                          PairAveragingOptimizer)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
num_classes = 2
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


def build_optimizer(name, n_shards=1):
    learning_rate = 0.1

    # KungFu: adjust learning rate based on number of GPUs.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate * n_shards)

    # KUNGFU: Wrap the TensorFlow optimizer with KungFu distributed optimizers.
    if name == 'sync-sgd':
        from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
        return SynchronousSGDOptimizer(optimizer)
    elif name == 'async-sgd':
        from kungfu.tensorflow.optimizers import PairAveragingOptimizer
        return PairAveragingOptimizer(optimizer, fuse_requests=True)
    elif name == 'sma':
        from kungfu.tensorflow.optimizers import SynchronousAveragingOptimizer
        return SynchronousAveragingOptimizer(optimizer)
    else:
        raise RuntimeError('unknown optimizer: %s' % name)


def pre_process(X):
    # normalize inputs from 0-255 to 0.0-1.0
    X = X.astype('float32')
    X = X / 255.0
    return X


def one_hot_encode(y):
    # one hot encode outputs
    y = np_utils.to_categorical(y)
    num_classes = y.shape[1]
    return y, num_classes

from keras.constraints import maxnorm
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

def define_model_v1(optimizer, num_classes):

    # create a model with keras
    model = tf.keras.Sequential()

    # add two hidden layer
    model.add(tf.keras.layers.Dense(load_data.img_width, activation='relu'))
    model.add(tf.keras.layers.Dense(load_data.img_height, activation='relu'))

    # add a dense layer with number of classes of nodes and softmax
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # compile the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def define_model_v2(optimizer, num_classes):

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(load_data.img_width, load_data.img_height, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model

def define_model_v3(optimizer, num_classes):

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(load_data.img_width, load_data.img_height, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='KungFu mnist example.')
    parser.add_argument('--kf-optimizer',
                        type=str,
                        default='sync-sgd',
                        help='kungfu optimizer')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=50,
                        help='number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='batch size')
    return parser.parse_args()


def train_model(model, x_train, y_train, x_val, y_val, n_epochs=1, batch_size=32):

    n_shards = current_cluster_size()
    shard_id = current_rank()
    train_data_size = len(x_train)

    # calculate the offset for the data of the KungFu node
    shard_size = train_data_size // n_shards
    offset = batch_size * shard_id

    # extract the data for learning of the KungFu node
    x = x_train[offset:offset + shard_size]
    y = y_train[offset:offset + shard_size]

    # train the model
    model.fit(x,
              y,
              batch_size=batch_size,
              epochs=n_epochs,
              callbacks=[BroadcastGlobalVariablesCallback(with_keras=True)],
              validation_data=(x_val, y_val),
              verbose=2)
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    # parse arguments from the command line
    args = parse_args()

    # build the KungFu optimizer
    optimizer = build_optimizer(args.kf_optimizer)

    # load data
    X_train, Y_train, x_val, y_val = load_data.load_datasets_rgb()

    # pre process
    X_train = pre_process(X_train)
    x_val = pre_process(x_val)

    # one hot encode
    # Y_train, num_classes = one_hot_encode(Y_train)
    # y_val, num_classes = one_hot_encode(y_val)

    # Convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    # build the Tensorflow model
    model = define_model_v2(optimizer, num_classes)

    # train the Tensorflow model
    train_model(model, X_train, Y_train, x_val, y_val, args.n_epochs, args.batch_size)


if __name__ == '__main__':
    main()
