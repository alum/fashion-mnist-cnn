import os
import scipy.ndimage
import numpy as np
import pandas as pd
import io
import IPython
import collections
import math
import datetime

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split, GridSearchCV

(x_train_indata, y_train_indata), (x_test_indata, y_test_indata) = fashion_mnist.load_data()

num_classes = 10

x_train, x_validation, y_train, y_validation = train_test_split(
    x_train_indata,
    y_train_indata,
    test_size=0.16666,
    random_state=42)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
y_train = to_categorical(y_train, num_classes=num_classes)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)
y_validation = to_categorical(y_validation, num_classes=num_classes)
x_train = inception_resnet_v2.preprocess_input(x_train)
x_validation = inception_resnet_v2.preprocess_input(x_validation)

def eval(model):
    score = model.evaluate(x_validation, y_validation, verbose=0)
    for i, _ in enumerate(score):
        print(model.metrics_names[i], score[i])

def model2(learning_rate=0.01, momentum=0.9, decay=1e-6):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(16, kernel_size=(2,2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(16, kernel_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.custom_name = 'model2'

    sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Compiled model {} with lr={} momentum={}, decay={}".format(model.custom_name, learning_rate, momentum, decay))
    return model

def prepare_img_data_for_inception():
    fname1 = 'x_train_inc.npy'
    fname2 = 'x_validation_inc.npy'
    if not (os.path.isfile(fname1) and os.path.isfile(fname2)):
        print('generating...')
        x_train_zoom = scipy.ndimage.zoom(x_train, (1,3,3,1), order=0)
        x_validation_zoom = scipy.ndimage.zoom(x_validation, (1,3,3,1), order=0)
        x_train_inc = np.repeat(x_train_zoom, 3, axis=3)
        x_validation_inc = np.repeat(x_validation_zoom, 3, axis=3)
        np.save(fname1, x_train_inc)
        np.save(fname2, x_validation_inc)
        print('done generating')
    else:
        print('loading...')
        x_train_inc = np.load(fname1)
        x_validation_inc = np.load(fname2)
        print('done loading')
    return x_train_inc, x_validation_inc


def preprocess_inception():
    print("creating inception model (this takes a while)...")
    inception = inception_resnet_v2.InceptionResNetV2(
                                                    include_top=False,
                                                    weights='imagenet',
                                                    input_tensor=None,
                                                    input_shape=(84,84,3),
                                                    pooling=None)
    print("extracting features from training images...")
    features_train = inception.predict(x_train_inc, verbose=1)
    print("extracting features from validation images...")
    features_validation = inception.predict(x_validation_inc, verbose=1)
    print("done extracting")
    # reshape, because predict() returns an array with shape ("num samples", 1, 1, "output layer size")
    features_train = np.reshape(features_train, (features_train.shape[0], features_train.shape[3]))
    features_validation = np.reshape(features_validation, (features_validation.shape[0], features_validation.shape[3]))
    return features_train, features_validation

feature_file_name1 = '/data/inception_features_train.npy'
feature_file_name2 = '/data/inception_features_validation.npy'
if not (os.path.isfile(feature_file_name1) and os.path.isfile(feature_file_name2)):
    x_train_inc, x_validation_inc = prepare_img_data_for_inception()
    features_train, features_validation = preprocess_inception()
    np.save(feature_file_name1, features_train)
    np.save(feature_file_name2, features_validation)
else:
    features_train = np.load(feature_file_name1)
    features_validation = np.load(feature_file_name2)


def model3(learning_rate=0.01, momentum=0.9, decay=1e-6):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(features_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.custom_name = "model3"
    sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Compiled model {} with lr={} momentum={}, decay={}".format(model.custom_name, learning_rate, momentum, decay))

    return model

# Perform two grid searches to find better epoch/batch_size and learning_rate/momentum parameters.
# The first run takes ~2h30m on a K80 GPU and the second one
# Uncomment below to activate

# Support TensorBoard callbacks when running through the KerasClassifer wrapper
# KerasClassifierTB adapted from https://stackoverflow.com/questions/45454905/how-to-use-keras-tensorboard-callback-for-grid-search
class KerasClassifierTB(KerasClassifier):
    def __init__(self, *args, **kwargs):
        super(KerasClassifierTB, self).__init__(*args, **kwargs)

    def fit(self, x, y, log_dir=None, **kwargs):
        cbs = None
        if log_dir is not None:
            # Make sure the base log directory exists
            try:
                os.makedirs(log_dir)
            except OSError:
                pass
            params = self.get_params()
            params.pop("build_fn", None)
            conf = ",".join("{}={}".format(k, params[k])
                            for k in sorted(params))
            conf_dir_base = os.path.join(log_dir, conf)
            # Find a new directory to place the logs
            try:
                d_string = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
                conf_dir = "{}_{}".format(conf_dir_base, d_string)
                os.makedirs(conf_dir)
            except OSError:
                pass
            cbs = [TensorBoard(log_dir=conf_dir, histogram_freq=0,
                               write_graph=False, write_images=False)]
        super(KerasClassifierTB, self).fit(x, y, callbacks=cbs, **kwargs)

# Grid search code adapted from https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
def run_grid_search_e_bs(model_name, model_fn, param_grid, x, y):
    model_wrapper = KerasClassifierTB(build_fn=model_fn, verbose=2)
    grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(x, y, log_dir="./graph/{}".format(model_name))
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def search_e_bs():
    seed = 7
    np.random.seed(seed)
    param_grid = {
        "batch_size": [16, 32, 128, 512, 2048],
        "epochs": [10, 30, 50]
    }
    run_grid_search_e_bs("model2", model2, param_grid, x_train, y_train)
    run_grid_search_e_bs("model3", model3, param_grid, features_train, y_train)

search_e_bs()

def run_grid_search_lr_m(model_name, model_fn, param_grid, x, y):
    model_wrapper = KerasClassifierTB(build_fn=model_fn, verbose=2, batch_size=128, epochs=50)
    grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(x, y, log_dir="./graph/{}".format(model_name))
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def search_lr_m():
    seed = 7
    np.random.seed(seed)
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "momentum": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    }
    run_grid_search_lr_m("model2", model2, param_grid, x_train, y_train)
    run_grid_search_lr_m("model3", model3, param_grid, features_train, y_train)

search_lr_m()
