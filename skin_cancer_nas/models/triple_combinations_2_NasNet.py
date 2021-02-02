import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras import backend as K
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from keras.models import load_model, Model
import pickle
from keras.applications.nasnet import NASNetMobile, preprocess_input
from keras.layers import GlobalAveragePooling2D, Input, Dense, BatchNormalization, Dropout
from keras.utils.np_utils import to_categorical
from keras import optimizers
from numpy.random import seed
seed(777)
from tensorflow import set_random_seed
set_random_seed(777)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage

from keras import layers
from keras import models
from keras import callbacks

# %matplotlib inline

from PIL import Image

from itertools import combinations
L = [i for i in range(0, 51, 2)]

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

path = 'skin_data'
data_folder = 'data/interim/skin_data'

with open(os.path.join(data_folder, 'X_loc.pkl'), 'rb') as f:
    images = pickle.load(f)

with open(os.path.join(data_folder, 'y_loc.pkl'), 'rb') as f:
    values = pickle.load(f)

with open(os.path.join(data_folder, 'X_vs4.pkl'), 'rb') as f:
    images_vs4 = pickle.load(f)

with open(os.path.join(data_folder, 'y_vs4.pkl'), 'rb') as f:
    values_vs4 = pickle.load(f)

data_vs4 = []
labels_vs4 = []
data_loc = []
labels_loc = []
for i in range(0, 51, 2):
    data_vs4.append(images_vs4[i][:222, :, :, 1])
    data_loc.append(images[i][:, :, :, 1])
    labels_vs4.append(values_vs4[i][:222])
    labels_loc.append(values[i])

data_vs4 = np.swapaxes(np.array(data_vs4), 0, 1)
data_loc = np.swapaxes(np.array(data_loc), 0, 1)
data = np.concatenate((data_vs4, data_loc))
data = np.swapaxes(np.array(data), 1, 3)

print(data.shape)

labels_vs4 = np.swapaxes(np.array(labels_vs4), 0, 1)
labels_loc = np.swapaxes(np.array(labels_loc), 0, 1)
labels = np.concatenate((labels_vs4, labels_loc))
labels = labels[:, 0]

print(labels.shape)

def get_data_three_channels(ch1, ch2, ch3):
    d = data[:, :, :, [int(ch1/2), int(ch2/2), int(ch3/2)]]
    print("Original dataset: ", d.shape)
    return d

def mirroring(img):
    image_obj = Image.fromarray(img)
    mir_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(mir_image)

def augment_data(data, labels):
    rotated_data = []
    rotated_labels = []
    for i, img in enumerate(data):
        rotated_imgs = [ndimage.rotate(img, degree*90) for degree in range(4)]
        rotated_data = rotated_data + rotated_imgs + [mirroring(img) for img in rotated_imgs]
        rotated_labels = rotated_labels + 8*[labels[i]]
    d = np.array(rotated_data)
    print('Augmented dataset: ', d.shape)
    return d, rotated_labels

# def build_model():
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(512, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     return model

def build_model():
#     base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#     base_model = keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    base_model = NASNetMobile(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    _output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = _output)
    for layer in base_model.layers:
        layer.trainable = False
    return model

print(build_model().summary())

triple_combs = [comb for comb in combinations(L, 3)]

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# __space = hp.choice('attributes', [
#     {
#         'ch1': hp.randint('ch1_label', 25), #*10 + 450,
#         'ch2': hp.randint('ch2_label', 25), #*10 + 450,
#         'ch3': hp.randint('ch3_label', 25), #*10 + 450        
#     }])

# histories_list = []
channels_list = []

def objective(x, f_log_metrics):
    ch1 = x['ch1']
    ch2 = x['ch2']
    ch3 = x['ch3']

#     k = 5
    
    gc.collect
    K.clear_session()
    
    X = get_data_three_channels(ch1, ch2, ch3)    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=777)
    
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    X_test_aug, y_test_aug = augment_data(X_test, y_test)

    X_train_aug = X_train_aug.astype(float)
    X_train_aug = X_train_aug/255
    encoder = LabelEncoder()
    encoder.fit(y_train_aug)
    y_train_aug = encoder.transform(y_train_aug)
    
    X_test_aug = X_test_aug.astype(float)
    X_test_aug = X_test_aug/255
    y_test_aug = encoder.transform(y_test_aug)
    
    model = build_model()
    adam = optimizers.Adam(lr=0.00005, decay=0.00005)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[f1])
    
    log_callback = callbacks.LambdaCallback(on_epoch_end=lambda _, logs: f_log_metrics(logs=logs))

    history = model.fit(X_train_aug, y_train_aug, batch_size=32, epochs=75, 
                        validation_data=(X_test_aug, y_test_aug), verbose=0, class_weight={1:10, 0:1}, callbacks=[log_callback])

    # histories_list.append(history)
    channels_list.append([ch1] + [ch2] + [ch3])
    
#     print('hist')
#     print(history.history['val_f1'])
    try:
        print('sum='+str(np.array(history.history['val_f1']).sum()) + '; channels='+str([ch1] + [ch2] + [ch3]))
    except:
        print('printing failed... continuing.')
        
    # return -1 * np.array(history.history['val_f1']).sum()
    return { 'history': history.history, 'channels': str([ch1] + [ch2] + [ch3]) }
    
#     return history, max(history.history['f1']), max(history.history['val_f1'])
#     return {
#         'loss': -1 * np.ndarray(history.history['val_f1']).sum(),
#         'status': STATUS_OK,
# #         'f1' : max(history.history['f1']),
# #         'val_f1' : max(history.history['val_f1']),
#         'ch1' : ch1,
#         'ch2' : ch2,
#         'ch3' : ch3,
#         'channels' : [ch1] + [ch2] + [ch3]
#     }


# # trials = Trials()
# best = fmin(objective,
#     space=__space,
#     algo=tpe.suggest,
#     max_evals=3)  #1300) #, 2300
# #     trials=trials)

# print(best)

