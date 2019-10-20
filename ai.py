#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:14:13 2019

@author: cihanerman
"""

#%% library
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
import cv2

#%% data pre-processing

dataset_path = "train.csv"
image_size=(48,48)
#test = pd.read_csv("test.csv")

#print(train.head())
#print(test.head())
#x = train["Pixels"]
#y = train["Emotion"]

def load_data(path):
    data = pd.read_csv(path)
    pixels = data['Pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['Emotion']).as_matrix()
    return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

faces, emotions = load_data(dataset_path)
faces = preprocess_input(faces)
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

#ytrain = to_categorical(ytrain, num_classes=7)
#ytest = to_categorical(ytest, num_classes=7)

v = xtrain.reshape(xtrain.shape[0],image_size[0],image_size[1])
plt.imshow(v[13, :, :], cmap="gray")
plt.axis('off')
plt.show()
#%% model
# parameters
batch_size = 32
num_epochs = 150
input_shape = (48, 48, 1)
num_classes = 7

model = Sequential()
model.add(Conv2D(8,(3,3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(8,(3,3), padding='same',))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(16,(3,3), padding='same',))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(16,(3,3), padding='same', ))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 3),  padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])

model.summary()

#%% train, data generation
 
# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

hist = model.fit_generator(data_generator.flow(xtrain, ytrain,batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1,
                        validation_data=(xtest,ytest))

#%% model save
model.save("model.h5")
#%% model evalation
print(hist.history.keys())
plt.plot(hist.history['loss'], label='Trainin loss')
plt.plot(hist.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history['acc'], label='Trainin acc')
plt.plot(hist.history['val_acc'], label='Validation acc')
plt.legend()
plt.show()
#%% save history
import json
with open('cnn_fruit_hist.json', 'w') as f:
    json.dump(hist.history, f)