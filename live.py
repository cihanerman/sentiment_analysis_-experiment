#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 01:47:12 2019

@author: cihanerman
"""
#%% library
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
 
#%% code
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'model.h5'
dataset_path = 'test.csv'
image_size=(48,48)

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
    return faces

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

faces = load_data(dataset_path)
faces = preprocess_input(faces)
v = faces.reshape(faces.shape[0],image_size[0],image_size[1])
#plt.imshow(v[13, :, :], cmap="gray")
#plt.axis('off')
#plt.show()


face_detection = cv2.CascadeClassifier("haarcascede_frontalface_default.xml")
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
orig_frame = cv2.imread("1.png")
frame = cv2.imread("1.png",0)
cv2.imshow('frame',orig_frame)
faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
 
faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
(fX, fY, fW, fH) = faces
roi = frame[fY:fY + fH, fX:fX + fW]
roi = cv2.resize(roi, (48, 48))
roi = roi.astype("float") / 255.0
roi = img_to_array(roi)
roi = np.expand_dims(roi, axis=0)
preds = emotion_classifier.predict(roi)[0]
emotion_probability = np.max(preds)
label = EMOTIONS[preds.argmax()]
cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
 
cv2.imshow('test_face', orig_frame)
cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
if (cv2.waitKey(2000) & 0xFF == ord('q')):
    sys.exit("Thanks")
cv2.destroyAllWindows()