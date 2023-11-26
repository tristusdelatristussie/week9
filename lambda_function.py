#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
#from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x /= 255
    x -= 1.
    return x


img_dl = download_image('https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg')
img_prep = prepare_image(img_dl,(150,150))
x = np.array(img_prep, dtype='float32')
X = np.array([x])
X = preprocess_input(X)

import os
MODEL_NAME = os.getenv('MODEL_NAME', 'bees-wasps-v2.tflite')
interpreter = tflite.Interpreter(model_path=MODEL_NAME)

#interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

#not used
classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

# url = 'http://bit.ly/mlbookcamp-pants'

def predict(url):
    
    img_dl = download_image(url)
    img_prep = prepare_image(img_dl,(150,150))
    x = np.array(img_prep, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)
    
    #X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    #return dict(zip(classes, float_predictions))
    return float_predictions

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result