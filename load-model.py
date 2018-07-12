from keras.layers import SimpleRNN, Embedding, Dense, LSTM, Input, Dense, Dropout, Activation
from keras.models import Sequential, model_from_json, Model,load_model
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

import os

import time
import pickle

import sys

import keras
import tensorflow as tf 

import sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

import functions

HAM = 'ham'
SPAM = 'spam'

SOURCES = [
    ('datos/GP',        SPAM),
    ('datos/kitchen-l', HAM),

]

# load json and create model
'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("%s: %.2f%%" % (loaded_model.metrics_names[1]))

'''

model_path = "model/model.ckpt"
tf.train.NewCheckpointReader(model_path)

with tf.Session() as sess:
# Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.import_meta_graph('model/model.ckpt.meta')

# Restore model weights from previously saved model
    saver.restore(sess, model_path)
model = load_model('my_model.h5')
predicted_sample = predict(sample_texts, model)