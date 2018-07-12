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
from functions import *

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

data=load_data(SOURCES)

new_index=[x for x in range(len(data))]
data.index=new_index

data['tokenized_text']=data.apply(tokenize, axis=1)
data['token_count']=data.apply(token_count, axis=1)
data['lang']='en'

df=data
len_unseen = 10000
df_unseen_test= df.iloc[:len_unseen]
df_model = df.iloc[len_unseen:]
texts=list(df_model['tokenized_text'])
num_max = 4000
tfidf_model=train_tf_idf_model(texts, num_max)

# prepare model input data
sample_texts,sample_target=prepare_model_input(tfidf_model,df_unseen_test,mode='')

predicted_sample = predict(sample_texts, model)
print(predicted_sample)