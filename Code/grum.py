#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 23:46:58 2021

@author: Arbo
"""
import datetime
overallstart = datetime.datetime.now() 
print ("started at: {}".format(overallstart))

import numpy as np                               # linear algebra
import pandas as pd                              # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re                                        # to handle regular expressions
from string import punctuation                   # to extract the puntuation symbols
import nltk
from nltk.tokenize import word_tokenize          # to divide strings into tokens
from nltk.stem import WordNetLemmatizer          # to lemmatize the tokens
from nltk.corpus import stopwords                # to remove the stopwords 

import random                                    # for generating (pseudo-)random numbers
import matplotlib.pyplot as plt                  # to plot some visualizations

import tensorflow as tf            
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import layers
# =============================================================================
# from kerastuner.tuners import RandomSearch
# from kerastuner.tuners import BayesianOptimization
# from kerastuner.tuners import Hyperband
# import kerastuner as kt
# =============================================================================


import text_proccessing
from text_proccessing import clean_text
from text_proccessing import remove_stopwords
from text_proccessing import lemmatize_text
from text_proccessing import concatenate_text
from text_proccessing import makeglove
from text_proccessing import make_embedding_matrix
from modhelp import train_val_split
from modhelp import test_listerine
from modhelp import suggest_nn3
from modhelp import initialize_nn
from modhelp import  train_nn

starti = datetime.datetime.now() 
print("{}: modules loaded in {} loading data".format(starti, starti-overallstart))
code_dir = os.getcwd()
print("Current working directory: {0}".format(code_dir))
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")
directory = os.path.join(data_dir,'splits')
dsa= os.path.dirname(parent_dir)
train_data = pd.read_csv(os.path.join(directory, 'InformativenessTrain_Processed.csv'))
test_data  = pd.read_csv(os.path.join(directory, 'InformativenessTest_Processed.csv'))
dtl = datetime.datetime.now()
print("{}: data loaded in {} cleaning text elapsed: {}".format(starti,dtl-starti,dtl-overallstart))


train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))
test_data['text'] = test_data['text'].apply(lambda x: clean_text(x))
train_data['text'] = train_data['text'].apply(lambda x:word_tokenize(x))
test_data['text'] = test_data['text'].apply(lambda x:word_tokenize(x))
train_data['text'] = train_data['text'].apply(lambda x : remove_stopwords(x))
test_data['text'] = test_data['text'].apply(lambda x : remove_stopwords(x))
train_data['text'] = train_data['text'].apply(lambda x : lemmatize_text(x))
test_data['text'] = test_data['text'].apply(lambda x : lemmatize_text(x))
train_data['text'] = train_data['text'].apply(lambda x : concatenate_text(x))
test_data['text'] = test_data['text'].apply(lambda x : concatenate_text(x))

step1 = datetime.datetime.now()
print("{}: text cleand in {} loading vector and spliting samples elapsed: {}".format(step1,dtl-step1, step1-overallstart))
path_to_glove_file = os.path.join(dsa,'WordVector','glove.twitter.27B.200d.txt')
train_samples, val_samples, train_labels, val_labels = train_val_split(train_data, 0.25)
# =============================================================================
# test_samples, test_labels = test_listerine(test_data)
# =============================================================================
print("indexing")
embeddings_index=makeglove(path_to_glove_file)
step2 = datetime.datetime.now()
print("{}: indexed {} matrix and vectorize ellapsed {}".format(step2, step2-step1, step2-overallstart))
embedding_matrix, vectorizer = make_embedding_matrix(train_samples, val_samples, embeddings_index)
step3 = datetime.datetime.now()
print("{}: mtx'd at vct'd in {} loading variables as vectors ellapsed {}".format(step3, step3-step2,  step3-overallstart))

x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()
# =============================================================================
# x_test = vectorizer(np.array([[s] for s in test_samples])).numpy()
# =============================================================================
y_train = np.asarray(train_labels).astype('float32').reshape((-1,1))
y_val = np.asarray(val_labels).astype('float32').reshape((-1,1))
# =============================================================================
# y_test = np.asarray(test_labels).astype('float32').reshape((-1,1))
# =============================================================================
step4 = datetime.datetime.now()
print("{}: training data ready in {} buildinf initial mod  ellapsed {}".format(step4, step4-step3,  step4-overallstart))

from sklearn.model_selection import KFold
import numpy as np


num_folds = 5
acc_list = []
acc_per_fold = []
loss_list = []
loss_per_fold = []
inputs = np.concatenate((x_train,x_val), axis=0)
targets = np.concatenate((y_train,y_val), axis=0)
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1

cppath = os.path.join(dsa,"cp.ckpt")
step5 = datetime.datetime.now()
print("{}: targets aqcuired in {} building on to model ellapsed {}".format(step5, step5-step4,  step5-overallstart))

for train, test in kfold.split(inputs, targets):
  step6 = datetime.datetime.now()
  print("{}: last fold in {} next fold, ellapsed {}".format(step6, step6-step5,  step6-overallstart))
    
  
  """
  This function initializes Keras model for binary text classification

  Parameters:
      embedding matrix with the dimensions (num_tokens, embedding_dim),
        where num_tokens is the vocabulary size of the input data,
        and emdebbing_dim is the number of components in the GloVe vectors

  Returns:
      model: Keras model    
  """

  num_tokens = embedding_matrix.shape[0]
  embedding_dim = embedding_matrix.shape[1]

  embedding_layer = Embedding(
  num_tokens,
  embedding_dim,
  embeddings_initializer=keras.initializers.Constant(embedding_matrix),
  trainable=False,                # we are not going to train the embedding vectors
  )

  #   Here we define the architecture of the Keras model. 
  int_sequences_input = keras.Input(shape=(None,), dtype="int64")
  x = embedding_layer(int_sequences_input) 
  x = layers.Dropout(.7)(x)
  x = layers.Bidirectional(layers.LSTM(128,                                        
                                        dropout=.4,
                                        return_sequences=True))(x)
  x = layers.Bidirectional(layers.LSTM(32,
                                        dropout=.5))(x)
  x = layers.Dense(128,activation= 'sigmoid')(x)
  x = layers.Dropout(.5)(x)
  preds = layers.Dense(1, activation='sigmoid')(x)
  model = keras.Model(int_sequences_input, preds)

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy',])

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train],
              batch_size=128,
              epochs=120,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3,
                                                          restore_best_weights = True,
                                                          verbose = 1, min_delta = .0002 ),
                         tf.keras.callbacks.ModelCheckpoint(filepath=cppath,
                                                 save_weights_only=True,
                                                 verbose=1)
                         ],
              verbose=1)
  acc = history.history['binary_accuracy']
  prec = history.history[]
  recall = history.history[]
  loss = history.history['loss']
  acc_list.append(acc)
  loss_list.append(loss)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1
step7 = datetime.datetime.now()
print("{}: training complete {} scores ellapsed {}".format(step7, step7-step5,  step7-overallstart))
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

model.save(os.path.join(dsa, 'model2'))

predictions = suggest_nn3(test_data, model,vectorizer)

submission_data = {"ID": test_data['id'].tolist(),"tweet": test_data['text'].tolist(), "target": predictions}

submission_df = pd.DataFrame(submission_data)
result = test_data.join(submission_df.target)

result.to_csv(os.path.join(tweet_dir,"dubblecheck"),index =False)

