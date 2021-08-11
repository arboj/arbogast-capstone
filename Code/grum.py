#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 23:46:58 2021

@author: Arbo
"""

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
import datetime

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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
overallstart = datetime.datetime.now() 
print ("started at: {}".format(overallstart))
code_dir = os.getcwd()
print("Current working directory: {0}".format(code_dir))
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")
directory = os.path.join(data_dir,'splits')
dsa= os.path.dirname(parent_dir)
train_data = pd.read_csv(os.path.join(directory, 'InformativenessTrain_Processed.csv'))
test_data  = pd.read_csv(os.path.join(directory, 'InformativenessTest_Processed.csv'))




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
path_to_glove_file = os.path.join(dsa,'WordVector','glove.twitter.27B.200d.txt')
train_samples, val_samples, train_labels, val_labels = train_val_split(train_data, 0.25)
test_samples, test_labels = test_listerine(test_data)


print("indexing")
embeddings_index=makeglove(path_to_glove_file)
step2 = datetime.datetime.now()
print("{}: indexed {} matrix and vectorize".format(step2, step2-step1))
embedding_matrix, vectorizer = make_embedding_matrix(train_samples, val_samples, embeddings_index)
step3 = datetime.datetime.now()
print("{}: mtx'd at vct'd in {} loading geoparse".format(step3, step3-step2))


initial_model = initialize_nn(embedding_matrix)
initial_model.summary()

model, history = train_nn(initial_model, train_samples, val_samples, train_labels, val_labels, vectorizer, stop=True)

predictions = suggest_nn3(test_data, model,vectorizer)

submission_data = {"ID": test_data['id'].tolist(),"tweet": test_data['text'].tolist(), "target": predictions}

submission_df = pd.DataFrame(submission_data)
result = test_data.join(submission_df.target)

result.to_csv(os.path.join(tweet_dir,"dubblecheck"),index =False)

