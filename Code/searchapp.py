#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:34:55 2021

@author: Arbo

"""


from capstone_twitter_search import twittsearch
import os
import pandas as pd

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
import datetime
import tensorflow as tf            
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
from mordecai import Geoparser
#from mordecai import batch_geoparse
from flatten_json import flatten_json
import numpy as np
import json
from tensorflow.keras import layers
import text_proccessing
from text_proccessing import clean_text
from text_proccessing import remove_stopwords
from text_proccessing import lemmatize_text
from text_proccessing import concatenate_text
from text_proccessing import makeglove
from text_proccessing import make_embedding_matrix
import modhelp
from modhelp import train_val_split
from modhelp import geo_df
from modhelp import suggest_nn2
# =============================================================================
# from kerastuner.tuners import RandomSearch
# from kerastuner.tuners import BayesianOptimization
# from kerastuner.tuners import Hyperband
# import kerastuner as kt
# =============================================================================
# =============================================================================
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# =============================================================================
overallstart = datetime.datetime.now() 
code_dir = os.getcwd()
print("Current working directory: {0}".format(code_dir))
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")
directory = os.path.join(data_dir,'splits')
dsa= os.path.dirname(parent_dir)
 
train_data = pd.read_csv(os.path.join(directory, 'InformativenessTrain_Processed.csv'))
print('cleaning')
start =  datetime.datetime.now() 
# Applying the cleaning function to both test and train datasets
train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))

train_data['text'] = train_data['text'].apply(lambda x:word_tokenize(x))
end =  datetime.datetime.now() 
print('cleand and totkeninzed{}'.format(end-start))

print('rv stop words')
start =  datetime.datetime.now() 
train_data['text'] = train_data['text'].apply(lambda x : remove_stopwords(x))
end =  datetime.datetime.now() 
print('rvd stop words{}'.format(end-start))

print('lemmatime')
start =  datetime.datetime.now() 
train_data['text'] = train_data['text'].apply(lambda x : lemmatize_text(x))
end =  datetime.datetime.now() 
print('lemmad in {}'.format(end-start))

print('concat')
start =  datetime.datetime.now() 
train_data['text'] = train_data['text'].apply(lambda x : concatenate_text(x))
end =  datetime.datetime.now() 
print('concated in {}'.format(end-start))

print('splitthis')
start =  datetime.datetime.now() 
train_samples, val_samples, train_labels, val_labels = train_val_split(train_data, 0.25)
end =  datetime.datetime.now() 
print ('Text done {}'.format(end-start))

print('Loading model,embedding and glove')
start =  datetime.datetime.now() 
model = tf.keras.models.load_model(os.path.join(dsa,'model1'))
path_to_glove_file = os.path.join(dsa,'WordVector','glove.twitter.27B.200d.txt')
embeddings_index=makeglove (path_to_glove_file)

embedding_matrix, vectorizer = make_embedding_matrix(train_samples, val_samples, embeddings_index)
end =  datetime.datetime.now() 

print("embedded in {}".format(end-start))


print ("scrapeing ")
scrapestart =  datetime.datetime.now()
print(scrapestart) 
text_query = "heat OR fire OR forestfire OR earthquake OR heatwave OR disaster OR typhoon OR cyclone OR tornado OR thunder OR lightning  OR hail OR torrent OR flood OR deluge"
since_date = '2021-07-07'
until_date = '2021-07-13'

tweets_df = twittsearch(text_query,since_date,until_date)
scrapeend =  datetime.datetime.now() 
print ("scraped ended at {} total {}".format(scrapeend, scrapeend-scrapestart))


print("Run geoprocessing over tweets")

geostart =  datetime.datetime.now() 
print(geostart)
df_js = geo_df(tweets_df) 
geoend = datetime.datetime.now() 
print ("geo ended at  {} for a total time of {} ".format(geoend, geoend-geostart))

print("merge the dfs")       
twts = pd.merge(tweets_df, df_js, on="TweetId")


print("processing ")
procstart =  datetime.datetime.now() 
twts['ptext'] = twts['Text'].apply(lambda x: clean_text(x))
twts['ptext'] = twts['ptext'].apply(lambda x: word_tokenize(x))
twts['ptext'] = twts['ptext'].apply(lambda x : remove_stopwords(x))
twts['ptext'] = twts['ptext'].apply(lambda x : lemmatize_text(x))
twts['ptext'] = twts['ptext'].apply(lambda x : concatenate_text(x))
procend=  datetime.datetime.now() 
print("processed {}".format(procend - procstart))

print("predict")
predstart = datetime.datetime.now() 

predictions = suggest_nn2(twts, model,vectorizer)

submission_data = {"ID": twts['TweetId'].tolist(),"tweet": twts['Text'].tolist(), "target": predictions}

submission_df = pd.DataFrame(submission_data)
result = twts.join(submission_df.target)

result.to_csv(os.path.join(tweet_dir,'result.csv'),index =False)
predend = datetime.datetime.now() 
print("predicted {}".format(predend-predstart))

# =============================================================================
# tweets_geo_df = twittsearch(text_query,since_date,until_date)
# =============================================================================

# =============================================================================
# tweets_geo_df.to_csv(os.path.join(tweet_dir,"tweets_geo.csv"))
# tweets_no_geo_df.to_csv(os.path.join(data_dir,"tweets_no_geo.csv"))
# =============================================================================
overallend = datetime.datetime.now() 
print('fin in {}'.format(overallend-overallstart))
# =============================================================================
# 
# tweet_map = pd.read_csv(os.path.join(data_dir,"tweets_geo.csv"))
# =============================================================================



