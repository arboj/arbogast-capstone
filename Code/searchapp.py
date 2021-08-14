#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:34:55 2021

@author: Arbo

Before running ensure an elastic instance to search geoname is running either in docker, 
or  mordecai is pointed to an elastic search instance


This code demonstrates how to pull data from the twitter API, run a prediction 
as to it's natural disater informativeness, and geoparse usign mordecai.

At line 146 the user must pass dates to query Twitter against

"""
import os
import pandas as pd
import numpy as np                               # linear algebra
import re                                        # to handle regular expressions
from string import punctuation                   # to extract the puntuation symbols
import nltk
from nltk.tokenize import word_tokenize          # to divide strings into tokens
from nltk.stem import WordNetLemmatizer          # to lemmatize the tokens
from nltk.corpus import stopwords                # to remove the stopwords 
from mordecai import Geoparser
from mordecai import batch_geoparse

import random                                    # for generating (pseudo-)random numbers
import matplotlib.pyplot as plt                  # to plot some visualizations
import datetime
import tensorflow as tf            
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping




import text_proccessing
from text_proccessing import clean_text
from text_proccessing import remove_stopwords
from text_proccessing import lemmatize_text
from text_proccessing import concatenate_text
from text_proccessing import makeglove
from text_proccessing import make_embedding_matrix
from text_proccessing import twtprocessed

import modhelp
from modhelp import train_val_split
from modhelp import geo_df
from modhelp import suggest_nn2

import capstone_twitter_search
from capstone_twitter_search import twittsearch
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
geotweets = os.path.join(parent_dir,"GeoTweets")
directory = os.path.join(data_dir,'splits')
dsa= os.path.dirname(parent_dir)
readstart = datetime.datetime.now() 


print ("{} - reading csv started".format(readstart))
# =============================================================================
# This section  will bring in the training data that was used train the model an build the orignial vectorizer
# =============================================================================
train_data = pd.read_csv(os.path.join(directory, 'InformativenessTrain_Tokenized.csv'),engine='python')
train_data['text']= train_data['text'].astype(str)
readend = datetime.datetime.now()
print ("{} - reading csv ended. Total time {}".format(readend,readend-readstart))

start =  datetime.datetime.now() 
print('splitthis started at: {}'.format(start))
# =============================================================================
# Render training data as lists and split to train and validation
# =============================================================================
train_samples, val_samples, train_labels, val_labels = train_val_split(train_data, 0.25)
end =  datetime.datetime.now() 
print ('Text done {}'.format(end-start))
start =  datetime.datetime.now()


print('Loading model,embedding and glove commenced at  {}'.format(start))
# =============================================================================
# Load trained RNN model
# =============================================================================
model = tf.keras.models.load_model(os.path.join(parent_dir,'model1'))
step1 = datetime.datetime.now()
print("{}: model loaded in {} loading vecotor".format(step1, step1-start))
path_to_glove_file = os.path.join(dsa,'WordVector','glove.twitter.27B.200d.txt')

print("indexing")
# =============================================================================
# Build embeddign index from GloVe
# =============================================================================
embeddings_index=makeglove(path_to_glove_file)
step2 = datetime.datetime.now()
print("{}: indexed {} matrix and vectorize".format(step2, step2-step1))
# =============================================================================
# Map index to to Matrix
# =============================================================================
embedding_matrix, vectorizer = make_embedding_matrix(train_samples, val_samples, embeddings_index)
step3 = datetime.datetime.now()
print("{}: mtx'd at vct'd in {} loading geoparse".format(step3, step3-step2))
# =============================================================================
# initiate a the geo parser
# =============================================================================
geo = Geoparser()
end =  datetime.datetime.now()
 
print("{}: geo complete in {} embedded in {}".format(end, end-step3, end-start))


scrapestart =  datetime.datetime.now() 
print("Scraping commenced at {}".format(scrapestart))
# =============================================================================
# Scrape tweets 
# =============================================================================

# =============================================================================
text_query = '("forest fire") OR wildfire OR bushfire OR \
(extreme heat) OR (record heat) OR heatwave OR ("heat wave") OR typhoon OR cyclone OR hurricane OR \
tornado OR ("storm surge") OR  blizzard OR snow OR ("ice storm") OR sleet OR thunderstorm OR \
hail OR flood OR flooding OR freeze OR frost OR (extreme cold) OR landslide OR tsunami OR ("tidal wave") OR \
earthquake OR eruption OR volcano OR lava OR lahar OR avalanche OR mudslide OR sinkhole'

since_date = '2021-06-01' # user must pick
until_date = '2021-06-02' # user must pick
tweetcount = 200000       # user must pick
twts = twittsearch(text_query,since_date,until_date,tweetcount) 

# =============================================================================
scrapend =  datetime.datetime.now()
print("Scraping ended at {}".format(scrapend))
print("Scraping time {}".format(scrapend-scrapestart))

procstart =  datetime.datetime.now() 
print("processing commenced at{}".format(procstart))
# =============================================================================
# Clean and process the scrapped twitter posts. Then save a csv file of the 
# preocessed data. This is uselfull for metrics.
# =============================================================================
twts = twtprocessed (twts)
twts.to_csv(os.path.join(tweet_dir,"processed_{}_{}.csv".format(since_date.replace('-',''),
                                                                until_date.replace('-',''))), index = False)
procend=  datetime.datetime.now()


print("processed {}".format(procend - procstart))
    

predstart =  datetime.datetime.now() 
# =============================================================================
# Predict informativenss
# =============================================================================
predictions = suggest_nn2(twts, model,vectorizer)

predicted_data= {"ID": twts['TweetId'].tolist(),"tweet": twts['Text'].tolist(), "target": predictions}

predicted_df = pd.DataFrame(predicted_data)
result = twts.join(predicted_df.target)
name =os.path.join(tweet_dir,"result_{}_{}.csv".format(since_date.replace('-',''),
                                                                until_date.replace('-','')))
result.to_csv(name,index =False)



# =============================================================================
# use just the informative data for geoparsing
# =============================================================================
result_inf = result[result['target']==0]
print("working over {} informative tweets".format(len(result_inf)))
predend = datetime.datetime.now() 
print("predicted {}".format(predend-predstart))

print("Run geoprocessing over tweets")

geostart =  datetime.datetime.now() 
# =============================================================================
# This initiaties the mordecai geoparsing module, uses tweets file from predicition
# =============================================================================
print(geostart)
df_js = geo_df(result_inf,geo,'Text') 
name =os.path.join(geotweets,"geod_{}_{}.csv".format(since_date.replace('-',''),
                                                                until_date.replace('-','')))
geoend = datetime.datetime.now()
df_js.to_csv(name, index = False)
print ("geo ended at  {} for a total time of {} ".format(geoend, geoend-geostart))
print ("pred and geo ended at  {} for a total time of {}.csv".format(geoend, geoend-predstart))

overallend = datetime.datetime.now() 
print('fin in {}'.format(overallend-overallstart))




