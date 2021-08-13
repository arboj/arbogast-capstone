#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:34:55 2021

@author: Arbo

"""



import os
import pandas as pd

import numpy as np                               # linear algebra

# =============================================================================
# import re                                        # to handle regular expressions
# from string import punctuation                   # to extract the puntuation symbols
# import nltk
# =============================================================================
from nltk.tokenize import word_tokenize          # to divide strings into tokens
# =============================================================================
# from nltk.stem import WordNetLemmatizer          # to lemmatize the tokens
# from nltk.corpus import stopwords                # to remove the stopwords 
# 
# import random                                    # for generating (pseudo-)random numbers
# import matplotlib.pyplot as plt                  # to plot some visualizations
# =============================================================================
import datetime
import tensorflow as tf            
# =============================================================================
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.callbacks import EarlyStopping
# from mordecai import Geoparser
# #from mordecai import batch_geoparse
# from flatten_json import flatten_json
# import numpy as np
# import json
# from tensorflow.keras import layers
# =============================================================================
from mordecai import Geoparser

import text_proccessing
# =============================================================================
# from text_proccessing import clean_text
# from text_proccessing import remove_stopwords
# from text_proccessing import lemmatize_text
# from text_proccessing import concatenate_text
# =============================================================================
from text_proccessing import makeglove
from text_proccessing import make_embedding_matrix

import modhelp
from modhelp import train_val_split
from modhelp import geo_df
from modhelp import suggest_nn2

# =============================================================================
# import capstone_twitter_search
# from capstone_twitter_search import twittsearch
# =============================================================================
# =============================================================================
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# =============================================================================
overallstart = datetime.datetime.now() 
print ("started at: {}".format(overallstart))
code_dir = os.getcwd()
print("Current working directory: {0}".format(code_dir))
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")
directory = os.path.join(data_dir,'splits')
dsa= os.path.dirname(parent_dir)
readstart = datetime.datetime.now() 


print ("{} - reading csv started".format(readstart))
train_data = pd.read_csv(os.path.join(directory, 'InformativenessTrain_Tokenized.csv'),engine='python')
train_data['text']= train_data['text'].astype(str)
readend = datetime.datetime.now()
print ("{} - reading csv ended. Total time {}".format(readend,readend-readstart))

start =  datetime.datetime.now() 
print('splitthis started at: {}'.format(start))
train_samples, val_samples, train_labels, val_labels = train_val_split(train_data, 0.25)
end =  datetime.datetime.now() 
print ('Text done {}'.format(end-start))
start =  datetime.datetime.now() 
print('Loading model,embedding and glove commenced at  {}'.format(start))

model = tf.keras.models.load_model(os.path.join(dsa,'model1'))
step1 = datetime.datetime.now()
print("{}: model loaded in {} loading vecotor".format(step1, step1-start))
path_to_glove_file = os.path.join(dsa,'WordVector','glove.twitter.27B.200d.txt')

print("indexing")
embeddings_index=makeglove(path_to_glove_file)
step2 = datetime.datetime.now()
print("{}: indexed {} matrix and vectorize".format(step2, step2-step1))
embedding_matrix, vectorizer = make_embedding_matrix(train_samples, val_samples, embeddings_index)
step3 = datetime.datetime.now()
print("{}: mtx'd at vct'd in {} loading geoparse".format(step3, step3-step2))

geo = Geoparser()
end =  datetime.datetime.now()
 
print("{}: geo complete in {} embedded in {}".format(end, end-step3, end-start))

# =============================================================================
# 
#  
# 
# text_query = "heat OR fire OR forestfire OR earthquake OR heatwave OR disaster OR typhoon OR cyclone OR tornado OR thunder OR lightning  OR hail OR torrent OR flood OR deluge"
# tweetcount = 200000
# # =============================================================================
# # since_dates = ["2021-06-01","2021-06-02","2021-06-03","2021-06-04","2021-06-05","2021-06-06","2021-06-07", "2021-06-08","2021-06-09",
# #        "2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14","2021-06-15","2021-06-16","2021-06-17", "2021-06-18",
# #        "2021-06-19","2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24","2021-06-25","2021-06-26","2021-06-27", 
# #        "2021-06-28","2021-06-29","2021-06-30"]
# # until_dates = ["2021-06-02","2021-06-03","2021-06-04","2021-06-05","2021-06-06","2021-06-07", "2021-06-08","2021-06-09",
# #        "2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14","2021-06-15","2021-06-16","2021-06-17", "2021-06-18",
# #        "2021-06-19","2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24","2021-06-25","2021-06-26","2021-06-27", 
# #        "2021-06-28","2021-06-29","2021-06-30","2021-07-01"]
# # =============================================================================
# 
# since_dates = ["2021-06-03","2021-06-04","2021-06-05","2021-06-06","2021-06-07", "2021-06-08","2021-06-09",
#        "2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14","2021-06-15","2021-06-16","2021-06-17", "2021-06-18",
#        "2021-06-19","2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24","2021-06-25","2021-06-26","2021-06-27", 
#        "2021-06-28","2021-06-29","2021-06-30"]
# until_dates = ["2021-06-04","2021-06-05","2021-06-06","2021-06-07", "2021-06-08","2021-06-09",
#        "2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14","2021-06-15","2021-06-16","2021-06-17", "2021-06-18",
#        "2021-06-19","2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24","2021-06-25","2021-06-26","2021-06-27", 
#        "2021-06-28","2021-06-29","2021-06-30","2021-07-01"]
# 
# 
# 
# 
# 
# 
# for x in range(len(since_dates)):
#     print ("scrapeing ")
#     scrapestart =  datetime.datetime.now()
#     print(scrapestart)
# 
#     since_date = since_dates[x]
#     until_date = until_dates[x]
#     
#     twts = twittsearch(text_query,since_date,until_date,tweetcount)
#     
#     # =============================================================================
#     # for table in os.listdir(tweet_dir):
#     #     if table.startswith("tweets_df"):
#     #         twts = pd.read_csv(os.path.join(tweet_dir,table), engine = 'python', )
#     #         twts = twts[['Datetime', 'TweetId', 'Text']]
#     #         twts['Text'] = twts['Text'].astype(str) 
#     # =============================================================================
#     
#     print("working over {} records".format(len(twts)))
# 
#     scrapeend =  datetime.datetime.now() 
#     print ("scraped ended at {} total {}".format(scrapeend, scrapeend-scrapestart))
#     # 
#     # =============================================================================
#     # print("Run geoprocessing over tweets")
#     # 
#     # geostart =  datetime.datetime.now() 
#     # print(geostart)
#     # df_js = geo_df(tweets_df) 
#     # geoend = datetime.datetime.now() 
#     # print ("geo ended at  {} for a total time of {} ".format(geoend, geoend-geostart))
#     # 
#     # print("merge the dfs")       
#     # twts = pd.merge(tweets_df, df_js, on="TweetId")
#     # 
#     # =============================================================================
#     
#     print("processing ")
#     procstart =  datetime.datetime.now() 
#     twts['ptext'] = twts['Text'].apply(lambda x: clean_text(x))
#     twts['ptext'] = twts['ptext'].apply(lambda x: word_tokenize(x))
#     twts['ptext'] = twts['ptext'].apply(lambda x : remove_stopwords(x))
#     twts['ptext'] = twts['ptext'].apply(lambda x : lemmatize_text(x))
#     twts['ptext'] = twts['ptext'].apply(lambda x : concatenate_text(x))
#     twts.to_csv(os.path.join(tweet_dir,"processed_{}_{}.csv".format(since_date.replace('-',''),until_date.replace('-',''))), index = False)
#     procend=  datetime.datetime.now()
#     
#     
#     print("processed {}".format(procend - procstart))
# =============================================================================
    
for table in os.listdir(tweet_dir):
    if table.startswith("processed_20210731"):
        twts = pd.read_csv(os.path.join(tweet_dir,table), engine = 'python', )
        twts['Text'] = twts['Text'].astype(str)
        twts['ptext'] = twts['ptext'].astype(str) 
    
        print("predict")
        predstart = datetime.datetime.now() 
        
        predictions = suggest_nn2(twts, model,vectorizer)
        
        submission_data = {"ID": twts['TweetId'].tolist(),"tweet": twts['Text'].tolist(), "target": predictions}
        
        submission_df = pd.DataFrame(submission_data)
        result = twts.join(submission_df.target)
        name =table.replace("processed","result")
        
        result.to_csv(os.path.join(tweet_dir,name),index =False)
        
    
        
        
        result_inf = result[result['target']==0]
        print("working over {} informative tweets".format(len(result_inf)))
        predend = datetime.datetime.now() 
        print("predicted {}".format(predend-predstart))
        
        print("Run geoprocessing over tweets")
        
        geostart =  datetime.datetime.now() 
        print(geostart)
        df_js = geo_df(result_inf,geo) 
        name =table.replace("processed","geod")
        geoend = datetime.datetime.now()
        df_js.to_csv(os.path.join(tweet_dir,name), index = False)
        print ("geo ended at  {} for a total time of {} ".format(geoend, geoend-geostart))
        print ("pred and geo ended at  {} for a total time of {}.csv".format(geoend, geoend-predstart))

overallend = datetime.datetime.now() 
print('fin in {}'.format(overallend-overallstart))




