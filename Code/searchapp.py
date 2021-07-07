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

from tensorflow.keras import layers
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

code_dir = os.getcwd()
print("Current working directory: {0}".format(code_dir))
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")
directory = os.path.join(data_dir,'splits')
dsa= os.path.dirname(parent_dir)
 
train_data = pd.read_csv(os.path.join(directory, 'InformativenessTrain_Processed.csv'))
print('cleaning')
def clean_text(text):
    '''Make text lowercase, remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    #get rid of usernames
    tweet_words = text.strip('\r').split(' ')
    for word in [word for word in tweet_words if '@' in word]:
            
            text = text.replace(word, "")
    #get rid of the re-tweet
    tweet_words = text.strip('\r').split(' ')
    for word in [word for word in tweet_words if 'rt' == word]:
            
            text = text.replace(word, "")
            
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Applying the cleaning function to both test and train datasets
train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))

train_data['text'] = train_data['text'].apply(lambda x:word_tokenize(x))
print('cleand and totkeninzed')
print('rv stop words')
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words 

train_data['text'] = train_data['text'].apply(lambda x : remove_stopwords(x))
print('rvd stop words')
print('lemmatime')
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]  ##Notice the use of text.

train_data['text'] = train_data['text'].apply(lambda x : lemmatize_text(x))

def concatenate_text(text):
    return ' '.join(text)

train_data['text'] = train_data['text'].apply(lambda x : concatenate_text(x))




def train_val_split(df, validation_split):
    """
    This function generates the training and validation splits from an input dataframe
    
    Parameters:
        dataframe: pandas dataframe with columns "text" and "target" (binary)
        validation_split: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split
    
    Returns:
        train_samples: list of strings in the training dataset
        val_samples: list of strings in the validation dataset
        train_labels: list of labels (0 or 1) in the training dataset
        val_labels: list of labels (0 or 1) in the validation dataset      
    """
       
    text = df['text'].values.tolist()                         # input text as list
    targets = df['class_label_cat'].values.tolist()                    # targets
    
#   Preparing the training/validation datasets
    
    seed = random.randint(1,50)   # random integer in a range (1, 50)
    rng = np.random.RandomState(seed)
    rng.shuffle(text)
    rng = np.random.RandomState(seed)
    rng.shuffle(targets)

    num_validation_samples = int(validation_split * len(text))

    train_samples = text[:-num_validation_samples]
    val_samples = text[-num_validation_samples:]
    train_labels = targets[:-num_validation_samples]
    val_labels = targets[-num_validation_samples:]
    
    print(f"Total size of the dataset: {df.shape[0]}.")
    print(f"Training dataset: {len(train_samples)}.")
    print(f"Validation dataset: {len(val_samples)}.")
    
    return train_samples, val_samples, train_labels, val_labels

train_samples, val_samples, train_labels, val_labels = train_val_split(train_data, 0.25)
print ('Text done')

print('Loading model and glove')
model = tf.keras.models.load_model(os.path.join(dsa,'model1'))
path_to_glove_file = os.path.join(dsa,'WordVector','glove.twitter.27B.200d.txt')
embeddings_index = {}
f = open(path_to_glove_file, 'r', encoding='utf8')
for line in f:
    splitLine = line.split(' ')
    word = splitLine[0]                                  # the first entry is the word
    coefs = np.asarray(splitLine[1:], dtype='float32')   # these are the vectors representing word embeddings
    embeddings_index[word] = coefs
print("Glove data loaded! In total:",len(embeddings_index)," words.")
print("Get embedd")

def make_embedding_matrix(train_samples, val_samples, embeddings_index):
    
    """
    This function computes the embedding matrix that will be used in the embedding layer
    
    Parameters:
        train_samples: list of strings in the training dataset
        val_samples: list of strings in the validation dataset
        embeddings_index: Python dictionary with word embeddings
    
    Returns:
        embedding_matrix: embedding matrix with the dimensions (num_tokens, embedding_dim), 
        where num_tokens is the vocabulary of the input data, 
        and emdebbing_dim is the number of components in the GloVe vectors (can be 50,100,200,300)
        vectorizer: TextVectorization layer      
    """
    
    vectorizer = TextVectorization(max_tokens=55000, output_sequence_length=50)
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
    vectorizer.adapt(text_ds)
    
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
      
    num_tokens = len(voc)
    
    hits = 0
    misses = 0

#   creating an embedding matrix
    embedding_dim = len(embeddings_index['the'])
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

#     print("Converted %d words (%d misses)" % (hits, misses))
    print(f"Converted {hits} words ({misses} misses).")

    return embedding_matrix, vectorizer

embedding_matrix, vectorizer = make_embedding_matrix(train_samples, val_samples, embeddings_index)
print("embedded")
print ("scrapeing ")
text_query = "heat OR fire OR forestfire OR earthquake OR hea OR heatwave OR disaster OR typhoon OR cyclone OR tornado OR thunder OR lightning OR storm OR surge OR hail OR torrent"
since_date = '2021-07-01'
until_date = '2021-07-05'

tweets_geo_df, tweets_no_geo_df  = twittsearch(text_query,since_date,until_date)
print ("scraped ")
print("processing ")
twts = tweets_geo_df
twts['ptext'] = twts['Text'].apply(lambda x: clean_text(x))
twts['ptext'] = twts['ptext'].apply(lambda x: word_tokenize(x))
twts['ptext'] = twts['ptext'].apply(lambda x : remove_stopwords(x))
twts['ptext'] = twts['ptext'].apply(lambda x : lemmatize_text(x))
twts['ptext'] = twts['ptext'].apply(lambda x : concatenate_text(x))
print("processed")
print("predict")
def suggest_nn2(df, model):
    """
    This function generates (binary) targets from a dataframe with column "text" using trained Keras model
    
    Parameters:
        df: pandas dataframe with column "text"
        model: Keras model (trained)
    
    Output:
        predictions: list of suggested targets corresponding to string entries from the column "text"
    """
    
    string_input = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    preds = model(x)
    end_to_end_model = keras.Model(string_input, preds)

    probabilities = end_to_end_model.predict(df["ptext"])
    
    predictions = [1 if i > 0.5 else 0 for i in probabilities]
    
    return predictions

predictions = suggest_nn2(twts, model)

submission_data = {"ID": twts['TweetId'].tolist(),"tweet": twts['Text'].tolist(), "target": predictions}

submission_df = pd.DataFrame(submission_data)
result = twts.join(submission_df.target)

result.to_csv(os.path.join(tweet_dir,'result.csv'),index =False)
print("predicted")

# =============================================================================
# tweets_geo_df = twittsearch(text_query,since_date,until_date)
# =============================================================================

tweets_geo_df.to_csv(os.path.join(tweet_dir,"tweets_geo.csv"))
tweets_no_geo_df.to_csv(os.path.join(data_dir,"tweets_no_geo.csv"))

print('fin')
# =============================================================================
# 
# tweet_map = pd.read_csv(os.path.join(data_dir,"tweets_geo.csv"))
# =============================================================================



