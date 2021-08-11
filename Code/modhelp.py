#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:39:38 2021

@author: Arbo
"""
from mordecai import Geoparser
import numpy as np
import pandas as pd 
import random
import tensorflow as tf            
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping

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

def test_listerine(df):
    """
    This function generates the test x and y from an input dataframe
    
    Parameters:
        dataframe: pandas dataframe with columns "text" and "class_label_cat" (binary)
        
    
    Returns:
        test_samples: list of strings in the training dataset

        test_labels: list of labels (0 or 1) in the training dataset
    
    """
       
    text = df['text'].values.tolist()                         # input text as list
    targets = df['class_label_cat'].values.tolist()                    # targets
    
#   Preparing the training/validation datasets
    
    seed = random.randint(1,50)   # random integer in a range (1, 50)
    rng = np.random.RandomState(seed)
    rng.shuffle(text)
    rng = np.random.RandomState(seed)
    rng.shuffle(targets)

   

    test_samples = text
   
    test_labels = targets
    
    
    print(f"Total size of the dataset: {df.shape[0]}.")

    
    return test_samples, test_labels


def geo_df(df,geo):
    
# =============================================================================
#     geo = Geoparser()
# =============================================================================
    df['geos'] = geo.batch_geoparse(df['Text'])
    df_geo = df[df["geos"].str.len() != 0]
    df_geo = df_geo.explode('geos')
    df_geo = pd.concat([df_geo.drop(['geos'], axis=1), df_geo['geos'].apply(pd.Series)], axis=1)
    df_geo = pd.concat([df_geo.drop(['geo'], axis=1), df_geo['geo'].apply(pd.Series)], axis=1)
    df_geo = df_geo[df_geo['lat'].notnull()]
    df_geo.lat = df_geo.lat.astype(float)
    df_geo.lon =df_geo.lon.astype(float)
    return df_geo

#     df_js = pd.DataFrame()
#     for row in range(len(result_inf)):
#         df_temp = pd.json_normalize(result_inf['geos'], record_path =['spans'], 
#         meta=['word',"country_predicted", "country_conf",['geo',"admin1"],
#               ['geo',"lat"],['geo',"lon"],['geo',"country_code3"],['geo',"geonameid"],
#               ['geo',"place_name"],['geo',"feature_class"],['geo',"feature_code"]],
#         errors='ignore'
#     )
#         df_temp['TweetId']=''
#         for i in range(len(df_temp)):
#             df_temp['TweetId'][i]=tweets_df['TweetId'][row]
#         df_js=df_js.append(df_temp,ignore_index=True)

#     df_js = df_js.rename(columns = {'TweetId':'TweetId', 'start':'start', 'end':'end', 
#                                     'word':'word','country_predicted':'country_predicted', 
#                                     'country_conf': 'country_conf','geo.admin1':'admin1', 
#                                     'geo.lat':'lat', 'geo.lon':'lon', 
#                                     'geo.country_code3':'country_code3','geo.geonameid':'geonameid', 
#                                     'geo.place_name':'place_name', 
#                                     'geo.feature_class':'feature_class','geo.feature_code':'feature_code'})
#     return df_js




def suggest_nn2(df, model, vectorizer):
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

def suggest_nn3(df, model, vectorizer):
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

    probabilities = end_to_end_model.predict(df["text"])
    
    predictions = [1 if i > 0.5 else 0 for i in probabilities]
    
    return predictions

def initialize_nn(embedding_matrix):
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
# =============================================================================
#     x = layers.Bidirectional(layers.LSTM(32,
#                                           dropout=.5))(x)
# =============================================================================
    x = layers.Dense(128)(x)
    x = layers.Dropout(.5)(x)
    preds = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(int_sequences_input, preds)
    
    return model

def train_nn(model, train_samples, val_samples, train_labels, val_labels, vectorizer, stop = True):
    """
    This function fits the training data using validation data to calculate metrics.
    
    Parameters:
        model: preinitialized Keras model
        train_samples: list of strings in the training dataset
        val_samples: list of strings in the validation dataset
        train_labels: list of labels (0 or 1) in the training dataset
        val_labels: list of labels (0 or 1) in the validation dataset
        vectorizer: TextVectorization layer
        stop (Boolean): flag for Early Stopping (aborting training when a monitored metric has stopped improving)
    
    Returns:
        model: trained Keras model
        history: callback that can be used to track the learning process
    """
    
    print('')
    print("Training the model...")
    
    model.compile(loss="binary_crossentropy", 
              optimizer="adam", 
              metrics=["binary_accuracy"])
    
    x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
    x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()
    
    y_train = np.asarray(train_labels).astype('float32').reshape((-1,1))
    y_val = np.asarray(val_labels).astype('float32').reshape((-1,1))

    
    
    if stop:
        early_stopping = EarlyStopping(monitor='val_loss', patience=1)
        history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=1)
    else:
        history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), verbose=1)
        
    return model, history

    