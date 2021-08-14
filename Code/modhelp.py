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


def geo_df(df,geo, field):
    '''
    This function calls the batch geo parsing method from Mordecai.
    Parameters: 
        df: a data frame containign data to be geo parse
        geo: the mordecai insstance
        field: The field with the data to be used for geoparsing
    Returns:
        Dataframe of geoparsed records
    '''

    df['geos'] = geo.batch_geoparse(df[field])
    df_geo = df[df["geos"].str.len() != 0]
    df_geo = df_geo.explode('geos')
    df_geo = pd.concat([df_geo.drop(['geos'], axis=1), df_geo['geos'].apply(pd.Series)], axis=1)
    df_geo = pd.concat([df_geo.drop(['geo'], axis=1), df_geo['geo'].apply(pd.Series)], axis=1)
    df_geo = df_geo[df_geo['lat'].notnull()]
    df_geo.lat = df_geo.lat.astype(float)
    df_geo.lon =df_geo.lon.astype(float)
    return df_geo




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



    