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
from tensorflow import keras

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

def geo_df(tweets_df):
    
    geo = Geoparser()
    tweets_df['geo'] = geo.batch_geoparse(tweets_df['Text'])
    df_js = pd.DataFrame()
    for row in range(len(tweets_df)):
        df_temp = pd.json_normalize(
        tweets_df.geo[row], 
        record_path =['spans'], 
        meta=['word',"country_predicted", "country_conf",['geo',"admin1"],
              ['geo',"lat"],['geo',"lon"],['geo',"country_code3"],['geo',"geonameid"],
              ['geo',"place_name"],['geo',"feature_class"],['geo',"feature_code"]],
        errors='ignore'
    )
        df_temp['TweetId']=''
        for i in range(len(df_temp)):
            df_temp['TweetId'][i]=tweets_df['TweetId'][row]
        df_js=df_js.append(df_temp,ignore_index=True)
        
    df_js = df_js.rename(columns = {'TweetId':'TweetId', 'start':'start', 'end':'end', 
                                    'word':'word','country_predicted':'country_predicted', 
                                    'country_conf': 'country_conf','geo.admin1':'admin1', 
                                    'geo.lat':'lat', 'geo.lon':'lon', 
                                    'geo.country_code3':'country_code3','geo.geonameid':'geonameid', 
                                    'geo.place_name':'place_name', 
                                    'geo.feature_class':'feature_class','geo.feature_code':'feature_code'})
    return df_js

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


