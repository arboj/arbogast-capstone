#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 08:29:53 2021

@author: Arbo
"""
import os
import pandas as pd
code_dir = os.getcwd()
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")
def append_df(tweet_dir):
    
    tweets_df = pd.DataFrame(
                         columns=['Datetime', 'TweetId', 'Text',])
    
    for table in os.listdir(tweet_dir):
        if table.startswith("tweets"):
            tempdf = pd.read_csv(os.path.join(tweet_dir,table), engine = 'python', )
            tempdf = tempdf[['Datetime', 'TweetId', 'Text']]
            tweets_df = tweets_df.append(tempdf)      
    tweets_df['Text'] = tweets_df['Text'].astype(str)        
    return tweets_df