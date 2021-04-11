#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:38:49 2021

@author: Arbo
"""
import os
import pandas as pd

code_dir = os.getcwd()
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")

tweets_geo = pd.read_csv(os.path.join(tweet_dir,"tweets_geo.csv"))
tweets_geo = tweets_geo['TweetId','text']
tweets_geo.to_csv(os.path.join(data_dir,"tweets_geo_txt_id.csv"))