#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:40:16 2021

@author: Arbo
"""
import os
import pandas as pd

code_dir = os.getcwd()
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")


def twittsearch(text_query,since_date,until_date):
    print('Import modules')
    import snscrape.modules.twitter as sntwitter
    import pandas as pd
# =============================================================================
#     import json
#     from mordecai import Geoparser
#     from flatten_json import flatten_json
# =============================================================================
    import numpy as np
    tweets_list = []
    # create query string
    query = '{0} since:{1} until:{2}'.format(text_query, since_date, until_date)
    print(query)

    # Using TwitterSearchScraper to scrape data and append tweets to list
    
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i>1000000:
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content])

        
    # Creating a dataframe from the tweets list above


    
    tweets_df = pd.DataFrame(tweets_list, 
                             columns=['Datetime', 'TweetId', 'Text'])
    return tweets_df

# =============================================================================
# text_query = "heat OR fire OR forestfire OR earthquake OR heat OR heatwave OR disaster OR typhoon OR cyclone OR tornado OR thunder OR lightning OR storm OR surge OR hail OR torrent OR flood OR deluge"
# since_date = '2021-07-07'
# until_date = '2021-07-13'
# 
# tweets_df = twittsearch(text_query,since_date,until_date)
# =============================================================================


