#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:40:16 2021

@author: Arbo
"""

def twittsearch(text_query,since_date,until_date):
    import snscrape.modules.twitter as sntwitter
    import pandas as pd
    
    
    # =============================================================================
    # default variables used for testing
    # text_query = 'Texas'
    # since_date = '2021-02-01'
    # until_date = '2021-03-01'
    # =============================================================================
    # create query string
    
    query = '{0} since:{1} until:{2}'.format(text_query, since_date, until_date)
    # Creating list to append tweet data to
    
    tweets_list2 = []
    
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i>500:
            break
        tweets_list2.append([tweet.date, tweet.id, tweet.content])
        
        
    # Creating a dataframe from the tweets list above
    tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text'])
    return tweets_df2


