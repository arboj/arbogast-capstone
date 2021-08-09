#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:40:16 2021

@author: Arbo
"""
import os
import pandas as pd
import snscrape.modules.twitter as sntwitter
code_dir = os.getcwd()
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")


def twittsearch(text_query,since_date,until_date,tweetcount):
    
    tweets_list = []

    query = '{0} since:{1} until:{2} filter:has_engagement'.format(text_query, since_date, until_date)
    print(query)

    # Using TwitterSearchScraper to scrape data and append tweets to list

    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):

        if i>tweetcount:

            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.url,
                            tweet.lang,tweet.retweetedTweet,tweet.quotedTweet])

    # Creating a dataframe from the tweets list above

    tweets_df = pd.DataFrame(tweets_list, 
                             columns=['Datetime', 'TweetId', 'Text','TweetLink',
                                     "Language", "RTFrom", "QTFrom"])
    
    print("found {} tweets ranging from {} to {}".format(len(tweets_df),
                                                            tweets_df.Datetime.min(),tweets_df.Datetime.max()))
    
    print("dropping duplicates")   
    tweets_df = tweets_df.drop_duplicates(subset=['TweetId'])
    print("total of tweets now: {}".format(len(tweets_df)))
    print("english only")

    tweets_df = tweets_df[tweets_df["Language"]=='en']
    print("total of tweets now: {}".format(len(tweets_df)))
    tweets_df = tweets_df[tweets_df["RTFrom"].isna()]
    print("total of tweets now: {}".format(len(tweets_df)))
    tweets_df = tweets_df[tweets_df["QTFrom"].isna()]
    print("total of tweets now: {} ranging from {} to {}".format(len(tweets_df),
                                                                tweets_df.Datetime.min(),tweets_df.Datetime.max()))
    
    tweets_df = tweets_df[['Datetime', 'TweetId', 'Text','TweetLink']]
    return tweets_df

# =============================================================================
# text_query = "heat OR fire OR forestfire OR earthquake OR heat OR heatwave OR disaster OR typhoon OR cyclone OR tornado OR thunder OR lightning OR storm OR surge OR hail OR torrent OR flood OR deluge"
# since_date = '2021-07-07'
# until_date = '2021-07-13'
# 
# tweets_df = twittsearch(text_query,since_date,until_date)
# =============================================================================


