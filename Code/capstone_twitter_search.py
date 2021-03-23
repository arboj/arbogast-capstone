#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:40:16 2021

@author: Arbo
"""

def twittsearch(text_query,since_date,until_date):
    print('Import modules')
    import snscrape.modules.twitter as sntwitter
    import pandas as pd
    from mordecai import Geoparser
    from flatten_json import flatten_json
    
    ## initialize the parser
    print("initalizing parser")
    geo = Geoparser()
    parse = []
    
    
    # =============================================================================
    # default variables used for testing
    # text_query = 'Texas'
    # since_date = '2021-02-01'
    # until_date = '2021-03-01'
    # =============================================================================
    # create query string
    
    query = '{0} since:{1} until:{2}'.format(text_query, since_date, until_date)
    # Creating list to append tweet data to
    print(query)
    tweets_list2 = []
    
    
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        
        if i>500:
            break
        #### define locations column
        locations = geo.geoparse(tweet.content)
        #### if the location colum has a value, for every parsed location copy the row data
        if len(locations) > 0:
            for j in range(len(locations)):
                print(locations[j])
                tweets_list2.append([tweet.date, tweet.id, tweet.content, locations[j]])
                
            
        else :
            print(0)
            tweets_list2.append([tweet.date, tweet.id, tweet.content, '0'])
        
        
    # Creating a dataframe from the tweets list above
    tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Location'])
# =============================================================================
#     
#     for i in range(len(tweets_df2)):
#         parse.append(flatten_json(geo.geoparse(tweets_df2['Text'][i])))
#         print (i)
#     tweets_df2['parse'] = parse
# =============================================================================
    return tweets_df2


