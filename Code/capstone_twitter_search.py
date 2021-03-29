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
    import json
    from mordecai import Geoparser
    from flatten_json import flatten_json
    import numpy as np
    
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
    tweets_geo_list = []
    tweets_list2 = []
    
    
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        
        if i>500:
            break
        #### define locations column
        locations = geo.geoparse(tweet.content)
        print(type(locations))
        #### if the location colum has a value, for every parsed location copy the row data
        if len(locations) > 0:
            print(type(locations))
            for j in range(len(locations)):
                llist=[]
                location_json = flatten_json(locations[j])
                for x in location_json:
                    llist.append(location_json[x]) 
                tweet_list1 = [tweet.date, tweet.id, tweet.content]
                tweet_list1.extend(llist) 
                tweets_geo_list.append(tweet_list1)
                
                print(tweets_geo_list)

        else :
            print(0)
            tweets_list2.append([tweet.date, tweet.id, tweet.content])
        
        
    # Creating a dataframe from the tweets list above

        
    tweets_geo_df = pd.DataFrame(tweets_geo_list, columns=
                              ['Datetime', 'Tweet Id', 'Text', 
                               "FoundWord","start_text","end_text",
                               "country_predicted", "country_conf","admin1",
                               "lat","lon","country_code3","geonameid","place_name",
                               "feature_class","feature_code"])
    
    tweet_no_geo = pd.DataFrame(tweets_list2, columns=
                              ['Datetime', 'Tweet Id', 'Text'])

    return (tweets_geo_df,tweet_no_geo)


