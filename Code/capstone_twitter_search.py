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

text_query = "snow OR weather OR power OR freeze OR ice OR blackout OR water OR rain OR sleet OR hail OR storm OR blizzard"
since_date = '2021-02-12'
until_date = '2021-02-20'

def twittsearch(text_query,since_date,until_date):
    print('Import modules')
    import snscrape.modules.twitter as sntwitter
    import pandas as pd
    import json
    from mordecai import Geoparser
    from flatten_json import flatten_json
    import numpy as np
    
    ## initialize the parser
    geo = Geoparser()
    parse = []
    # create query string
    query = '{0} since:{1} until:{2}'.format(text_query, since_date, until_date)
    # Creating list to append tweet data to
    tweets_geo_list = []
    tweets_list2 = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i>10000:
            break
        #### define locations column
        locations = geo.geoparse(tweet.content)
        #### if the location colum has a value, for every parsed location copy the row data
        if len(locations) > 0:
            for j in range(len(locations)):
                llist=[]
                location_json = flatten_json(locations[j])
                for x in location_json:
                    llist.append(location_json[x]) 
                tweet_list1 = [tweet.date, tweet.id, tweet.content]
                tweet_list1.extend(llist) 
                tweets_geo_list.append(tweet_list1)
                
                #print("Row {0} Has {1} Location(s)".format(i,len(locations)))

        else :
            #print("Row {0} Has no Locations".format(i))
            tweets_list2.append([tweet.date, tweet.id, tweet.content])
    print("Processed {0} tweets there are {1} locations and {2} tweets without".format(i,len(tweets_geo_list),len(tweets_list2)))
        
    # Creating a dataframe from the tweets list above

        
    tweets_geo_df = pd.DataFrame(tweets_geo_list, columns=
                              ['Datetime', 'TweetId', 'Text', 
                               "FoundWord","start_text","end_text",
                               "country_predicted", "country_conf","admin1",
                               "lat","lon","country_code3","geonameid","place_name",
                               "feature_class","feature_code"])
    
    tweet_no_geo = pd.DataFrame(tweets_list2, columns=
                              ['Datetime', 'TweetId', 'Text'])

    tweets_geo_df.to_csv(os.path.join(tweet_dir,"tweets_geo.csv",index_label = 'index'))
    tweets_no_geo_df.to_csv(os.path.join(data_dir,"tweets_no_geo.csv",index_label = 'index'))


