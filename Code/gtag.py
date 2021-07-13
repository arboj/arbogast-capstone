#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:25:53 2021

@author: Arbo
"""
from mordecai import Geoparser
geo = Geoparser()
tweets_df['geo'] = geo.batch_geoparse(tweets_df['Text'])


df['geo2'] = df['geo'].apply(lambda x: eval(x.replace(": nan,", ": 'nan',")))
def geo_df(tweets_df):
    from flatten_json import flatten_json
    tweets_geo_list=[]
    if len(tweets_df.geo)>0:
        for j in range (len(tweets_df.geo)):
           llist=[] 
           location_json = flatten_json(tweets_df.geo[j])
           for x in location_json:
               llist.append(location_json[x])
           tweet_list1 = [tweets_df.Datetime, tweets_df.TweetId, tweets_df.Text]
           tweet_list1.extend(llist)
           tweets_geo_list.append(tweet_list1)



pd.DataFrame(tweets_geo_list, columns=
                              ['Datetime', 'TweetId', 'Text', 
                               "FoundWord","start_text","end_text",
                               "country_predicted", "country_conf","admin1",
                               "lat","lon","country_code3","geonameid","place_name",
                               "feature_class","feature_code"])


df_js = pd.DataFrame()
for row in range(len(tweets_df)):
    print(row)
    df_temp = pd.json_normalize(
    tweets_df.geo[row], 
    record_path =['spans'], 
    meta=['word',"country_predicted", "country_conf",['geo',"admin1"],
                               ['geo',"lat"],['geo',"lon"],['geo',"country_code3"],['geo',"geonameid"],['geo',"place_name"],
                               ['geo',"feature_class"],['geo',"feature_code"]],
    errors='ignore'
)
    df_temp['TweetId']=''
    print(tweets_df['TweetId'][row])
    for i in range(len(df_temp)):
        df_temp['TweetId'][i]=tweets_df['TweetId'][row]
    df_js=df_js.append(df_temp,ignore_index=True)
        
result = pd.merge(tweets_df, df_js, on="TweetId")