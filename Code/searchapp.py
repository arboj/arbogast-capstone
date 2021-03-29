#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:34:55 2021

@author: Arbo

"""


from capstone_twitter_search import twittsearch
import os
import pandas

code_dir = os.getcwd()
print("Current working directory: {0}".format(code_dir))
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")

searchterms = ['snow','weather','power','freeze', 'ice', 'blackout','water']
text_query = "snow OR weather OR power OR freeze OR ice OR blackout OR water'"


since_date = '2021-02-12'
until_date = '2021-02-20'

# =============================================================================
# d = {i: twittsearch(i,since_date,until_date) for i in searchtearms}
# =============================================================================


# =============================================================================
# for text_query in searchterms:
#     #Dynamically create Data frames
#     vars()[text_query] = twittsearch(text_query,since_date,until_date)
# =============================================================================

tweets_geo_df = twittsearch(text_query,since_date,until_date)[0]
tweets_no_geo_df = twittsearch(text_query,since_date,until_date)[0]

tweets_geo_df.to_csv(os.path.join(data_dir,"tweets_geo.csv"))
tweets_no_geo_df.to_csv(os.path.join(data_dir,"tweets_no_geo.csv"))