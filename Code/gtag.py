#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:25:53 2021

@author: Arbo
"""
def geotag (text):
    import pandas as pd
    import json
    from mordecai import Geoparser
    from flatten_json import flatten_json
    import numpy as np
    
    ## initialize the parser
    geo = Geoparser()
    parse = []
    # create query string
    # Creating list to append tweet data to
    tweets_geo_list = []
    tweets_list2 = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    locations = geo.geoparse(text)
    
    return locations

