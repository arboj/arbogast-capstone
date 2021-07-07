#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:25:53 2021

@author: Arbo
"""

from mordecai import Geoparser
from mordecai import batch_geoparse
from flatten_json import flatten_json
import numpy as np

## initialize the parser
geo = Geoparser()

geo.batch_geoparse(Tweets_df['Text'])





