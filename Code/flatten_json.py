#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:19:50 2021

@author: Arbo
"""
# out{} is a empty dictionary. 
def flatten_json(input_doc):
    out = {}

# flatten() is called for every key value pair and checked for its type either dictionatry or list.  

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
#                 print(a,name)
                flatten(a, name + str(i) + '_')
                i += 1
        else:
#             print(name)
            out[name[:-1]] = x

    flatten(input_doc)
    return out