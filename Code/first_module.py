#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 13:10:38 2021

@author: Arbo
"""

def main ():
    print ("First module's name: {}".format(__name__))

if __name__ == '__main__':
    main()
else:
    print( "run from import")
