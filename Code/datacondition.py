#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:02:41 2021

@author: Arbo
"""
import os 
import pandas as pd
import nltk
import sklearn
import keras
import collections as col
import pprint
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pprint
from sklearn.datasets import load_files
# =============================================================================
# from sklearn.cross_validation import train_test_split
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

import string
import nltk
from nltk.stem.porter import PorterStemmer


code_dir = os.getcwd()
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")
labled_data_folder= '/Volumes/Elements/DataScience/cb/crisis_datasets_benchmarks/data/all_data_en'

inf_train =  pd.read_table(os.path.join(labled_data_folder,"crisis_consolidated_informativeness_filtered_lang_en_train.tsv"))
inf_test  =  pd.read_table(os.path.join(labled_data_folder,"crisis_consolidated_informativeness_filtered_lang_en_train.tsv"))
tf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

train_counts = tf.fit_transform(inf_train['text'])
test_data = tf.transform(inf_test['text'])
nb = MultinomialNB()
nb = nb.fit(train_counts, inf_train['class_label'])
predicted = nb.predict(test_data)

print("Prediction accuracy = {0:5.1f}%".format(100.0 * nb.score(test_data, inf_test['class_label'])))
print('Number of Features = {}'.format(nb.feature_log_prob_.shape[1]))

num_k = 10000
ch2 = SelectKBest(chi2, k=num_k)
xtr = ch2.fit_transform(train_counts, inf_train['class_label'])
xt = ch2.transform(test_data)


nb = nb.fit(xtr, inf_train['class_label'])
predicted = nb.predict(xt)

print("NB prediction accuracy = {0:5.1f}%".format(100.0 * nb.score(xt, inf_test['class_label'])))
print('Number of Features = {}'.format(nb.feature_log_prob_.shape[1]))

feature_names = tf.get_feature_names()

indices = ch2.get_support(indices=True)
feature_names = np.array([feature_names[idx] for idx in indices])

pp = pprint.PrettyPrinter(indent=2, depth=1, width=80, compact=True)

top_count = 5

for idx, target in enumerate(inf_train['class_label']):
    top_names = np.argsort(nb.coef_[idx])[-top_count:]
    tn_lst = [name for name in feature_names[top_names]]
    tn_lst.reverse()

    print('\n{0}:'.format(target))
    pp.pprint(tn_lst)
    
