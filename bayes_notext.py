#!/usr/bin/env python
"""
Naive Bayes classifier for Avito
"""
__author__ = "deniederhut"
__license__ = "GPL"
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
load FeatureEngineering/processFeatures_positive

data = pd.read_table('/Users/dillonniederhut/Desktop/avito_train.tsv',nrows=100000)
#replace with file path to your training data

features = pd.concat([pd.get_dummies(data.subcategory), pd.DataFrame({'urlbin' : url_cnt(data.urls_cnt), 'emailbin' : emails_cnt(data.emails_cnt), 'phonebin' : phone_cnt(data.phones_cnt), 'pricelog' : price(data.price)})], axis = 1)
features_train, features_test, target_train, target_test = train_test_split(features, data.is_blocked, test_size = 0.25)
del data, features

bayes = MultinomialNB()
bayes.fit(features_train, target_train)
prediction = np.round(bayes.predict(features_test))
print classification_report(target_test, prediction)
