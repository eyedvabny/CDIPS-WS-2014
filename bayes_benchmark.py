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

data = pd.read_table('/Users/dillonniederhut/Desktop/avito_train.tsv',nrows=100000)
#replace with file path to your training data

features = pd.get_dummies(data.subcategory)
features_train, features_test, target_train, target_test =\
    train_test_split(features, data.is_blocked, test_size = 0.25)

bayes = MultinomialNB()
bayes.fit(features_train, target_train)
prediction_float = bayes.predict(features_test)
prediction = np.round(prediction_float)
print classification_report(target_test, prediction)
