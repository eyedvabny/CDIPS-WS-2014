#!/usr/bin/env python
"""
Linear SVM for Avito
Includes tokenized word counts
"""
__author__ = "deniederhut"
__license__ = "GPL"
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

data = pd.read_table('/media/dillon/dinsfire/avito_train.tsv',nrows=10000)
#replace with file path to your training data

response = data.is_blocked
dummies = sparse.csc_matrix(pd.get_dummies(data.subcategory))
vect = text.CountVectorizer(decode_error = u'ignore')
corpus = np.array(data.description,str)
del data
counts = vect.fit_transform(corpus)
features = sparse.hstack((dummies,counts))
features_train, features_test, target_train, target_test = train_test_split(features, response, test_size = 0.25)

clf = LinearSVC()
clf.fit(features_train, target_train)
prediction = np.round(svm.predict(features_test))
print classification_report(target_test, prediction)
print average_precision_score(target_test, prediction)
print roc_auc_score(target_test, prediction)

#stop here if you do not want to create a kaggle file

test = pd.read_table('/Users/dillonniederhut/Desktop/avito_test.tsv')
dummies = sparse.csc_matrix(pd.get_dummies(test.subcategory))
vect = text.CountVectorizer(decode_error = u'ignore')
corpus = np.array(test.description,str)
del test
counts = vect.fit_transform(corpus)
testFeatures = sparse.hstack((dummies,counts))
%run kaggle.py

