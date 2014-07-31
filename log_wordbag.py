#!/usr/bin/env python
"""
Logistic regression for Avito
Takes test data as an input and returns classification report, average 
precison, and area under curve.
Includes tokenized word counts and data in sparse format
For the whole dataset, memory use peaks at about 5Gb
"""
__author__ = "deniederhut"
__license__ = "GPL"
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import text
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split

data = pd.read_table('/media/dillon/dinsfire/avito_train.tsv', nrows = 1000)
test = pd.read_table('/media/dillon/dinsfire/avito_test.tsv', nrows = 1000)
#replace with file path to your training data

response = data.is_blocked
dummies = sparse.csc_matrix(pd.get_dummies(data.subcategory))
pretestdummies = pd.get_dummies(test.subcategory)
testdummies = sparse.csc_matrix(efg.drop(['Растения', 'Товары для компьютера'],axis=1))
vect = text.CountVectorizer(decode_error = u'ignore')
corpus = np.concatenate((np.array(data.description,str), np.array(test.description,str)))
del data, test
vect.fit(corpus)
counts = vect.transform(np.array(data.description,str))
testcounts = vect.transform(np.array(test.description,str))
features = sparse.hstack((dummies,counts))
features_train, features_test, target_train, target_test = train_test_split(features, response, test_size = 0.25)

clf = linear_model.LogisticRegression()
clf.fit(features_train, target_train)
prediction = clf.predict(features_test)
print classification_report(target_test, prediction)
print average_precision_score(target_test, prediction)
print roc_auc_score(target_test, prediction)

#stop here if you do not want to create a kaggle file

clf.fit(features, response)
corpus = np.array(test.description,str)
del test
counts = vect.transform(corpus)
testFeatures = sparse.hstack((testdummies,testcounts))
predicted_scores = clf.predict_proba(testFeatures).T[1]
##predicted_scores needs to be formatted and written to .csv





