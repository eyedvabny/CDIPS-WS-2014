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

data = pd.read_table('/media/dillon/dinsfire/avito_train.tsv',nrows=1000)
#replace with file path to your training data

dummies = sparse.csc_matrix(pd.get_dummies(data.subcategory))
del data
vect = text.CountVectorizer(decode_error = u'ignore')
corpus = np.array(data.description,str)
counts = vect.fit_transform(corpus)
features = sparse.hstack(dummies,counts))
features_train, features_test, target_train, target_test = train_test_split(features, data.is_blocked, test_size = 0.25)

log = linear_model.LogisticRegression()
log.fit(features_train, target_train)
prediction = log.predict(features_test)
print classification_report(target_test, prediction)
print average_precision_score(target_test, prediction)
print roc_auc_score(target_test, prediction)


