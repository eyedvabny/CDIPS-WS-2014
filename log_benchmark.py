#!/usr/bin/env python
"""
Logistic regression for Avito
"""
__author__ = "deniederhut"
__license__ = "GPL"
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

data = pd.read_table('/media/dillon/dinsfire/avito_train.tsv')
features = pd.concat([pd.get_dummies(data.category), pd.get_dummies\
    (data.subcategory)],axis=1)
features_train, features_test, target_train, target_test =\
    train_test_split(features, data.is_blocked, test_size = 0.25)

log = linear_model.LogisticRegression()
log.fit(features_train, target_train)
prediction = log.predict(features_test)
print classification_report(target_test, prediction)


