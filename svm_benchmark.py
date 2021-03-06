#!/usr/bin/env python
"""
Linear SVM for Avito
"""
__author__ = "deniederhut"
__license__ = "GPL"
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

data = pd.read_table('/Users/dillonniederhut/Desktop/avito_train.tsv',nrows=100000)
#replace with file path to your training data

features = pd.get_dummies(data.subcategory)
features_train, features_test, target_train, target_test =\
    train_test_split(features, data.is_blocked, test_size = 0.25)

svm = LinearSVC()
svm.fit(features_train, target_train)
prediction = np.round(svm.predict(features_test))
print classification_report(target_test, prediction)

coef = pd.DataFrame(svm.coef_)
out = pd.concat([pd.DataFrame({'names' : features.columns}),np.abs(coef.T)],axis=1)
out = out[out[[2]] > 0.612]
