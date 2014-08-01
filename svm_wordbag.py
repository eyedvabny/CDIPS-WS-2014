#!/usr/bin/env python
"""
Linear SVM for Avito
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
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import os

def validate(input_train, rows=True, test=0.25):
    """
    Takes file as input and returns classification report, average precision, and
    AUC for a bigram model. By default, loads all rows of a dataset, trains on .75,
    and tests on .25. 
    ----
    input_train : 'full path of the file you are loading'
    rows : True - loads all rows; insert an int for specific number of rows
    test : float proportion of dataset used for testing
    """
    if rows == True:
        data = pd.read_table(input_train)
    else:
        data = pd.read_table(input_train, nrows = rows)
    response = data.is_blocked
    dummies = sparse.csc_matrix(pd.get_dummies(data.subcategory))
    words = np.array(data.description,str)
    del data
    vect = text.CountVectorizer(decode_error = u'ignore',strip_accents='unicode',ngram_range=(1,2))
    counts = vect.fit_transform(words)
    features = sparse.hstack((dummies,counts))
    features_train, features_test, target_train, target_test = train_test_split(features, response, test_size = test)
    clf = LinearSVC()
    clf.fit(features_train, target_train)
    prediction = clf.predict(features_test)
    return classification_report(target_test, prediction), average_precision_score(target_test, prediction), roc_auc_score(target_test, prediction)

def run(input_train, input_test, output_name):
    """
    Takes a file path as input, a file path as output, and produces a sorted csv of
    item IDs for Kaggle submission
    -------
    input_train : 'full path of the training file'
    input_test : 'full path of the testing file'
    output_name : 'full path of the output file'
    """

    data = pd.read_table(input_train)
    test = pd.read_table(input_test)
    testItemIds = test.itemid
    response = data.is_blocked
    dummies = sparse.csc_matrix(pd.get_dummies(data.subcategory))
    pretestdummies = pd.get_dummies(test.subcategory)
    testdummies = sparse.csc_matrix(pretestdummies.drop(['Растения', 'Товары для компьютера'],axis=1))
    words = np.array(data.description,str)
    testwords = np.array(test.description,str)
    del data, test
    vect = text.CountVectorizer(decode_error = u'ignore', strip_accents='unicode', ngram_range=(1,2))
    corpus = np.concatenate((words, testwords))
    vect.fit(corpus)
    counts = vect.transform(words)
    features = sparse.hstack((dummies,counts))
    clf = LinearSVC()
    clf.fit(features, response)
    testcounts = vect.transform(testwords)
    testFeatures = sparse.hstack((testdummies,testcounts))
    predicted_scores = clf.predict_proba(testFeatures).T[1]
    f = open(output_name,'w')
    f.write("id\n") 
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
        f.write("%d\n" % (item_id))
    f.close()

