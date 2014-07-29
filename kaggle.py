#!/usr/bin/env python
"""
Trains fit on whole feature set, 'features' and response variable, 'response',
and returns data file correctly formatted for Kaggle submission. Assumes
you have already run one of the machines, and have it stored in memory as
>> clf = machine(parameters)
and that the test features are preprocessed and stored in memory as 'testFeatures'
"""
__author__ = "deniederhut"
__license__ = "GPL"

import os

assert 'features' in locals()
assert 'response' in locals()
assert 'clf' in locals()
assert 'testFeatures' in locals()

clf.fit(features, response)
predicted_scores = clf.predict_proba(testFeatures).T[1]
f = open(os.path.join(outputFolder,output_file), "w")
    f.write("id\n")
    
for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
    f.write("%d\n" % (item_id))
    f.close()
    logging.info("Done.")
