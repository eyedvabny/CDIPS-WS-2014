﻿# coding: utf-8
"""
Benchmarks for the Avito fraud detection competition


18.07.2014: Modifications by Daniel:
    Adding comments to make the script more understandable
    Comment out commands to terminate execution after generating train features
    
    Adding function to test the predictive power of features. I am not sure if that how it should work

"""
import csv
import re
import nltk.corpus
from collections import defaultdict
import scipy.sparse as sp
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from nltk import SnowballStemmer
import random as rnd 
import logging
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

# The location of the datafolder has to be adjusted:
dataFolder = "~/Documents/Programming/DSWorshop/Illicit_content"

# Loading russinan stopwords (words that will not be involved in the analysis)
stopwords= frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian") if word!="не")    

# Get the stems of the words (by default stemming is not used -> getWords is called with 'stemmRequired = False')
stemmer = SnowballStemmer('russian')
 
# These lines are for fixing words where the cyrillic alphabet is mixed with the latin
 # To trick simple algorithms
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

# Seting up logging events during the run.
# Will not work from IDE
# I have added an output filename for future diagnostics
logging.basicConfig(filename='test.log', format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level=logging.DEBUG)

# Import and function that fixes the words to make them esier to recognise
def correctWord (w):
    """ Corrects word by replacing characters with written similarly depending on which language the word. 
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""

    if len(re.findall(ur"[а-я]",w))>len(re.findall(ur"[a-z]",w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)



def getItems(fileName, itemsLimit=None):
    """ 
    This function reads data file.
    Return with an iterable dictionary with the field names as keys
    Return all fields of the advertisement, there is no selection at this point.
    
    itemsLimit: this parameter tells how many items (advertisements) the function should return.
    
    Regardless of the itemsLimit value, the function reads all lines, and randomly select from them
    
    """
    
    with open(os.path.join(dataFolder, fileName)) as items_fd:
        logging.info("Sampling...")

        # The first read (if we need only a sample from the input) is 
        # just reading how many advertisement are there       
        if itemsLimit:
            countReader = csv.DictReader(items_fd, delimiter='\t', quotechar='"')
            numItems = 0
            for row in countReader:
                numItems += 1
            items_fd.seek(0)        
            rnd.seed(0)
            
            # Geting random numbers from the range of the total lines
            sampleIndexes = set(rnd.sample(range(numItems),itemsLimit))
            
        logging.info("Sampling done. Reading data...")
        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        itemNum = 0
        
        for i, item in enumerate(itemReader):
            
            item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
            if not itemsLimit or i in sampleIndexes:
                itemNum += 1
                
                # Yield is returning an generator object
                # That can be iterated only once!
                # Saves memory and cpu
                # http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
                yield itemNum, item
                
    
def getWords(text, stemmRequired = False, correctWordRequired = False):
    """ Splits the text into words, discards stop words and applies stemmer if required. 
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required     
    """
    # making all words lowercase
    # removing all punctuations
    cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())

    # Correcting words if required (no by default) -> stemming if required
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(correctWord(w)) for w in cleanText.split() if len(w)>1 and w not in stopwords]

    # Don't applying corrections. -> stemming if required
    else:
        words = [w if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    
    return words

    
def processData(fileName, featureIndexes={}, itemsLimit=None):
    """ Processing data. """
    processMessage = ("Generate features for " if featureIndexes else "Generate features dict from ")+os.path.basename(fileName)
    logging.info(processMessage+"...")
    print processMessage

    wordCounts = defaultdict(lambda: 0)
    targets = []
    item_ids = []
    row = []
    col = []
    cur_row = 0
    
    # REturning dict with the feature names and values
    # Each item is a line from the tsv file.
    # For feature details see kaggle documentation    
    for processedCnt, item in getItems(fileName, itemsLimit):
        
        # Getting words from the title and the descritption
        # The 'words' are returned from the getwords function!!!
        for word in getWords(item["title"]+" "+item["description"], 
                             stemmRequired = False, correctWordRequired = False):

            # Only true in the first calling when there is no feature index
            if not featureIndexes:

                wordCounts[word] += 1 # Counting words
            
            # At the second calling when there is feature index submitted:
            else:
                
                if word in featureIndexes:
                    col.append(featureIndexes[word])
                    row.append(cur_row)
        # Not the best way to
        if featureIndexes:
            cur_row += 1
            if "is_blocked" in item: # Not necessarely has all row this field set... that's why we are selecting. OK it's always set.
                targets.append(int(item["is_blocked"]))

            item_ids.append(int(item["itemid"]))
                    
        if processedCnt%1000 == 0:                 
            logging.debug(processMessage+": "+str(processedCnt)+" items done")
    
    # 
    if not featureIndexes:
        index = 0
        for word, count in wordCounts.iteritems():
            if count >= 3:
                
                # Comment out this line to find out what the loop is doing:                
                #print word.encode("utf-8")+" "+str(count)+" "+str(index)
                featureIndexes[word]=index
                index += 1
                
        return featureIndexes

    # if the featureIndexes is provided, a matrix is generated.    
    else:
        # The trickiest line in the script$$$$ 
        features = sp.csr_matrix((np.ones(len(row)),(row,col)), shape=(cur_row, len(featureIndexes)), dtype=np.float64)
        if targets:
            return features, targets, item_ids
        else:
            return features, item_ids




def main():
    """ Generates features and fits classifier. """
    
    # The following 5 command lines can be outcommented if the features are already created.
    # There is no need to process the data every single time.
    # Fine tuning the learning algorythm is much faster without that extra step.
    
    # by reading the train dataset the feature index is created.
    # First calling of the processdata function
    # originally the items are limited to 300000
    featureIndexes = processData(os.path.join(dataFolder,"avito_train_small.tsv"), itemsLimit=5000) # Original itemsLimit=300000

#    # Trainfeature is created using the indexfeatures...
    # Second calling of the processdata function
    trainFeatures,trainTargets, trainItemIds=processData(os.path.join(dataFolder,"avito_train_small.tsv"), featureIndexes, itemsLimit=5000) # Original itemsLimit=300000
#
#    # Building the test dataset... just like the training...
    testFeatures, testItemIds=processData(os.path.join(dataFolder,"avito_test.tsv"), featureIndexes)
#
#    # Dumping data into file...
#    joblib.dump((trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), os.path.join(dataFolder,"train_data.pkl"))
    joblib.dump((trainFeatures, trainTargets, trainItemIds), os.path.join(dataFolder,"train_data_small.pkl"))

#
#    # loading data pack...
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(dataFolder,"train_data.pkl"))
#
    logging.info("Feature preparation done, fitting model...")
#
#    # Stochastic gradient model
    clf = SGDClassifier(    loss="log", 
                            penalty="l2", 
                            alpha=1e-4, 
                            class_weight="auto")
    #
    clf.fit(trainFeatures,trainTargets)
#
    logging.info("Predicting...")
#
#    #     
    predicted_scores = clf.predict_proba(testFeatures).T[1]
#
#    
    logging.info("Write results...")
#    #    
    output_file = "avito_starter_solution.csv"
    logging.info("Writing submission to %s" % output_file)
    f = open(os.path.join(dataFolder,output_file), "w")
    f.write("id\n")
    
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
        f.write("%d\n" % (item_id))
    f.close()
    logging.info("Done.")
                               
if __name__=="__main__":            
    main()            
    

    
        
        
        
        
        
        
        