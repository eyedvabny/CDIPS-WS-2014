# coding: utf-8
"""
Benchmarks for the Avito fraud detection competition


18.07.2014: Modifications by Daniel:
    Adding comments to make the script more understandable
    Comment out commands to terminate execution after generating train features
    
    Adding function to test the predictive power of features. I am not sure if that how it should work

30.07.2014:  Modifications by Daniel:
    The script groups the training and test features by category
    Tre resulting variables are dictionaries where each key is a category name and the value is a list or matrix
    The process stops after dumping the variables
    
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
import processFeatures as pF

# The location of the datafolder has to be adjusted:
dataFolder = "/Users/daniel/Documents/Programming/DSWorshop/Illicit_content"

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
    
    with open(fileName) as items_fd:
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
    # m             aking all words lowercase
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
    
    # Setting up log messages:
    processMessage = ("Generate features for " if featureIndexes else "Generate features dict from ")+os.path.basename(fileName)
    logging.info(processMessage+"...")

    # all variables are modified to deal with the separation by categories:
    wordCounts = {}
    targets = {}
    item_ids = {}
    row = {}
    col = {}
    cur_row = {}
    
    # variables storing the other fields:
    prices = {}
    urls   = {}
    phones = {}
    emails = {}
    length = {}
    
    # The variable with the text data:
    features = {}
    
    # Returning dict with the feature names and values
    # Each item is a line from the tsv file.
    # For feature details see kaggle documentation    
    for processedCnt, item in getItems(fileName, itemsLimit):

        # building dictionaries:
        # testing just once, and then initialize all the dictionaries
        category = item["category"]        
        
        try:
            prices[category]
        
        except:
            logging.info("A new category has been found: "+category)
            print "A new category has been found: "+category
            wordCounts[category] = defaultdict(lambda: 0)            
            prices[category] = []
            urls[category] = []
            phones[category] = []
            emails[category] = []
            length[category] = []
            col[category] = []
            row[category] = []
            item_ids[category] = []
            targets[category] = []
            cur_row[category] = 0
            features[category] = {}
            
        wordcnt = 0
        # Getting words from the title and the descritption
        # The 'words' are returned from the getwords function!!!
        for word  in getWords(item["title"]+" "+item["description"], 
                              stemmRequired = False, correctWordRequired = False):
            wordcnt += 1
            
            # Only true in the first calling when there is no feature index
            if not featureIndexes:
                wordCounts[category][word] += 1 # Counting words
            
            # At the second calling when there is feature index submitted:
            else:
                # we have to test if we have the keys in the dict... 
                # only in those cases, when the training 
                if word in featureIndexes[category]:
                    col[category].append(featureIndexes[category][word]) # <-
                    row[category].append(cur_row[category]) # <-
        
        # Not the best way to
        if featureIndexes:

            # fetching all the other fields as well...  
            # Collecting these values, when the featureindex is already submitted      
            prices[category].append(int(item["price"]))
            urls[category].append(int(item["urls_cnt"]))
            phones[category].append(int(item["phones_cnt"]))
            emails[category].append(int(item["emails_cnt"]))
            length[category].append(wordcnt) 
            

            cur_row[category] += 1
            if "is_blocked" in item: # Not necessarely has all row this field set... that's why we are selecting. OK it's always set.
                targets[category].append(int(item["is_blocked"]))

            item_ids[category].append(int(item["itemid"]))
                    
        if processedCnt%10000 == 0:                 
            logging.debug(processMessage+": "+str(processedCnt)+" items done")
    
    # 
    if not featureIndexes:
        for category in wordCounts.keys():
            index = 0
            featureIndexes[category] = {}
            
            for word, count in wordCounts[category].iteritems():
                if count >= 3:
                    featureIndexes[category][word] = index
                    index += 1                       
                    
                    
        return featureIndexes

    # if the featureIndexes is provided, a matrix is generated.    
    else:
        # The trickiest line in the script
        # Building the sparse matrix for all category:        
        for category in row.keys():
            features[category] = sp.csr_matrix((np.ones(len(row[category])),(row[category],col[category])), 
                                    shape=(cur_row[category], len(featureIndexes[category])), 
                                    dtype=np.float64)
            print "Number of features: "+category+": "+str(cur_row[category])+" "+str(len(featureIndexes[category]))

        # testing if target exists:
        key = targets.keys()[0]        
        
        if targets[key]:
            print "ids: %d, prices: %d, urls: %d, phones: %d, emails: %d, length: %d"%(len(item_ids), len(prices), len(urls), len(phones), len(emails), len(length))
            return features, targets, item_ids, prices, urls, phones, emails, length
        else:
            print "ids: %d, prices: %d, urls: %d, phones: %d, emails: %d, length: %d"%(len(item_ids), len(prices), len(urls), len(phones), len(emails), len(length))
            return features, item_ids, prices, urls, phones, emails, length




def main():
    """ Generates features and fits classifier. """
    
    # The following 5 command lines can be outcommented if the features are already created.
    # There is no need to process the data every single time.
    # Fine tuning the learning algorythm is much faster without that extra step.
    
    # by reading the train dataset the feature index is created.
    # First calling of the processdata function
    # Data limited to 300000
    featureIndexes = processData(os.path.join(dataFolder,"avito_train.tsv"), itemsLimit=600000)
    print "featureIndex generated!"
    print len(featureIndexes)

    # Trainfeature is created using the indexfeatures...
    # Second calling of the processdata function
    trainFeatures, trainTargets, trainItemIds, trainPrices, trainUrls, trainPhones, trainEmails, trainLength = processData(os.path.join(dataFolder,"avito_train.tsv"), itemsLimit=600000) # Original itemsLimit=300000

    # Building the test dataset... just like the training...
    testFeatures, testItemIds, testPrices, testUrls, testPhones, testEmails, testLength = processData(os.path.join(dataFolder,"avito_test.tsv"), featureIndexes)

    # Dumping data into file...
    # joblib.dump((trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), os.path.join(dataFolder,"train_data.pkl"))
    joblib.dump((trainFeatures,trainTargets,trainItemIds,trainPrices,trainUrls,trainPhones,trainEmails,trainLength,
                 testFeatures, testItemIds,testPrices,testUrls,testPhones,testEmails,testLength), os.path.join(dataFolder,"SeparatedByCategory.pkl"))


    # loading data pack...
    # trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(dataFolder,"train_data.pkl"))

    #logging.info("Feature preparation done, fitting model...")

    # Stochastic gradient model
#    clf = SGDClassifier(    loss="log", 
#                            penalty="l2", 
#                            alpha=1e-4, 
#                            class_weight="auto")
#    #
#    clf.fit(trainFeatures,trainTargets)
#
#    logging.info("Predicting...")
#
#    #     
#    predicted_scores = clf.predict_proba(testFeatures).T[1]
#
#    
#    logging.info("Write results...")
#    #    
#    output_file = "avito_starter_solution.csv"
#    logging.info("Writing submission to %s" % output_file)
#    f = open(os.path.join(dataFolder,output_file), "w")
#    f.write("id\n")
#    
#    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
#        f.write("%d\n" % (item_id))
#    f.close()
#    logging.info("Done.")
                               
if __name__=="__main__":            
    main()