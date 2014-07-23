
import csv
import re
import scipy.sparse as sp
import numpy as np
import pandas as pd
import os
import random as rnd 
import matplotlib.pyplot as plt
import datetime

numItems = 150000 # Number of items to be read from the train data:
os.chdir("/Users/daniel/Documents/Programming/DSWorshop/Illicit_content")

fileName = "avito_train.tsv" # File name
df = pd.DataFrame(columns=["itemid","category","subcategory","title","price",
                           "is_proved","is_blocked", "phones_cnt", "emails_cnt", "urls_cnt", "close_hours"])

with open(fileName) as items_fd:
    
    countReader = csv.DictReader(items_fd, delimiter='\t', quotechar='"')
    numItems = 0
    
    for row in countReader:
        numItems += 1
    
    items_fd.seek(0)        
    rnd.seed(0)

    # Geting random numbers from the range of the total lines
    sampleIndexes = set(rnd.sample(range(numItems),numItems))

    itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
    itemNum = 0

    for i, item in enumerate(itemReader):

        if i in sampleIndexes:
            item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}            
            df = df.append(item, ignore_index=True)            
            #df.append() = pd.Series(item)
            itemNum += 1
            print itemNum