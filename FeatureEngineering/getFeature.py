# -*- coding: utf-8 -*-
"""
A script to access features from the training dataset.

v.1.0 Last modified: 2014.07.20 by Daniel Suveges
    # reads dataset
    # select a given feature and the response variable from the tsv file
    # process given field if necessary
    # test if there is a correlation between the field and the reponse variable


"""


# I will remove those libs, that I don't use:
import csv
import re
import scipy.sparse as sp
import numpy as np
import os
import random as rnd 

# GetItems
def getItems(fileName, itemsLimit, fields):
    """ 
    input:    
        fileName : name of the imput file train tsv
        itemslimit : number of the items to return (by default 30000)
        fields: list of fields to return in an array
    
    Workflow:    
        1) reads input train tsv file
        2) takes itemLimits number of lines from the inputfile
        3) takes the requested fields and return a pd dataframe
    
    """
    
    with open( fileName) as items_fd:

        # The first read (if we need only a sample from the input) is 
        # just reading how many advertisement are there       
        countReader = csv.DictReader(items_fd, delimiter='\t', quotechar='"')
        numItems = 0
        for row in countReader:
            numItems += 1
        items_fd.seek(0)        
        rnd.seed(0)
        
        # Geting random numbers from the range of the total lines
        sampleIndexes = set(rnd.sample(range(numItems),itemsLimit))

        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        itemNum = 0
        
        for i, item in enumerate(itemReader):
            
            if i in sampleIndexes:
                item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}            
                itemNum += 1
                #itemSelect = {k: item[k] for k in fields}
                
                yield itemNum, item
                



    