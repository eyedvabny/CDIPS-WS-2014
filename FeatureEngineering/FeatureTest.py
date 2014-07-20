# -*- coding: utf-8 -*-
"""
A script to test features for machine learning.

v. 1.0 Last modified: 07.19.2014 by Daniel Suveges
    reads two arrays: feature and response
    calcuates the average response for each value of the feature
    creates a barchart, save png if desired
    
    
TODO:
    1) testing inputs if they are valid for the function
    2) built-in test function    
    2) not only categorical, but also for continous features as well 
    eg. time interval, number of words etc.

"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random

def categoricalFeatureTest (feature, response, featureName = "Feature", writeFile=True):
    """
    feature: categorical variable, array, processed as scalar
    response: response variable, binary data: 0 or 1 is accepted in this version
    freatureName: name of the feature, optional, for documentation purpose
    writeFile: boolean, if True, the generated plot is saved as a png 
    """    
    
    # Testing input arrays if they are valid, at this point of the development, 
    # I assume, it is correct

    averageResponse = float(sum(response)) / len(response)
    
    # generating pandas dataframe:
    df = pd.DataFrame({featureName:feature, 
                       "response":response})
    
    # Number of features:
    featuresCount = len(set(feature))
    
    # Looping through all features:
    featureRatio = {}
    featureCount = {}
    for f in set(feature):
        responseForF = df[df[featureName] == f]["response"]
        featureRatio[f] = float(sum(responseForF)) / len(responseForF)
        featureCount[f] = len(responseForF)
    
    # PLotting response rates of each feature values:
    now = datetime.datetime.now()
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.bar(range(featuresCount),featureRatio.values(), width=0.3, align="center")
    
    # setting x axis labels with the feature counts:
    xtickLabels = []
    for key in featureRatio.keys():
        xtickLabels.append(key+"\n"+str(featureCount[key]))
    plt.xticks(range(featuresCount), xtickLabels)
    plt.tight_layout(pad=3.8)
    ax.plot([-0.5,featuresCount-0.5],[averageResponse,averageResponse],"r-", ms=3)
    ax.set_ylabel('Response rate', size=15)
    ax.set_title('Testing feature: '+featureName+"\n"+now.strftime("%m/%d %H:%M:%S"), size=15)
    
    # If desired, the plot is saved:    
    if writeFile:
        filename = "TestedFeature-"+featureName+"_"+now.strftime("%Y-%m-%d_%H%M%S")+".png"
        fig.savefig(filename,  )
        
    # Printing report:
    print "Number of items: "+str(len(feature))    
    print "Average response rate: "+str(averageResponse)    
    print "The tested feature has: "+str(featuresCount)+" values\nList of values:"
    for key in featureRatio.keys():
        print key+" n="+str(featureCount[key])+" response rate="+str(featureRatio[key])



# Testing the tester :)
def testCategoricalFeatutreTest(length=1000):
    """
    This function just generates a random distribution of features and responses
    and calls the tester.
    """

    # generating random feature set
    featureSet = ["Value_1"]*5 + ["Value_2"] + ["Value_3"] + ["Value_4"]*2
    feature = [random.choice(featureSet) for r in range(length)]
    
    # generating random feature set
    responseSet = [0]*9 + [1]
    response = [random.choice(responseSet) for r in range(length)]
    
    # calling categoricalFeatureTest
    categoricalFeatureTest(feature, response, featureName = "Test Feature", writeFile=True)