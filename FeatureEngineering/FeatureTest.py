# -*- coding: utf-8 -*-
"""
A script to test features for machine learning.

v. 1.5 Last modified: 07.19.2014 by Daniel Suveges
    reads two arrays: feature and response
    calcuates the average response for each value of the feature
    creates a barchart, save png if desired
    bot for visualizing the correlation of categorical and numerical feature
    
TODO:
    1) testing inputs if they are valid for the function
  

"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random
import numpy as np

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


def NumericalFeatutreTest(feature, response, featureName = "Feature", responseName = "Response", writeFile=True):
    
    # In the future, a more intensive errorhandling will be implemened to make
    # sure the submitted arrays are proper
    
    # Dataframe is assambled from the input
    df = pd.DataFrame({featureName : feature,
                       responseName : response})
                       
    # getting extremes: 
    bins = 100
    minimum = min(feature)
    maximum = max(feature)

    resp1 = np.histogram(df[df.response == 1][featureName], bins=bins, range=(minimum, maximum), density=True)
    resp0 = np.histogram(df[df.response == 0][featureName], bins=bins, range=(minimum, maximum), density=True)
    
    # PLotting response rates of each feature values:
    now = datetime.datetime.now()
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(resp1[1][:-1],resp1[0], "b-", label=responseName+"= 1, n="+str(len(df[df.response == 1][featureName])) )
    #plt.plot(resp0[1][:-1],resp0[0], "r-", label=responseName+"= 0, n="+str(len(df[df.response == 0][featureName])) )
    ax.set_ylabel('Density', size=15)
    ax.set_xlabel('Feature value', size=15)
    ax.set_title('Testing numeric feature: '+featureName+"\n"+now.strftime("%m/%d %H:%M:%S"), size=15)
    plt.legend(loc='upper right')
    # If desired, the plot is saved:    
    if writeFile:
        filename = "TestedFeature-"+featureName+"_"+now.strftime("%Y-%m-%d_%H%M%S")+".png"
        fig.savefig(filename,  )
        

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
    
    
# Testing the tester :)
def testNumericalFeatutreTest(length=1000):
    """
    This function prepare a biased imput for numericalFeatureTest()
    Two different distributions are mixed then shorted and visualized
    """
    
    # distribution of the response:
    f1 = 1000
    f2 = 9000

    # generating random distribution 1 response = 0
    sigma1 = 14
    mu1 = 35    
    feature1 = [sigma1*np.random.randn()+mu1 for i in range(f1)]
    resp1 = [0]*f1

    # generating random distribution 2 response = 1
    sigma2 = 50
    mu2 = 75    
    feature2 = [sigma2*np.random.randn()+mu2 for i in range(f2)]
    resp2 = [1]*f2
    
    # mixing values
    feature = feature1 + feature2
    response = resp1 + resp2
    random = np.random.permutation(f1+f2)
    
    feature_random = []
    response_random = []
    for i in random:
        feature_random.append(feature[i])
        response_random.append(response[i])

    # submit for the test script
    NumericalFeatutreTest(feature_random, response_random, featureName="test feature", writeFile=True)

