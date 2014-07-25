# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:04:51 2014

All scripts process the entire array. Not the individual values!!
Modified to work with Bayes classifier which does not accept negative integers

@author: daniel
"""

import numpy as np

def url_cnt(urls):
    ''' 
    A function to pool url_cnt field 

    return 0, 1 or 2 depending which bin the submitted count is in    
    '''

    urlCountClear = []
    
    for count in urls:
        count = int(count)        
        
        if count > 10:
            urlCountClear.append(2)
        elif count > 0:
            urlCountClear.append(1)
        else:
            urlCountClear.append(0)

    return urlCountClear
        

def phone_cnt(phones):
    ''' 
    A function to pool phone_cnt field 

    return 0, 1 or 2 depending which bin the submitted count is in    
    '''
    phoneCountClear = [] 
    
    for count in phones:
        count = int(count) 
        
        if count > 10:
            phoneCountClear.append(0)
        elif count > 2:
            phoneCountClear.append(1)
        else:
            phoneCountClear.append(2)

    return phoneCountClear 


def emails_cnt(emails):
    ''' 
    A function to pool email_cnt field 

    return 0, 1 or 2 depending which bin the submitted count is in    
    '''
    emailsCountClear = []

    for count in emails:     
        count = int(count)         
        if count > 0:
            emailsCountClear.append(1)
        else:
            emailsCountClear.append(0)

    return emailsCountClear


def price(prices):
    ''' 
    A function to pool price field 

    The price is a really wide range, therefore we at first take the log value, then int 
    '''
    pricesClear = []
    
    for price in prices:
        price = int(price) 
        if price == 0:
            pricesClear.append(0)
        else:
            pricesClear.append(int(np.log10(price)))
    
    return pricesClear
