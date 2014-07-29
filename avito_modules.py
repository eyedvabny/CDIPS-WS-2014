﻿# coding: utf-8
"""
Module for word stemming and cleaning.
Extracted from Avito-provided sample code
"""

#RegExp used in correctWord()
import re

# NLTK: natural language processing
import nltk.corpus
from nltk import SnowballStemmer

# Use Pandas for parsing large tables
import pandas as pd

#Filter out useless (stop) words
stopwords= frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian") if word!="не")    

#Reduce the words down to their stems
#http://snowball.tartarus.org/algorithms/russian/stemmer.html
stemmer = SnowballStemmer('russian')

#Some cyrillic letters look like latin letters but have different unicode
#Since unicode have to use ord() to get Unicode ordinals
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

def correctWord (w):
    """ Corrects word by replacing characters with written similarly depending on which language the word. 
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""
    
    #Return an empty unicode string if w is not unicode
    if not isinstance(w,unicode):
        return u''

    # If number of russian letters is longer than number of english letters
    # Assume russian is the main language and translate the rogue english letter
    # Otherwise translate the rogue russian letters in english words
    # Words from getWords will be all lowercase
    if len(re.findall(ur"[а-я]",w))>len(re.findall(ur"[a-z]",w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)

def getWords(text, stemmRequired = False, correctWordRequired = False):
    """ Splits the text into words, discards stop words and applies stemmer. 
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required     
    """
    
    #Return an empty unicode string if w is not unicode
    if not isinstance(text,unicode):
        return u''

    # Change to lowercase and remove all non-alphanumeric characters
    cleanText = re.sub(u'[^a-zа-я]', ' ', text.lower())
    
    # Check if we want to fix up mixed english-russian words
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired else stemmer.stem(correctWord(w)) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    else:
        words = [w if not stemmRequired else stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    
    return words

def splitIntoWords(textSeries, stemmRequired = False, correctWordRequired = False):
    """
    Split an array of strings into one large unique series
    -----------
    textSeries: Pandas Series of unprocessed strings
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required
    """
    # Convert the array into a list
    string_list = textSeries.tolist()

    # Run the function on every string in the list
    word_list = [word for ugly_string in string_list for word in getWords(ugly_string)]
    
    # Filter out
    return pd.Series(word_list).drop_duplicates()




