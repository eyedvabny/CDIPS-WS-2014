# coding: utf-8
"""
Module for word stemming and cleaning.
Extracted from Avito-provided sample code
"""

#RegExp used in correctWord()
import re as _re

# NLTK: natural language processing
import nltk.corpus as _nc
from nltk import SnowballStemmer as _ss

#Pandas for Series manipulation
import pandas as _pd

#Filter out useless (stop) words
_stopwords= frozenset(word.decode('utf-8') for word in _nc.stopwords.words("russian") if word!="не")    

#Reduce the words down to their stems
#http://snowball.tartarus.org/algorithms/russian/stemmer.html
_stemmer = _ss('russian')

#Some cyrillic letters look like latin letters but have different unicode
#Since unicode have to use ord() to get Unicode ordinals
_engChars = [ord(_char) for _char in u"cCyoOBaAKpPeE"]
_rusChars = [ord(_char) for _char in u"сСуоОВаАКрРеЕ"]
_eng_rusTranslateTable = dict(zip(_engChars, _rusChars))
_rus_engTranslateTable = dict(zip(_rusChars, _engChars))

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
    if len(_re.findall(ur"[а-я]",w))>len(_re.findall(ur"[a-z]",w)):
        return w.translate(_eng_rusTranslateTable)
    else:
        return w.translate(_rus_engTranslateTable)

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
    cleanText = _re.sub(u'[^a-zа-я]', ' ', text.lower())
    
    # Check if we want to fix up mixed english-russian words
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired else _stemmer.stem(correctWord(w)) for w in cleanText.split() if len(w)>1 and w not in _stopwords]
    else:
        words = [w if not stemmRequired else _stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in _stopwords]
    
    return words

def splitIntoWords(textSeries, stemmRequired = False, correctWordRequired = False):
    """
    Split an array of strings into one large unique series
    -----------
    textSeries: Pandas Series of unprocessed strings
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required
    """
    # Run the getWords function on every string in the list
    word_list = [word for ugly_string in textSeries for word in getWords(ugly_string,stemmRequired,correctWordRequired)]
    
    # Filter out
    return _pd.Series(word_list).drop_duplicates()
