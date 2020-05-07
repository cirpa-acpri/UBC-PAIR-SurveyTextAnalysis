
# coding: utf-8

# In[ ]:


from nltk.collocations import *
import nltk
from nltk.corpus import wordnet
import numpy as np
import os
import sys

import cleaning.cleaning as clng   


class Bigram:

    def __init__(self):
        return
    
    def replace(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lm in syn.lemmas():
                 synonyms.append(lm.name())

        if len(synonyms) >= 1:
            return synonyms[0]
        else:
            return None 

    def replace_lines(self, line_list):

        result_list = []

        for line in line_list:
            line = line.split(' ')
            line = [x for x in line if x]
            line = [self.replace(word) if word[0].isdigit() == True else word for word in line]
            result_list.extend(line)

        return result_list

    def analyze(self, col):
        print("Start Bigram...")
        cln = clng.Cleaning()
        L2 = cln.clean(col)
        L2 = L2.replace('nan', np.nan)
        L2 = L2.dropna().tolist()
        if '' in L2:
            L2.remove('')
        L = []
        L2 = self.replace_lines(L2)
        bgs = nltk.bigrams(L2)

        #compute frequency distribution for all the bigrams in the text
        fdist = nltk.FreqDist(bgs)
        dict1={}
        for k,v in fdist.most_common(5):
            dict1[k] = round(v)
        return(dict1)

