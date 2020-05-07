# install pattern
# install gensim 
# install nltk
# install pyspellchecker
import re
import pandas as pd
import numpy as np
import gensim
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from spellchecker import SpellChecker

   
class Cleaning: 

    def __init__(self):
        self.WORDS = {}
        return
    
    # remove urls (starts with https, http)
    def remove_URL(self, col):
        
        text = col.tolist()
        TEXT=[]
        
        for word in text:
            if pd.isnull(word):
                TEXT.append(word)
            else:
                TEXT.append(re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(word)))
        
        se = pd.Series(TEXT)
        
        return(se)


    def count_mark(self, col):

        df = pd.DataFrame()
        rdf = pd.DataFrame()

        # remove the special characters (numbers, exclamations and question marks) from the text
        # store them in a dataframe for later use
        text = col.tolist()

        for row in text:
            if pd.isnull(row):
                ser = pd.Series([np.nan,np.nan,np.nan,np.nan], index=['Number', 'Exclamation_count', 'Question_Mark_count', 'Comments_OE'])
                df = df.append(ser, ignore_index=True)
            else:
                numVals = []
                excCount = []
                quesCount = []

                num = re.findall(r'\b\d+\b', row)
                numVals.append(num)
                # remove the number from the text
                for n in num:
                    row = row.replace(n, '') 

                excCount.append(row.count('!'))
                row = row.replace('!', '')

                quesCount.append(row.count('?'))
                row = row.replace('?', '')

                numSeries = pd.Series(numVals)
                rdf['Number'] = numSeries.values

                excSeries = pd.Series(excCount)
                rdf['Exclamation_count'] = excSeries

                quesSeries = pd.Series(quesCount)
                rdf['Question_Mark_count'] = quesSeries

                txtSeries = pd.Series(row)
                rdf['Comments_OE'] = txtSeries

                df = df.append(rdf, ignore_index=True)
                rdf = pd.DataFrame()

        df.reset_index(inplace=True)
        return(df)


    def remove_special(self, col):

        tokenizer = RegexpTokenizer(r'\w+')
        text = col.str.lower().tolist()
        TEXT=[]

        for word in text:
            if pd.isnull(word):
                TEXT.append(word)
            else:
                TEXT.append(' '.join(tokenizer.tokenize(word)))

        se = pd.Series(TEXT)
        return(se)



    def remove_stop (self, col):
    
        stop_words = stopwords.words('english')
    
        # Customize stopwords list, add UBC and ubc
        fileName = "config/pair_stopwords.txt"
        lineList = [line.rstrip('\n') for line in open(fileName)]

        stop_words.extend(lineList)
#       print (stop_words)
        TEXT = []
        filtered_text = []
    
        for resp in col:
            if pd.isnull(resp):
                TEXT.append(resp)
            else:
                resp = resp.replace(' co op ', ' coop ') # problem specific 
                resp = resp.replace(' co operative ', ' cooperative ') # problem specific 
                
                wordlst = resp.split()
                for word in wordlst:
                    if word not in stop_words:
                        filtered_text.append(word)
                TEXT.append(' '.join(filtered_text))
                filtered_text = []
        
        se = pd.Series(TEXT)
    
        return(se)

    # function to convert nltk tag to wordnet tag
    def __nltk_tag_to_wordnet_tag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def lemmatize_response(self, col):
    
        lemmatizer = WordNetLemmatizer() 

        TEXT = []
        lemma_text = []

        for resp in col:
            if pd.isnull(resp):
                TEXT.append(resp)
            else:
                #tokenize the response and find the POS tag for each token
                nltk_tagged = nltk.pos_tag(nltk.word_tokenize(resp))  
                #tuple of (token, wordnet_tag)
                wordnet_tagged = map(lambda x: (x[0], self.__nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
            
                lemmatized_resp = []
                for word, tag in wordnet_tagged:
                    if tag is None:
                        #if there is no available tag, append the token as is
                        lemmatized_resp.append(word)
                    else:        
                        #else use the tag to lemmatize the token
                        lemmatized_resp.append(lemmatizer.lemmatize(word, tag))
                TEXT.append(' '.join(lemmatized_resp))

        se = pd.Series(TEXT)

        return(se)


#    def __loadSpellCheckDict(self):
#        # replace the path below with your path location to the vector lib
#        model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz',binary=True)
#        words = model.index2word
#        w_rank = {}
#        for i,word in enumerate(words):
#            w_rank[word] = i
#
#        self.WORDS = w_rank
#
#        return
#
#    def __P(self, word): 
#        "Probability of `word`."
#        # use inverse of rank as proxy
#        # returns 0 if the word isn't in the dictionary
#        return - self.WORDS.get(word, 0)
#
#    def __known(self, words): 
#        "The subset of `words` that appear in the dictionary of WORDS."
#        return set(w for w in words if w in self.WORDS)
#
#    def __candidates(self, word): 
#        "Generate possible spelling corrections for word."
#        return (self.__known([word]) or self.__known(self.__edits1(word)) or self.__known(self.__edits2(word)) or [word])
#
#    def __edits1(self, word):
#        "All edits that are one edit away from `word`."
#        letters    = 'abcdefghijklmnopqrstuvwxyz'
#        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
#        deletes    = [L + R[1:]               for L, R in splits if R]
#        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
#        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
#        inserts    = [L + c + R               for L, R in splits for c in letters]
#        return set(deletes + transposes + replaces + inserts)
#
#    def __edits2(self, word): 
#        "All edits that are two edits away from `word`."
#        return (e2 for e1 in self.__edits1(word) for e2 in self.__edits1(e1))
#
#    def __correction(self, word): 
#        "Most probable spelling correction for word."
#        return max(self.__candidates(word), key=self.__P)


#    def fix_typo(self, col):
#
#        # Load the dictionary if not loaded
#        if not self.WORDS:
#            self.__loadSpellCheckDict()
#
#        TEXT=[]
#        for i in col:
#            if pd.isnull(i):
#                TEXT.append(i)
#            else:
#                words = i.split()
#                fixedWords = []
#                for j in words:
#                    #remove characters that shows more than two
#                    pattern = re.compile(r"(.)\1{2,}")
#                    text = pattern.sub(r"\1\1", j)
#                    fixedWords.append(self.__correction(text))
#                TEXT.append(' '.join(fixedWords))
#
#        se = pd.Series(TEXT) 
#
#        return(se)




    def fix_typo(self, col):
        text = col.astype(str).tolist()
        spell = SpellChecker()
        TEXT=[]
    
        for resp in text:
            wordlst = resp.split()
            misspelled = spell.unknown(wordlst)
            ww=set(['website', 'webpage', 'facebook', 'midterm', 'internship', '1st', '2nd', '3rd', '4th', 'lgbt', 'skytrain', 'translink','jumpstart', 'emails', 'mentorship', 'undergrad','foodcourt', 'grad'])
            #print(misspelled-ww)
            line = [spell.correction(word) if word in (misspelled - ww) else word for word in wordlst]
            TEXT.append(' '.join(line))
        return(pd.Series(TEXT))
    
    # Combine all the cleaning steps in this function
    def clean(self, field, url=True, count_mark=True, specChars_lower=True, stopwords=True, typo=True, lemmatize=True, noProperNoun=True):

        # Change empty values to NaN
        field = field.replace(" ", np.nan)
        
        # Remove urls
        if url:
            field = self.remove_URL(field)

        # Remove !, ? and numerical values from the text
        if count_mark:
            sentiExp = self.count_mark(field)
            # Extract the cleaned text column
            field = sentiExp['Comments_OE']
            # Store the counts of exclamations, questions marks and numbers for later use
            sentiExp.drop(['Comments_OE'], axis=1)

        # Remove all characters except text and convert the text to lower case
        if specChars_lower:
            field = self.remove_special(field)

        # Remove stop words
        if stopwords:
            field = self.remove_stop(field)

        # Remove typos
        if typo:
            field = self.fix_typo(field)

        # Remove Lemmatization
        if lemmatize:
            field = self.lemmatize_response(field)

        return field
    
    



