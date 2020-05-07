import yaml 
import pandas as pd
import numpy as np
import re
import sys
import logging
from nltk.tokenize import word_tokenize
import nltk.data
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.sentiment.util import mark_negation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)


class sentiAnalysis:
    
    def __init__(self):
        return
        
    def removeEmptyData(self, data):
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data
    
    def getValid(self, column):
        invalids = ['nope', 'none', 'no comments', 'no comment', 'no thanks', 'no thank you', 'none thanks', 'none thank you', 'no i don\'t', 'no i do not']
        column = column.apply(lambda x: np.nan if (x is np.nan or len(x) < 4 or x.lower() in invalids) else x)
        return column
    
    def getSentences(self, ser):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentencesSer = ser.apply(tokenizer.tokenize)
        return sentencesSer
    
    def getPOS(self, ser):
        posSer = ser.apply(lambda x: nltk.pos_tag(word_tokenize(x)))
        return posSer
    
    def removeProperNouns(self, ser):
    
        TEXT = []
        
        for resp in ser:
            if pd.isnull(resp):
                TEXT.append(resp)
            else:
                lst = nltk.tag.pos_tag(resp.split())
                sent = ""
                for word, tag in lst:
                    if tag not in ['NNP', 'NNPS']:
                        sent = sent + " " + word   
                TEXT.append(sent)
                
        se = pd.Series(TEXT)

        return(se)
    
    def getSentiPhrase_allpos(self, lst):
    
        usefulPhrases = []
        searchBreadth = 1
        dictOfPhrases = { i : lst[i] for i in range(0, len(lst)) }

        ndict = {}
        jdict = {}
        vdict = {}
        rdict = {}
            
        for key, val in dictOfPhrases.items():
            
            # get the pos for the word processed
            pos = val[1]
            
            if pos.startswith('J'):
                jdict[key] = val
            elif pos.startswith('V'):
                vdict[key] = val
            elif pos.startswith('N'):
                ndict[key] = val
            elif pos.startswith('R'):
                rdict[key] = val
         
        # get pattern JJ<-searchbreadth->NN
        for knoun, vnoun in ndict.items():
            phrase = ""
            for b in range(1, searchBreadth + 1):
                if int(knoun - b) in jdict: # look for the adjective before and after the noun
                    phrase = (jdict[knoun - b][0], ndict[knoun][0])
                elif int(knoun + b) in jdict:
                    phrase = (jdict[knoun + b][0], ndict[knoun][0])

                if phrase != "":
                    usefulPhrases.append(phrase)
                    break
        
        # get pattern RB VB
        for kverb, vverb in vdict.items():
            phrase = ""
            for b in range(1, searchBreadth + 1):
                # look for the adverb before the verb as per https://arxiv.org/ftp/cs/papers/0212/0212032.pdf - page 2
                if int(kverb - b) in rdict: 
                    phrase = (rdict[kverb - b][0], vdict[kverb][0])
                
                if phrase != "":
                    usefulPhrases.append(phrase)
                    break
        
        # get pattern RB JJ
        for kadj, vadj in jdict.items():
            phrase = ""
            for b in range(1, 2):
                # look for the adverb before the adjetive 
                if int(kadj - b) in rdict: 
                    phrase = (rdict[kadj - b][0], jdict[kadj][0])
                    
                if phrase != "":
                    usefulPhrases.append(phrase)
                    break
                    
        # get pattern VB JJ
        for kadj, vadj in jdict.items():
            phrase = ""
            for b in range(1, 2):
                # look for the verb before the adjetive 
                if int(kadj - b) in vdict: 
                    phrase = (vdict[kadj - b][0], jdict[kadj][0])
    #                 print (phrase)
                    
                if phrase != "":
                    usefulPhrases.append(phrase)
                    break
                    
        return usefulPhrases
    
    def getSentimotions_allpos(self, ser):
        sentiPhrases = ser.apply(lambda x: self.getSentiPhrase_allpos(x))
        return sentiPhrases
    
    def getWordpos(self, pos, word):
    
        wpos = None
        for w, p in pos:
            if w == word:
    #             print ("Word pos - ", w, p)
                wpos = p
                break
            
        return p


    def calcSentiScore_pos(self, pos, phrases):
        
        sScore = 0.0
        for p in phrases:
            pScore = 0.0
            i = 0
            wordseq = []
            for w in p:
                wScore = 0.0
                synsets = None
                if self.getWordpos(pos, w).startswith('J'):
                    synsets = wn.synsets(w, pos=wn.ADJ)
                    if not synsets:
                        continue
                    wSynset = swn.senti_synset(synsets[0].name())
                    wScore = wSynset.pos_score() - wSynset.neg_score()
                    wordseq.append((wn.ADJ, wScore))
                elif self.getWordpos(pos, w).startswith('R'):
                    synsets = wn.synsets(w, pos=wn.ADV)
                    if not synsets:
                        continue
                    wSynset = swn.senti_synset(synsets[0].name())
                    wScore = wSynset.pos_score() - wSynset.neg_score()
                    wordseq.append((wn.ADV, wScore))
                
            # there's only one senti word in the phrase e.g JJ NN
            if (len(wordseq) == 1):
                pScore = pScore + wordseq[0][1]
            
            # there are 2 senti words in the phrase e.g. RB JJ (extremely bad)
            if (len(wordseq) == 2):
                rScore = wordseq[0][1] # adverb score
                jScore = wordseq[1][1] # adjective score
                if jScore < 0 and rScore > 0:
                    rScore = -(rScore) # negate the positive sentiment if the second word is negative
                    
                elif jScore > 0 and rScore < 0:
                    jScore = -(jScore)
                    
                pScore = pScore + rScore
                pScore = pScore + jScore
                
            sScore = sScore + pScore 
            
        return sScore
                
    def getSentiScore_pos(self, df):
        sentiScore = df.apply(lambda x: self.calcSentiScore_pos(x['pos'], x['sentiPhrases']), axis=1)
        return sentiScore
    
    def getNegations(self, ser):
        negValues = ser.apply(lambda x: '_NEG' in (','.join(mark_negation(x.split()))))
        return negValues
    
    def updateNegResp(self, negations, negatives, score):
        
        if negations == True or len(negatives) > 0:
            if score == 0:
                score = -1.0
            elif score > 0:
                score = -score
        
        return score

    def updateNegationScores(self, df):
        nser = df.apply(lambda x: self.updateNegResp(x['negations'], x['negatives'], x['sentiScore']), axis=1)
        return nser
        
    # Using the negative word list
    def getNegativeSent(self, x, nglst):
        
        ngWords = []
        for word in str(x).split():
            if word.lower() in nglst:
                ngWords.append(word)

        return ngWords

    def getNegatives(self, ser):
        
        # Customize stopwords list, add UBC and ubc
        fileName = "config/negative_words.txt"
        nglst = [line.rstrip('\n') for line in open(fileName)]
        nglst = [x.lower() for x in nglst]
        negatives = ser.apply(lambda x: self.getNegativeSent(str(x), nglst))
        
        return negatives
    
    def getModalsSent(self, x, mlst):
        mWords = []
        for word in x.split():
            if word.lower() in mlst:
                mWords.append(word)

        return mWords

    def getModals(self, ser):
        
        # Customize stopwords list, add UBC and ubc
        fileName = self._modalPath
        mlst = [line.rstrip('\n') for line in open(fileName)]
        mlst = [x.lower() for x in mlst]
        modals = ser.apply(lambda x: self.getModalsSent(x, mlst))
        
        return modals
        
    def updateModalResp(self, modals, score):
    
        if modals:
            if score == 0:
                score = -1.0
            elif score > 0:
                score = -score
        
        return score
            

    def updateModalScores(self, df):
        mser = df.apply(lambda x: self.updateModalResp(x['modals'], x['sentiScore']), axis=1)
        return mser
        
    def getPositiveSent(self, x, plst):
    
        pWords = []
        for word in x.split():
            if word.lower() in plst:
                pWords.append(word)

        return pWords

    def getPositives(self, ser):
        
        # Customize stopwords list, add UBC and ubc
        fileName = self._positivesPath
        plst = [line.rstrip('\n') for line in open(fileName)]
        plst = [x.lower() for x in plst]
        positives = ser.apply(lambda x: self.getPositiveSent(x, plst))
        
        return positives
        
    def updatePositivesResp(self, positives, score):

        if positives:
            if score == 0:
                score = 1.0
            elif score < 0:
                score = -score
        
        return score
            

    def updatePositiveScores(self, df):
        pser = df.apply(lambda x: self.updatePositivesResp(x['positive'], x['sentiScore']), axis=1)
        return pser
    
    def getFlags(self, col):
        fileName = 'flag/flag_words.txt'
        search_list = [line.rstrip('\n') for line in open(fileName)]
        flagSer = col.apply(lambda x: False if x is np.nan else (True if any(word in search_list for word in x.split()) else False))
        return flagSer
    
    def updateFlaggedResp(self, flags, score):
        
        if flags == True:
            if score > 0:
                score = -1.0    
        return score
       
    def updateFlagScores(self, df):
        fser = df.apply(lambda x: self.updateFlaggedResp(x['flags'], x['sentiScore']), axis=1)
        return fser

    def getContraSent(self, x, clst):
        
        cWords = []
        
        for word in x.split():
            if word.lower() in clst:
                cWords.append(word)

        return cWords

    def getContra(self, ser):
        
        # Customize stopwords list, add UBC and ubc
        fileName = "config/contradiction_words.txt"
        clst = [line.rstrip('\n') for line in open(fileName)]
        clst = [x.lower() for x in clst]
        contra = ser.apply(lambda x: self.getContraSent(x, clst))
        
        return contra
        
    def updateContraResp(self, contra, senti):

        score = senti
        if float(senti) > 0 and len(contra) > 0:
            score = -senti
        
        return score
        

    def updateContraScores(self, df):
        ctser = df.apply(lambda x: self.updateContraResp(x['contra'], x['sentiScore']), axis=1)
        return ctser


    def sentiment_analyzer_scores(self, sentence):
        
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(sentence)
    #     print("{:-<40} {}".format(sentence, str(score)))
        return (str(score['compound']))
        
    def getVaderScores(self, ser):
        vser = ser.apply(lambda x: self.sentiment_analyzer_scores(x))
        return vser
    
    def updateVaderResp(self, vader, senti):

        score = senti
        if float(senti) == 0 and float(vader) != 0:
            score = vader
        if float(senti) > 0 and float(vader) < 0:
            score = vader
        
        return score
            

    def updateVaderScores(self, df):
        vdser = df.apply(lambda x: self.updateVaderResp(x['vader'], x['sentiScore']), axis=1)
        return vdser
    
    def _setModalPath(self, path=None):
        self._modalPath = path
        
    def _setPositivesPath(self, path=None):
        self._positivesPath = path
    
    def _setNegativesPath(self, path=None):
        self._negativesPath = path
    
    
    def analyze(self, dataSent, id):
        
        #dataSent['sentences'] = self.getValid(dataSent['sentences'])
        #dataSent = self.removeEmptyData(dataSent)
        #data['comments'] = fixQuotes(data['comments'])
        #data['sentences'] = getSentences(data['comments'])
        
        dataSent['noProperNoun'] = self.removeProperNouns(dataSent['sentences'])
        #dataSent['cleanedSentence'] = self.getCleanedData(dataSent['noProperNoun'])
        dataSent['pos'] = self.getPOS(dataSent['noProperNoun']) # on the raw sentences minus proper nouns
        
        dataSent['sentiPhrases'] = self.getSentimotions_allpos(dataSent['pos'])
        
        dataSent['sentiScore'] = self.getSentiScore_pos(dataSent)
        
        dataSent['positive'] = self.getPositives(dataSent['sentences'])
        dataSent['sentiScore'] = self.updatePositiveScores(dataSent)
        
        dataSent['negations'] = self.getNegations(dataSent['sentences'])
        dataSent['negatives'] = self.getNegatives(dataSent['sentences'])
        dataSent['sentiScore'] = self.updateNegationScores(dataSent)

        dataSent['modals'] = self.getModals(dataSent['sentences'])
        dataSent['sentiScore'] = self.updateModalScores(dataSent)
        
        dataSent['flags'] = self.getFlags(dataSent['sentences'])
        dataSent['sentiScore'] = self.updateFlagScores(dataSent)
        
        dataSent['contra'] = self.getContra(dataSent['sentences'])
        dataSent['sentiScore'] = self.updateContraScores(dataSent)
        
        dataSent['vader'] = self.getVaderScores(dataSent['sentences'])
        dataSent['sentiScore'] = self.updateVaderScores(dataSent)
        
        dataSent['sentiment'] = dataSent['sentiScore'].apply(lambda x: 'positive' if float(x) > 0 else ('negative' if float(x) < 0 else 'neutral'))
        
        sentiResults = dataSent[[id, 'sentences', 'sentences_cleaned', 'tags', 'sentiScore', 'sentiment']]
        
        #sentiResults.to_csv('tmp/sentiResults_sent.csv', encoding="ISO-8859-1")
        
        return sentiResults
        
        
    def getSentiResponse(self, dataSent, id):
    
        dataSent['sentiScore'] = pd.to_numeric(dataSent['sentiScore'])
        dataSent_less = dataSent[[id, 'sentences', 'sentiScore']]
        aggRespdf = dataSent_less.groupby(id)[['sentences', 'sentiScore']].apply(lambda x: x.sum())
        aggRespdf['sentiment'] = aggRespdf['sentiScore'].apply(lambda x: 'positive' if float(x) > 0 else ('negative' if float(x) < 0 else 'neutral'))
        
        return aggRespdf