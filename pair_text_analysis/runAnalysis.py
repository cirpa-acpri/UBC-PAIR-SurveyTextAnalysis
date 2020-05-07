import os
import sys
import warnings
import errno
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import fpdf
import re


import matplotlib
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PyPDF2 import PdfFileMerger


# suppressing all warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import topic.topic_tagging as tm
import senti.sentiAnalysis as sa
import keyphrase.Unigram as ug
import keyphrase.Bigram as bg
import flag.flag as fl
import viz.viz as viz

import logging
import yaml 
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk.data

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

def getFiles():
    
    try:
        with open("config/config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "config/config.yml")
    else:
        logging.info('Config file opened for reading')
        
    config = {}
    config['appPath'] = os.getcwd()
    for section in cfg:
        if section == 'data':
            config['data'] = cfg[section]
        if section == 'files':
            config['files'] = cfg[section]
        if section == 'stopwords_file':
            config['stopwordsFile'] = cfg[section]
        if section == 'googWordVector_File':
            config['typofix'] = cfg[section]
        if section == 'cleaning':
            config['cleaning'] = cfg[section]
        if section == 'modal_words':
            config['modal'] = cfg[section]
        if section == 'positive_words':
            config['positive'] = cfg[section]
        if section == 'negative_words':
            config['negative'] = cfg[section]
        if section == 'contradiction_words':
            config['contradiction'] = cfg[section]
        if section == 'word2Vec_File':
            config['word2vec'] = cfg[section]
        if section == 'flag_words':
            config['flag'] = cfg[section]
    
    return config
	
# Create the output folder
if not os.path.exists('output') or not os.path.isdir('output'):
    os.mkdir('output')
if not os.path.exists('tmp') or not os.path.isdir('tmp'):
    os.mkdir('tmp')
    
conf = getFiles()

dataPath = conf['data']['path']
files = conf['files']
word2vec = conf['word2vec']['path']
appPath = conf['appPath']

totalOP = {}
for item in files:
    
    f = files[item]
    fileOP = {}
    
    name = ""
    fileId = ""
    sentiCols = []
    topicCols = []
    kpuCols = []
    kpbCols = []
    topicColSet = set()
    flagColSet = set()
    
    for elem in f:
        
        idict = elem
        
        if 'name' in idict:
            name = idict['name'] 
        if 'id' in idict:
            fileId = idict['id']
        if 'senti_columns' in idict:
            sentiCols = idict['senti_columns']
        if 'topic_columns' in idict:
            topicCols = idict['topic_columns']
        if 'kp-unigram_columns' in idict:
            kpuCols = idict['kp-unigram_columns']
        if 'kp-bigram_columns' in idict:
            kpbCols = idict['kp-bigram_columns']

    filepath = dataPath + '/' + name
    if not os.path.exists(filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
    if fileId == "":
        raise ValueError('% file is not provided an id value in the config', name)
        
    topicColSet = set(topicCols + sentiCols)
    flagColSet = set(sentiCols + topicCols + kpuCols + kpbCols)

    processedCols = {}  
    flagsDf = pd.DataFrame()
    
    logging.info("Processing file: %s..", name)
    # Process sentiments and topics
    for col in topicColSet:

        logging.info("Processing Column: %s..", col)
        
        logging.info("Start Topic Tagging of sentences..")
        # Perform topic tagging for the column
        try:
            topicObj = tm.topic_tagging(path_full_data = word2vec, path_data_col = filepath, id_col_name = fileId, col_name = col)
            d1, d2, d3, d4 = topicObj.assign_topics()
            
        except:
            logging.error("Error in topic tagging sentences", exc_info=1)
        else:
            logging.info("Topic Tagging of sentences completed")
            
            if col in sentiCols:
                logging.info("Start Sentiment Analysis of sentences..")
                # Perform sentiment analysis for the column at the sentence level
                try:
                    sentiObj = sa.sentiAnalysis()
                    sentiObj._modalPath = conf['modal']['path']
                    sentiObj._positivesPath = conf['positive']['path']
                    sentiObj._negativesPath = conf['negative']['path']

                    # Get columns for sentiment analysis
                    tm2saDf = d1[[fileId, 'sentences', 'sentences_cleaned', 'tags']]
                    df = sentiObj.analyze(tm2saDf, fileId)
                except:
                    logging.error("Error in sentiment analysis", exc_info=1)
                else:
                    logging.info("Sentiment Analysis of sentences completed")
            
                    logging.info("Getting Sentiment Analysis of responses")
                    # Perform sentiment analysis for the column at the response level
                    respDf = sentiObj.getSentiResponse(df, fileId)
            #         print (respDf.head())

                    # rename the column to its initial name
                    colpos = list(df).index('sentences')
                    df.columns.values[colpos] = col

                    # Rename the columns to match the processed column name
                    df.rename(index=str, columns={"tags": col + "_tags" , "sentences_cleaned": col + "_sentences_cleaned", "sentiScore": col + "_sentiScore", "sentiment": col + "_sentiment"}, inplace=True)
                    
                    procLevel = {}
                    # Add the processed df into a list
                    procLevel['sentence'] = df
                    
                    # Combine for the response level, the sentiment anaysis output and the topic tagging output 
                    tpsa_resp = pd.merge(d4, respDf, left_on=fileId, right_on=fileId)
                    tpsa_resp.rename(index=str, columns={"topics": col + "_tags" , "sentences": col + "_sentences", "sentiScore": col + "_sentiScore", "sentiment": col + "_sentiment"}, inplace=True)
                    procLevel['response'] = tpsa_resp
            
            # If only topic modeling is specified
            else:
                # rename the column to its initial name
                colpos = list(d1).index('sentences')
                d1.columns.values[colpos] = col

                # Rename the columns to match the processed column name
                d1.rename(index=str, columns={"tags": col + "_tags" , "sentences_cleaned": col + "_sentences_cleaned"}, inplace=True)
            
                procLevel = {}
                # Add the processed df into a list
                procLevel['sentence'] = d1
            
                # update the response level output for only topic tagging
                tpsa_resp = d4
                tpsa_resp.rename(index=str, columns={"topics": col + "_tags" , "sentences": col + "_sentences"}, inplace=True)
                procLevel['response'] = tpsa_resp
                
            processedCols[col] = procLevel
    
    # Process the flags
    try:
        fdf = pd.read_csv(filepath, sep=",", encoding="ISO-8859-1")
    except:
        logging.error("Could not open the file", name, "for processing flags", exc_info=1)
    else:
        logging.info('Start Processing flags for file: %s..', name)
        
        flst = []
        flst.append(fileId)
        flst = flst + list(flagColSet)

        # Get all the flag columns  
        fdf = fdf[flst]

        try:
            flagObj = fl.Flag()
            flagObj._setFlagPath(conf['flag']['path'])

            for col in flagColSet:
                ser = fdf[col]
                flags = flagObj.getFlags(ser)
                flagcol = col + '_flag'
                fdf[flagcol] = flags
        except:
            logging.error ("Error in setting the flags, check if the flag_words file is present in config", exc_info=1)
        else:
            logging.info("Flaging responses completed")
        
    # Process keyphrase
    #kpCols = []    
    logging.info("Start keyphrase analysis..")
    kpUG = {}
    kpBG = {}
    kpDF = pd.read_csv(filepath, encoding="ISO-8859-1")
    #kpCols = kpuCols + kpbCols

    # PRocess unigram
    try:
        for col in kpuCols :

            ser = kpDF[col]
            if col in kpuCols:
                kpuObj = ug.Unigram()
                kpuObj.analyze(ser)
                kpus = kpuObj.get_keywords()
                kpUG[col] = kpus
    except:
        logging.error ("Error in Unigram keyphrase analysis", exc_info=1)
    else:
        if len(kpuCols) > 0:
            logging.info("Unigram keyphrase completed")
    
    # PRocess bigram
    try:

        for col in kpbCols:

            ser = kpDF[col]
            if col in kpbCols:
                kpbObj = bg.Bigram()
                kpbs = kpbObj.analyze(ser)
                kpBG[col] = kpbs
    except:
        logging.error ("Error in Bigram keyphrase analysis", exc_info=1)
    else:
        if len(kpbCols) > 0:
            logging.info("Bigram keyphrase completed")
    
    fileOP['id'] = fileId
    fileOP['sentopic'] = processedCols # sentiments, topic and flags are added to this 
    fileOP['flags'] = fdf
    fileOP['kpUG'] = kpUG
    fileOP['kpBG'] = kpBG
    
    # Add the processed file output to the totalOP
    logging.info("Added file %s output to overll output", name)
    totalOP[name] = fileOP
	
def writeOutput(totalOP):

    logging.info("Started writing the output..")
    for file in totalOP:

        print ("The filename is : ", file)

        fileContents = totalOP[file]
        fileId = fileContents['id']

        columnDetails = fileContents['sentopic']
        allFlagDetails = fileContents['flags']
                
        tmsaDf = pd.DataFrame()

        for colname in columnDetails: 
            print ("\nThe column is : ", colname)
            levels = columnDetails[colname]
            
            colFlag = allFlagDetails[[fileId, colname + '_flag']]
            allFlagDetails = allFlagDetails.drop([colname, colname + '_flag'], axis=1)
            
            if 'response' in levels:
                respDf = levels['response']
                if tmsaDf.empty:
                    tmsaDf = respDf
                else:
                    tmsaDf = pd.merge(tmsaDf, respDf, on=fileId, how="outer")
                    
                tmsaDf = pd.merge(tmsaDf, colFlag, on=fileId)  
                 
            
        # If there is any remaining column other than the ID column
        if len(list(allFlagDetails)) > 1:
            if tmsaDf.empty:
                tmsaDf = allFlagDetails
            else:
                tmsaDf = pd.merge(tmsaDf, allFlagDetails, on=fileId)  

        # Remove 'sentences' columns which we do not need
        delete_cols = [x for x in tmsaDf.columns if x.endswith('_sentences')]
        tmsaDf.drop(delete_cols, axis = 1, inplace = True)
   
        # If the file is already in the output, do not overwrite; create a new file with a different name
        opfile = 'output/' + os.path.splitext(file)[0] + '_processed.csv'
        if os.path.exists(opfile):
            cnt = 1
            while (os.path.exists('output/' + os.path.splitext(file)[0] + '_processed_' + str(cnt) + '.csv')):
                cnt = cnt + 1
            opfile = 'output/' + os.path.splitext(file)[0] + '_processed_' + str(cnt) + '.csv'
            tmsaDf.to_csv(opfile, encoding='ISO-8859-1')
        else:
            opfile = 'output/' + os.path.splitext(file)[0] + '_processed.csv'
            tmsaDf.to_csv(opfile, encoding='ISO-8859-1')
        
        logging.info("Output file %s written", opfile)
		
writeOutput(totalOP)

# Generate output pdfs ##############################
logging.info("Started writing the output..")

visualizeObj = viz.Viz(totalOP = totalOP, fileId = fileId)
visualizeObj.visualize()