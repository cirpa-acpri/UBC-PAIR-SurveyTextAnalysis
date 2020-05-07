import logging
import yaml 
import pandas as pd
import numpy as np
import logging
import os
import shutil
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import fpdf
import re
import matplotlib
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PyPDF2 import PdfFileMerger, PdfFileReader

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')
nlp = spacy.load('en_core_web_sm')


class Viz():


    def __init__(self, totalOP, fileId):
        """Initialize an instance of a class"""

        self.totalOP = totalOP
        self.fileId = fileId




    def visualize(self):
        """Function to create all the plots and generate pdfs"""


        # Define list of topics
        topics_list = ['academics'
        , 'career'
        , 'commute'
        , 'diversity'
        , 'community'
        , 'extracurricular'
        , 'facilities'
        , 'finance'
        , 'housing'
        , 'wellness']



        for file1 in self.totalOP:
            
            logging.info("The visualization step has started....")
            logging.info('The filename is :{}'.format(file1))

            fileContents = self.totalOP[file1]
            
            for part in fileContents:

                # if part == 'id':
                #     fileId = fileContents['id']


                                
                # Flagged content print #############
                logging.info("The printing of flagged content has started....")
                
                if part == 'flags':

                    # Check if the output files already exists, don't overwrite, create new files
                    outflagfile = 'output/' + file1 + '_FLAG'
                    if os.path.exists(outflagfile + '.pdf'):
                        cnt = 1
                        while (os.path.exists(outflagfile + '_' + str(cnt) + '.pdf')):
                            cnt = cnt + 1
                        outflagfile = outflagfile + '_' + str(cnt) + '.pdf'
                    
                    else:
                        outflagfile = outflagfile + '.pdf'


                    
                    # Open a pdf
                    pdf = fpdf.FPDF(format='letter')

                    allFlagDetails = fileContents[part]

                    # New page
                    pdf.add_page()
        
                    # Set title
                    pdf.set_font("Arial", size=15, style = 'B')
                    pdf.write(5, 'Flag')
                    pdf.ln()
                    pdf.ln()
                    
                    colnames = allFlagDetails.columns.values
                    id = self.fileId
                
                    col = []
                    for text in colnames:
                        try: 
                            col.append(re.search('(.*?)_flag', str(text)).group(1))
                        except AttributeError:
                            pass           
        
                    for i in col:
                        pdf.set_font("Arial", size=11, style = 'B')
                        pdf.write(5, i)
                        pdf.ln()
                        df_temp = allFlagDetails[allFlagDetails[i+'_flag'] == True][[id, i]]
                        
                        #print(df_temp)
                        if df_temp.shape[0] != 0:
                            for j in range(df_temp.shape[0]):
                                # print(df_temp[id].iloc[j])
                                pdf.set_font("Arial", size=10)
                                pdf.write(5, str(df_temp[id].iloc[j]))
                                pdf.ln()
                                pdf.write(5, str(df_temp[i].iloc[j]))
                                pdf.ln(h = 7)
                        else:
                            pdf.write(5, "There is no flagged content in " + i + " question.")
                            pdf.ln(h = 7)
            
                    pdf.output(outflagfile)






                if part == 'sentopic':

                    columnDetails = fileContents[part]
                    
                    for colname in columnDetails: 

                        # Open the pdf pages for final plots
                        pp = PdfPages('tmp/' + colname + '_C_plots.pdf')


                        # Sentiment and topic related column names
                        tags_col_name = colname + '_tags'
                        sent_col_name = colname + '_sentiment'
                        sentiScore_col_name = colname + '_sentiScore'
                        cleaned_col_name = colname + '_sentences_cleaned'

                        
                        levels = columnDetails[colname]

                        for frame in levels:

                            # Response level data
                            if frame =='response':
                                respDf = levels[frame]
                                if sent_col_name not in respDf.columns:
                                    respSent = False
                                else:
                                    respSent = True 


                            # Sentence level data
                            if frame == 'sentence':
                                sentDf = levels[frame]
                                if sent_col_name not in sentDf.columns:
                                    sentSent = False
                                else:
                                    sentSent = True



                        logging.info("Visualization started for {} in {}".format(colname, file1))


                        # Plot 1 ############################

                        logging.info("Plotting the bar plot for topic distribution....")

                        # Topic distribution count for plotting a bar chart
                        topic_dict = {topic: str(respDf[tags_col_name]).count(topic) for topic in topics_list}
                        topic_dict = {k: v for k, v in sorted(topic_dict.items(), key = lambda x: x[1], reverse = True)}

                        # If topic doesn't exist in any response, don't show it in the plot
                        topic_dict = {k: v for k,v in topic_dict.items() if v > 0}


                        fig, ax = plt.subplots()

                        # Plot for Topic distribution
                        plt.bar(topic_dict.keys(), topic_dict.values(), color = '#002145', width = 0.7)
                        plt.xlabel('Topic', fontsize = 15)
                        plt.ylabel('Number of responses', fontsize = 15)
                        plt.xticks(list(topic_dict.keys()), fontsize = 12, rotation = 30)
                        plt.title('Topic Distribution', fontsize = 18)

                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        fig.set_size_inches(12, 8.27)

                        plt.tight_layout()
                        plt.savefig(pp, format='pdf')
                        plt.close()



                        if sentSent:

                            # Plot 2 ################################

                            logging.info("Plotting the stacked bar plot for topic-wise sentiment.....")

                            # Topic distribution count for plotting a bar chart
                            # topic_dict = {topic: str(sentDf[tags_col_name]).count(topic) for topic in topics_list}
                            # topic_dict = {k: v for k, v in sorted(topic_dict.items(), key = lambda x: x[1], reverse = True)}

                            # Create an empty dictionary for topic-wise sentiment
                            topic_sent_dict = {topic: [0, 0, 0] for topic in list(topic_dict.keys())}

                            # Calculation for the topic-wise sentiment plot
                            for topic in topic_dict.keys():                           
                                
                                for i in range(len(sentDf)):
                                    
                                    if topic in str(sentDf[tags_col_name][i]) and sentDf[sent_col_name][i] == 'positive':
                                        topic_sent_dict[topic][0] += 1 
                                        
                                    elif topic in str(sentDf[tags_col_name][i]) and sentDf[sent_col_name][i] == 'negative':
                                        topic_sent_dict[topic][1] += 1
                                        
                                    elif topic in str(sentDf[tags_col_name][i]) and sentDf[sent_col_name][i] == 'neutral':
                                        topic_sent_dict[topic][2] += 1 

                            # Repeat the same for percentages instead of absolute values                                   
                            topic_sent_dict_percent = {topic: [0, 0, 0] for topic in topic_sent_dict.keys()}

                            for k, v in topic_sent_dict.items():
                                if sum(v) > 0:
                                    topic_sent_dict_percent[k] = [round(x/sum(v)*100, 2) for x in v]
                                else:
                                    topic_sent_dict_percent[k] = [0, 0, 0]

                            # If topic doesn't exist in any response, don't show it in the plot
                            topic_sent_dict_percent = {k: v for k,v in topic_sent_dict_percent.items() if sum(v) > 0}

                            # Preparation of dictionary for stacked bar plot   
                            neg = []
                            pos = []
                            neu = []

                            for topic, sent in topic_sent_dict.items():
                                pos.append(topic_sent_dict[topic][0])
                                neg.append(topic_sent_dict[topic][1])
                                neu.append(topic_sent_dict[topic][2])

                            sent_all = {'positive': pos, 'negative': neg, 'neutral': neu}

                            # Repeat the same for percentages instead of absolute values
                            neg = []
                            pos = []
                            neu = []

                            for topic, sent in topic_sent_dict_percent.items():
                                neg.append(topic_sent_dict_percent[topic][0])
                                pos.append(topic_sent_dict_percent[topic][1])
                                neu.append(topic_sent_dict_percent[topic][2])

                            sent_all_percent = {'negative': pos, 'positive': neg, 'neutral': neu}


                            # Plotting objects
                            fig, ax = plt.subplots()
                            barWidth = 0.7
                            names = list(topic_sent_dict_percent.keys())

                            # UBC colors
                            colors = ["#002145", "#0055B7", "#00A7E1"]
                            r = list(range(len(list(topic_dict.keys()))))


                            # Create neutral Bars
                            plt.bar(r, sent_all_percent['neutral']
                                    , bottom = [i+j for i,j in zip(sent_all_percent['positive'], sent_all_percent['negative'])]
                                    , color = colors[2], width = barWidth, label = "Neutral")

                            # Create negative Bars
                            plt.bar(r, sent_all_percent['negative']
                                    , bottom = sent_all_percent['positive']
                                    , color = colors[1], width = barWidth, label = "Negative")

                            # Create positive Bars
                            plt.bar(r, sent_all_percent['positive']
                                    , color = colors[0], width = barWidth, label = "Positive")


                            # Custom x axis
                            plt.xticks(r, names, fontsize = 12, rotation = 30)
                            plt.xlabel("Topic", fontsize = 15)
                            plt.ylabel("Percentage of sentiment", fontsize = 15)
                            
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['bottom'].set_visible(False)
                            ax.spines['left'].set_visible(False)

                            # Add title
                            plt.title('Topic-wise Sentiment Distribution', fontsize = 18)

                            # Add a legend
                            plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

                            # Show graphic
                            fig.set_size_inches(11.69, 8.27)
                            plt.tight_layout()

                            plt.savefig(pp, format='pdf')
                            plt.close()




                        if sentSent:
                            # Plot 3 - 2 Overall sentiment Word Clouds ######################

                            logging.info("Generating word clouds....")

                            # Overall column - Positive sentiment 
                            dt_pos = ' '.join(sentDf.loc[(sentDf[sent_col_name] == 'positive') & (sentDf[sentiScore_col_name] >= 0.8)][cleaned_col_name])
                        
                            # Remove some stopwords
                            stop_words =['become', 'need', 'ubc', 'want', 'student', 'campus', 'people', 'year', 'way']
                            for word in stop_words:
                                dt_pos = dt_pos.replace(word, '')

                            if dt_pos != '':
                                # Generate word cloud
                                sent = nlp(dt_pos)
                                selected_words = []
                                for token in sent:
                                # Store words only with cadidate POS tag
                                    if token.pos_ == 'NOUN' and token.is_stop is False:
                                         selected_words.append(token.text)
                                dt_pos = ' '.join(selected_words)
                                
                                wordcloud = WordCloud(max_font_size = 60, max_words = 80, background_color = "white").generate(dt_pos)
                                plt.figure()
                                plt.title(colname + ' - Positive')
                                plt.imshow(wordcloud, interpolation="bilinear")
                                plt.axis("off")
                                plt.savefig(pp, format='pdf')


                            # Overall column - Negative sentiment 
                            dt_neg = ' '.join(sentDf.loc[(sentDf[sent_col_name] == 'negative') & (sentDf[sentiScore_col_name] <= -0.8)][cleaned_col_name])
                            
                            # Remove some stopwords
                            stop_words = ['become', 'need', 'ubc', 'want', 'student', 'campus', 'people', 'year', 'way']
                            for word in stop_words:
                                dt_neg = dt_neg.replace(word, '')

                            if dt_neg != '':
                                # Generate word cloud
                                sent = nlp(dt_neg)
                                selected_words = []
                                for token in sent:
                                # Store words only with cadidate POS tag
                                    if token.pos_ == 'NOUN' and token.is_stop is False:
                                         selected_words.append(token.text)
                                dt_neg = ' '.join(selected_words)
                                wordcloud = WordCloud(max_font_size = 60, max_words = 80, background_color = "white").generate(dt_neg)
                                plt.figure()
                                plt.title(colname + ' - Negative')
                                plt.imshow(wordcloud, interpolation = "bilinear")
                                plt.axis("off")
                                plt.savefig(pp, format='pdf')



                        

                        # Plot 4 - Topic-wise Word Clouds ######################

                        for topic in list(topic_dict.keys()):
                        
                            # Consider only strong sentiment sentiment containing each topic
                            sentDf['topic_count_temp'] = sentDf[tags_col_name].apply(lambda x: str(x).count(topic))
                            dt_temp = sentDf.loc[sentDf['topic_count_temp'] > 0]
                            
                            if sentSent:
                                dt_topic = ' '.join(dt_temp.loc[abs(dt_temp[sentiScore_col_name]) >= 0.5][cleaned_col_name])
                            else:
                                dt_topic = ' '.join(dt_temp[cleaned_col_name])

                            
                            # Remove some stopwords
                            stop_words =['become', 'need', 'ubc', 'want', 'student', 'campus', 'people', 'year', 'really', 'year', 'way', topic]
                            for word in stop_words:
                                dt_topic = dt_topic.replace(word, '')
                            
                            if dt_topic != '':
                            
                                sent = nlp(dt_topic)
                                selected_words = []
                                for token in sent:
                                # Store words only with cadidate POS tag
                                    if (token.pos_ == 'NOUN' or token.pos_ == 'ADJ') and token.is_stop is False:
                                         selected_words.append(token.text)
                                dt_topic = ' '.join(selected_words)
                                wordcloud = WordCloud(max_font_size = 60, max_words = 80, background_color = "white").generate(dt_topic)
                                plt.figure()
                                plt.title(topic)
                                plt.imshow(wordcloud, interpolation = "bilinear")
                                plt.axis("off")
                                plt.savefig(pp, format = 'pdf')


                        # Close pdf pages   
                        pp.close()
                
                    
                    
                    

                        if respSent:

                            logging.info("Generating a pdf for sentiment-topic wise top 10 responses....")

                            # To calculate number of words
                            respDf['word_count'] = respDf[colname].apply(lambda x: len(str(x).split(' ')) if x != '' else 0)


                            # Open a pdf
                            pdf = fpdf.FPDF(format = 'letter')

                            for topic in list(topic_dict.keys()):                            
                                                        
                                # Extract top 10 negative sentences
                                dt_temp = respDf.loc[(respDf[sent_col_name] == 'negative') & respDf[tags_col_name].str.contains(topic, na = False, regex = False)]

                                if dt_temp.shape[0] != 0:
                                    l = dt_temp[dt_temp['word_count'] > 15].sort_values([sentiScore_col_name], ascending=[1])
                                    l = l[[self.fileId, colname]]

                                    if l.shape[0] < 15:
                                        l = l
                                    else:
                                        l = l.iloc[:15]   

                                    if l.shape[0] != 0: 

                                        # New page
                                        pdf.add_page()
                                        pdf.set_font("Arial", size = 15, style = 'B')
                                                                
                                        pdf.write(5, topic + '- Negative')
                                        pdf.ln()
                                                                        
                                        pdf.set_font("Arial", size = 10) 

                                        for i in range(l.shape[0]):
                                            pdf.write(5, str(l[self.fileId].iloc[i]))
                                            pdf.ln()
                                            pdf.write(5, str(l[colname].iloc[i]))
                                            pdf.ln()
                                            pdf.ln()   
        
                        
                            
                                # Extract top 10 postive sentences
                                dt_temp = respDf.loc[(respDf[sent_col_name] == 'positive') & respDf[tags_col_name].str.contains(topic, na = False, regex = False)]
                                
                                if dt_temp.shape[0] != 0:
                                    l = dt_temp[dt_temp['word_count'] > 15].sort_values([sentiScore_col_name], ascending=[0])
                                    l = l[[self.fileId, colname]]
                                    
                                    if l.shape[0] < 15:
                                        l = l
                                    else:
                                        l = l.iloc[:15]
                
                                    if l.shape[0] != 0:

                                        # Next page    
                                        pdf.add_page()
                                        pdf.set_font("Arial", size = 15, style = 'B')
                    
                                        pdf.write(5, topic + '- Positive')
                                        pdf.ln()

                                        pdf.set_font("Arial", size = 10)

                                        for i in range(l.shape[0]):
                                            pdf.write(5, str(l[self.fileId].iloc[i]))
                                            pdf.ln()
                                            pdf.write(5, str(l[colname].iloc[i]))
                                            pdf.ln()
                                            pdf.ln()

                            pdf.output('tmp/' + colname + '_D_top10.pdf')

                        
                        else:

                            logging.info("Generating a pdf for sentiment-topic wise top 10 responses....")

                            # To calculate number of words
                            respDf['word_count'] = respDf[colname].apply(lambda x: len(str(x).split(' ')) if x != '' else 0)


                            # Open a pdf
                            pdf = fpdf.FPDF(format = 'letter')

                            for topic in list(topic_dict.keys()):                            
                                                        
                                # Extract top 10 negative sentences
                                dt_temp = respDf[respDf[tags_col_name].str.contains(topic, na = False, regex = False)]

                                if dt_temp.shape[0] != 0:
                                    l = dt_temp.sort_values(['word_count'], ascending=[0])[[self.fileId, colname]]
                                    
                                    if l.shape[0] < 15:
                                        l = l
                                    else:
                                        l = l.iloc[:15]     

                                    if dt_temp.shape[0] != 0: 

                                        # New page
                                        pdf.add_page()
                                        pdf.set_font("Arial", size = 15, style = 'B')
                                                                
                                        pdf.write(5, topic)
                                        pdf.ln()
                                                                        
                                        pdf.set_font("Arial", size = 10) 

                                        for i in range(l.shape[0]):
                                            pdf.write(5, str(l[self.fileId].iloc[i]))
                                            pdf.ln()
                                            pdf.write(5, str(l[colname].iloc[i]))
                                            pdf.ln()
                                            pdf.ln()
      
    

                            pdf.output('tmp/' + colname + '_D_top10.pdf')





                # Key-phrase Unigram Visualization plot #############
                logging.info("The KP unigram visualization has started....")

                if part == 'kpUG':

                    kpUgDetails = fileContents[part]
                    for column in kpUgDetails: 

                        opfile = 'tmp/'+ column + '_A_kp_Uni.pdf'

                        with PdfPages(opfile) as export_pdf:
                            fig, ax = plt.subplots()
                            plt.bar(kpUgDetails[column].keys(), kpUgDetails[column].values(), color = '#002145', width = 0.7)
                            plt.xlabel('Keyword', fontsize = 15)
                            plt.ylabel('Importance Score', fontsize = 15)
                            plt.xticks(list(kpUgDetails[column].keys()), fontsize = 12, rotation = 30)
                            plt.title('Keyword Extraction', fontsize = 18)

                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['bottom'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            fig.set_size_inches(12, 8.27)

                            plt.tight_layout()
                            export_pdf.savefig()
                            plt.close()



                # Key-phrase Bigram Visualization plot #############
                logging.info("The KP bigram visualization has started....")

                if part == 'kpBG':
                    kpBgDetails = fileContents[part]
                    # print(kpBgDetails)
                    for column in kpBgDetails: 

                        new = []

                        opfile = 'tmp/'+ column + '_B_kp_Bi.pdf'

                        with PdfPages(opfile) as export_pdf:
                            fig, ax = plt.subplots()
                            #print(list(kpBgDetails[column].keys()))
                            for i in kpBgDetails[column].keys():
                                #print(list(i))
                                new.append(' '.join(list(i)))
                            #print(new)

                            plt.bar(new, kpBgDetails[column].values(), color = '#002145', width = 0.7)
                            plt.xlabel('Key-phrase', fontsize = 15)
                            plt.ylabel('Frequency', fontsize = 15)
                            plt.xticks(new, fontsize = 12, rotation = 30)
                            plt.title('Key-phrase Extraction', fontsize = 18)

                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['bottom'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            fig.set_size_inches(12, 8.27)

                            plt.tight_layout()
                            export_pdf.savefig()
                            plt.close()




            # Clean up 'tmp/' folder and merge the pdf files
            logging.info("Cleaning up tmp folder and merging files related to each question....")
            allcol = []
            for part in fileContents:

                if part == "sentopic":
                    columnDetails = fileContents[part]
                    allcol.append(list(columnDetails.keys()))

                if part == "kpUG":
                    columnDetails = fileContents[part]
                    allcol.append(list(columnDetails.keys()))

                if part == "kpBG":
                    columnDetails = fileContents[part]
                    allcol.append(list(columnDetails.keys()))

            # All columns given 
            allcol = [x for y in allcol for x in y]
            allcol = list(set(allcol)) # Some columns might be repeated 

            # Find files related to each column
            for col in allcol:

                pdfs = []
                listdir = os.listdir('tmp')

                for ffile in listdir:
                    if col in ffile:
                        pdfs.append(ffile)

                
                # Check if the output files already exists, don't overwrite, create new files
                outfile = 'output/' + file1 + '_' + col + '_RESULT'
                if os.path.exists(outfile + '.pdf'):
                    cnt = 1
                    while (os.path.exists(outfile + '_' + str(cnt) + '.pdf')):
                        cnt = cnt + 1
                    outfile = outfile + '_' + str(cnt) + '.pdf'
                
                else:
                    outfile = outfile + '.pdf'
                
                if len(pdfs) > 0:

                    pdfs.sort() # Order plots before text

                    # Merge the pdf files related to one question 
                    merger = PdfFileMerger()

                    for colpdf in pdfs:
                        colpdf = 'tmp/' + colpdf
                        merger.append(colpdf)

                    merger.write(outfile)
                    merger.close()

            
            # Remove the tmp folder after all the output files are generated
            shutil.rmtree('tmp')


        logging.info('Done!!')





            










                                


















                                        



                                                

                                                

                                                
                                                

                                            





















