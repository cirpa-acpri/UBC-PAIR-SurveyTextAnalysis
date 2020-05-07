import gensim
import logging
import pandas as pd
import math
import nltk
import sys
import os
import numpy as np
import cleaning.cleaning as cln

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)


class topic_tagging():


    """Tag the topics to responses using word2vec model"""
    def __init__(self, path_full_data, path_data_col, id_col_name, col_name, typo_ind = False):
        
        # Creating the cleaning object
        self.cln = cln.Cleaning()

        # Previous data for training
        self.path_full_data = path_full_data


        # Current data - path, ID and data column
        self.path_data_col = path_data_col
        self.id_col_name = id_col_name
        self.col_name = col_name

        
        # Option to fix typos
        self.typo_ind = typo_ind



    def sent_to_words(self, sentences):
        """Simple pre-processing of sentences"""

        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence)))

    
    def getValid(self, column):
        """Removes useless sentences"""

        invalids = ['nope', 'none', 'no comments', 'no comment', 'no thanks', 'no thank you', 'none thanks', 'none thank you', 'no i don\'t', 'no i do not']
        column = column.apply(lambda x: np.nan if (x is np.nan or len(x) < 4 or x.lower() in invalids) else x)
        return column



    
    
    def removeEmptyData(self, data):
        """Remove rows with NA values"""
        
        data = data.dropna()
        data = data.reset_index(drop =True)
        return data



    def getSentences(self, ser):
        """Split a response into sentences"""

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentencesSer = ser.apply(tokenizer.tokenize)
        return sentencesSer




    def train_word2vec(self, size = 50, window = 20, min_count = 5, epochs = 40):

        """Trains the word2vec model on the previous years data combined with the column"""


        # Read the entire previous data for training
        full_data = pd.read_csv(self.path_full_data, encoding = "ISO-8859-1")

        # Also read the column which we are performing analysis for
        col_data = pd.read_csv(self.path_data_col
                                , encoding = "ISO-8859-1"
                                , usecols = [self.id_col_name, self.col_name])
        

        # Clean the data in the column
        col_data[self.col_name] = self.cln.clean(col_data[self.col_name], typo = self.typo_ind)
        col_data.replace(np.nan, '', inplace = True)
        col_name_list = list(col_data[self.col_name].apply(lambda x: str(x).split(' ')))


        # Make a list of lists of the data
        input_list = list(full_data['response'].apply(lambda x: x.split(' ')))
        input_list = input_list + col_name_list

        # Remove the responses having only one or two words
        input_list = [x for x in input_list if len(x) > 1]

        # Build vocabulary and train model
        model = gensim.models.Word2Vec(
            input_list,
            size = size,
            window = window,
            min_count = min_count)

        model.train(input_list, total_examples = len(input_list), epochs = epochs)

        return model






    def prepare_lexicons(self, topnwords = 80, distance_cutoff = 0.45):

        """Assigns topics after spliting the responses into sentences"""

        model = self.train_word2vec()


        # 10 topics
        topic_dict = {0: 'academics'
                        , 1: 'career'
                        , 2: 'commute'
                        , 3: 'diversity'
                        , 4: 'community'
                        , 5: 'extracurricular'
                        , 6: 'facilities'
                        , 7: 'finance'
                        , 8: 'housing'
                        , 9: 'wellness'
                        }

        # Some important words that should be included under each topic
        topics = [['academic', 'exam', 'study', 'learn', 'education', 'class', 'course', 'grade', 'assignment'
            , 'degree', 'research', 'elective'
            , 'professor', 'project', 'scholarship', 'knowledge']
            , ['career', 'job', 'coop', 'employment']
            , ['commute', 'skytrain', 'transport', 'commuter']
            , ['diversity', 'diverse', 'background']
            , ['community', 'welcome', 'support', 'social', 'friend', 'fun', 'network', 'home']
            , ['extracurricular', 'club', 'sport', 'activity']
            , ['facility', 'infrastructure', 'food', 'building', 'gym']
            , ['finance', 'tuition', 'expensive']
            , ['housing', 'live', 'residence']
            , ['wellness', 'health', 'stress', 'depression', 'anxiety']]

        # For each topic, collect the words most similar to them in a list of lists
        topic_lexicons = []

        # Loop through the ten topics
        for topic in topics:

            temp_words = []

            # Loop through each word that we have given manually under each topic
            for word in topic:

                # Consider most similar words according to some cutoffs
                similar_words = model.wv.most_similar(positive = word, topn = topnwords)
                temp_words1 = [x for (x,y) in similar_words if y >= distance_cutoff]

                temp_words = temp_words + temp_words1

            temp_words = temp_words + topic


            # Take unique words, there might be duplicates
            topic_lexicons.append(list(set(temp_words)))

        # Some manual adjustments
        # Remove 'commute' from other topic
        topic_lexicons[8].remove('commute')

        return topic_lexicons






    def assign_topics_to_sentences(self):
        """ Assign tags to single sentences"""
        

        # 10 topics
        topic_dict = {0: 'academics'
                        , 1: 'career'
                        , 2: 'commute'
                        , 3: 'diversity'
                        , 4: 'community'
                        , 5: 'extracurricular'
                        , 6: 'facilities'
                        , 7: 'finance'
                        , 8: 'housing'
                        , 9: 'wellness'
                        }

        # Some important words that should be included under each topic
        topics = [['academic', 'exam', 'study', 'learn', 'education', 'class', 'course', 'grade', 'assignment'
            , 'degree', 'research', 'elective'
            , 'professor', 'project', 'scholarship', 'knowledge']
            , ['career', 'job', 'coop', 'employment']
            , ['commute', 'skytrain', 'transport', 'commuter']
            , ['diversity', 'diverse', 'background']
            , ['community', 'welcome', 'support', 'social', 'friend', 'fun', 'network', 'home']
            , ['extracurricular', 'club', 'sport', 'activity']
            , ['facility', 'infrastructure', 'food', 'building', 'gym']
            , ['finance', 'tuition', 'expensive']
            , ['housing', 'live', 'residence']
            , ['wellness', 'health', 'stress', 'depression', 'anxiety']]

        # Read the data - id and reponse column
        dt = pd.read_csv(self.path_data_col
                         , encoding = "ISO-8859-1"
                         , usecols = [self.id_col_name, self.col_name])


        
        # Remove rows with NA values
        dt = self.removeEmptyData(dt)
        
        # Split into sentences
        dt['sentences'] = self.getSentences(dt[self.col_name])
        
        
        

        
        

        # Store number of sentences in each response as a column
        dt['num_sent'] = dt['sentences'].apply(lambda x: len(x))

        # Split each row into multiple rows - one row for each sentence
        dt = (dt
         .set_index([self.id_col_name, self.col_name, 'num_sent'])['sentences']
         .apply(pd.Series)
         .stack()
         .reset_index()
         .drop('level_3', axis = 1)
         .rename(columns = {0:'sentences'}))


        # Clean the sentences
        dt['sentences_cleaned'] = self.cln.clean(dt['sentences'], typo = self.typo_ind)

        # Remove useless sentences
        dt['sentences_cleaned'] = self.getValid(dt['sentences_cleaned'])

        # Remove rows with NA values
        dt = self.removeEmptyData(dt)

        # Tokenize words in the cleaned sentences
        responses = list(self.sent_to_words(dt['sentences_cleaned'].values.tolist()))


        # Call the lexicon function
        topic_lexicons = self.prepare_lexicons()

        # Lists to store results
        count_topic_all = []
        actual_topic_all = []

        # Tag each response into a topic
        for response in responses:

            count_topic = []
            actual_topic = []

            for topic in topic_lexicons:

                # Count occurance of each word in word stock in the response
                temp = sum(dict((x, response.count(x)) for x in topic).values())
                count_topic.append(temp)


            for index, value in enumerate(count_topic):

                # Consider the topic if atleast one(?) word from its word-stock occurs in the response
                if value > 0:
                    actual_topic.append(topic_dict[index])


            # If more than 3 topics are tagged for single sentence, refine by increasing
            # cutoff to at least 2 words instead of 1
            if len(actual_topic) > 3:

                actual_topic = []
                for index, value in enumerate(count_topic):

                    if value > 1: # Increase cutoff
                        actual_topic.append(topic_dict[index])

            count_topic_all.append(count_topic)
            actual_topic_all.append(actual_topic)


        dt['tags'] = actual_topic_all
        dt['num_tags'] = count_topic_all


        # Select only the most important columns
        dt_less = dt[[self.id_col_name, 'sentences', 'tags']]

        return dt, dt_less






    def sentence_to_response_level_tags(self):
        """Converts the sentence level tagging to response level"""


        # Load the dataset which has topic tags for each sentence
        dt, dt_less = self.assign_topics_to_sentences()
        
        # Minimum number of tags needed to tag the overall response
        dt['min_num_tags'] = dt['num_sent'].apply(lambda x: math.ceil(0.3*x))
        
        # Final dataset with full survey response and its tags
        final_dt = dt.groupby(self.id_col_name).agg({'tags': sum
                                                , 'num_sent': min
                                                , 'min_num_tags': min
#                                                 , 'sentences': lambda x: "%s" % '. '.join(x)
                                                , self.col_name: min})
        final_dt.reset_index(level = 0, inplace = True)
        final_dt['topics'] = final_dt.apply(lambda x: set([i for i in x.tags if x.tags.count(i) >= x.min_num_tags])
                                            , axis = 1)

        final_dt_less = final_dt[[self.id_col_name, self.col_name, 'topics']]

        return final_dt, final_dt_less
    


 
    
    
    def assign_topics(self):
        """Combine last two functions to avoid running twice"""

        # Load the dataset which has topic tags for each sentence
        dt, dt_less = self.assign_topics_to_sentences()
        
        dt_copy = dt
        
        # Minimum number of tags needed to tag the overall response
        dt['min_num_tags'] = dt['num_sent'].apply(lambda x: math.ceil(0.3*x))
        
        # Final dataset with full survey response and its tags
        final_dt = dt.groupby(self.id_col_name).agg({'tags': sum
                                                , 'num_sent': min
                                                , 'min_num_tags': min
#                                                 , 'sentences': lambda x: "%s" % '. '.join(x)
                                                , self.col_name: min})
        final_dt.reset_index(level = 0, inplace = True)
        final_dt['topics'] = final_dt.apply(lambda x: set([i for i in x.tags if x.tags.count(i) >= x.min_num_tags])
                                            , axis = 1)

        final_dt_less = final_dt[[self.id_col_name, self.col_name, 'topics']]

        return dt_copy, dt_less, final_dt, final_dt_less

