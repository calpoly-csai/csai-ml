import re
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import re
import sklearn.neighbors
import spacy
import sys

from datetime import datetime

from google.api_core.client_options import ClientOptions
from google.cloud import automl_v1
from google.cloud.automl_v1.proto import service_pb2

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.externals import joblib

from os import listdir
from os.path import isfile, join

class NIMBUS_NLP:

    @staticmethod
    def predict_question(input_question):
        '''
        Runs through variable extraction and the question classifier to
        predict the intended question.

        Args: input_question (string) - user input question to answer

        Return: return_tuple (tuple) - contains the user's input question,
                                       the variable extracted input question,
                                       the entity extracted, and the predicted
                                       answer

        '''

        variable_extraction = Variable_Extraction()
        normalized_sentence, entity = variable_extraction.\
                                        extract_variables(input_question)
        
        classifier = Question_Classifier(save_model=False)
        answer = classifier.classify_question(normalized_sentence)

        return_tuple = (input_question, normalized_sentence,
                        entity, answer)

        return return_tuple

class Variable_Extraction:
    def __init__(self):
        self.model_name = None # replace with the project model id
        
        credential_path = None # replace with the path to the credential json
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
        self.entity = ""

    def inline_text_payload(self, sent):
        '''
        Converts the input sentence into GCP's callable format

        Args: sent (string) - input sentence

        Return: (dict) - GCP NER input format

        '''

        return {'text_snippet': {'content': sent, 'mime_type': 'text/plain'} }

    def get_prediction(self, sent):
        '''
        Obtains the prediction from the input sentence and returns the
        normalized sentence

        Args: sent (string) - input sentence

        Return: request (PredictObject) - predictiton output
        ''' 
        
        params = {}
        
        # Setup API 
        options = ClientOptions(api_endpoint='automl.googleapis.com')
        
        # Create prediction object
        predictor = automl_v1.PredictionServiceClient(client_options=options)

        # Format input sentence
        payload = self.inline_text_payload(sent)
        
        # Make prediction API call
        request = predictor.predict(self.model_name, payload, params)

        # Return the output of the API call
        return request

    def extract_variables(self, sent):
        '''
        Takes the prediction and replaces the entity with its corresponding tag

        Args: sent (string) - input sentence

        Return: (tuple) - (normalized sentence, entity) 

        '''

        # Make the prediction
        request = self.get_prediction(sent)

        # Obtain the entity in the sentence
        self.entity = request.payload[0].text_extraction.text_segment.content 
        
        # Obtain the predicted tag 
        tag = request.payload[0].display_name
        
        return sent.replace(self.entity, '[' + tag + ']'), self.entity


#TODO: Add the Question_Classifier code directly into this file
class Question_Classifier:

    def __init__(self, question_data_file_name=None, save_model=False):
        self.save_model = save_model
        self.nlp = spacy.load('en_core_web_sm')
        if(question_data_file_name):
            self.questions = pd.read_csv(question_data_file_name)
        # The possible WH word tags returned through NLTK part of speech tagging
        self.WH_WORDS = {'WDT', 'WP', 'WP$', 'WRB'}
        self.overall_features = {}
        self.classifier = None
        self.use_new = True

        if(save_model == True):
            self.build_question_classifier()
        else:
            self.load_latest_classifier()

    @staticmethod
    def save_model(model, model_name):
        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
        now = datetime.now()
        date_time = now.strftime("_%m_%d_%Y_%H_%M_%S")
        save_path = PROJECT_DIR + '/models/' + model_name + date_time + '.pkl'
        f = open(save_path, 'wb')
        pickle.dump(model, f)
        f.close()
        print('Saved model :', save_path)

    @staticmethod
    def load_model(model_name):
        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
        train_path = PROJECT_DIR + '/models/' + model_name + '.joblib'
        return joblib.load(train_path)

    @staticmethod
    def load_latest_model():
        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
        train_path = PROJECT_DIR + '/models/'
        onlyfiles = [f for f in listdir(train_path) if isfile(join(train_path, f))]
        r = [(f, datetime.strptime(re.findall(r'([\d]+_[\d]+_[\d]+_[\d]+_[\d]+_[\d]+)', f)[0], '%m_%d_%Y_%H_%M_%S')) for f in onlyfiles]
        r = sorted(r, key=lambda x: x[1])
        model_path = r[-1][0]
        return joblib.load(train_path + model_path)

    def load_latest_classifier(self):
        self.classifier = self.load_latest_model()

    def get_question_features(self, question):
        # print("using new algorithm")
        """
        Method to extract features from each individual question.
        """
        features = {}

        # Extract the main verb from the question before additional processing
        main_verb = str(self.extract_main_verb(question))

        # ADD ALL VARIABLES TO THE FEATURE DICT WITH A WEIGHT OF 90
        matches = re.findall(r'(\[(.*?)\])', question)
        for match in matches:
            question = question.replace(match[0], '')
            features[match[0]] = 90

        question = re.sub('[^a-zA-Z0-9]', ' ', question)

        # PRE-PROCESSING: TOKENIZE SENTENCE, AND LOWER AND STEM EACH WORD
        words = nltk.word_tokenize(question)
        words = [word.lower() for word in words if '[' and ']' not in word]

        filtered_words = self.get_lemmas(words)

        # ADD THE LEMMATIZED MAIN VERB TO THE FEATURE SET WITH A WEIGHT OF 60
        stemmed_main_verb = self.nlp(main_verb)[0]
        features[stemmed_main_verb] = 60

        # TAG WORDS' PART OF SPEECH, AND ADD ALL WH WORDS TO FEATURE DICT
        # WITH WEIGHT 60
        words_pos = nltk.pos_tag(filtered_words)
        for word_pos in words_pos:
            if self.is_wh_word(word_pos[1]):
                features[word_pos[0]] = 60

        # ADD FIRST WORD AND NON-STOP WORDS TO FEATURE DICT
        filtered_words = [
            word for word in filtered_words if word not in nltk.corpus.stopwords.words('english')]
        for word in filtered_words:
            # ADD EACH WORD NOT ALREADY PRESENT IN FEATURE SET WITH WEIGHT OF 30
            if word not in features:
                features[word] = 30

        return features

    def get_question_features_old_algorithm(self, question):
        print("using old algorithm....")
        """
            Method to extract features from each individual question.
            """
        features = {}

        # ADD ALL VARIABLES TO THE FEATURE DICT WITH A WEIGHT OF 90
        matches = re.findall(r'(\[(.*?)\])', question)
        for match in matches:
            question = question.replace(match[0], '')
            features[match[0]] = 90
        question = re.sub('[^a-zA-Z0-9]', ' ', question)

        # PRE-PROCESSING: TOKENIZE SENTENCE, AND LOWER AND STEM EACH WORD
        words = nltk.word_tokenize(question)
        words = [word.lower() for word in words if '[' and ']' not in word]
        filtered_words = self.get_lemmas(words)

        # ADD FIRST WORD AND NON-STOP WORDS TO FEATURE DICT
        features[filtered_words[0]] = 60
        filtered_words = [
            word for word in filtered_words if word not in nltk.corpus.stopwords.words('english')]
        for word in filtered_words:
            features[word] = 30

        return features

    # EXTRACTS THE MAIN VERB FROM A QUESTION USING THE DEPENDENCY TREE
    # THE MAIN VERB OF THE QUESTION SHOULD BE THE ROOT OF THE TREE
    # Note: this method of extracting the main verb is not perfect, but
    # for single sentence questions that should have no ambiguity about the main verb,
    # it should be sufficient.
    def extract_main_verb(self, question):
        doc = self.nlp(question)
        sents = list(doc.sents)
        if len(sents) == 0:
            raise ValueError("Empty question")

        return sents[0].root

    def get_lemmas(self, words):
        return [self.nlp(word)[0].lemma_ for word in words]

    def is_wh_word(self, pos):
        return pos in self.WH_WORDS

    def build_question_classifier(self):
        """
        Build overall feature set for each question based on feature vectors of individual questions.
        Train KNN classification model with overall feature set.
        """

        # READ QUESTIONS
        questions = pd.read_csv('question_set_clean.csv')
        questions['features'] = questions['questionFormat'].apply(self.get_question_features)
        # old alg: questions['features'] = questions['questionFormat'].apply(self.get_question_features_old_algorithm)

        question_features = questions['features'].values.tolist()

        # BUILD OVERALL FEATURE SET FROM INDIVIDUAL QUESTION FEATURE VECTORS
        for feature in question_features:
            for key in feature:
                if key not in self.overall_features:
                    self.overall_features[key] = 0
        self.overall_features["not related"] = 0
        vectors = []
        for feature in question_features:
            vector = dict.fromkeys(self.overall_features, 0)
            for key in feature:
                vector[key] = feature[key]
            vectors.append(np.array(list(vector.values())))

        y_train = questions['questionFormat']
        vectors = np.array(vectors)
        y_train = np.array(y_train)
        self.classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        self.classifier.fit(vectors, y_train)
        if (self.save_model == True):
            self.save_model(self.classifier, "nlp-model")

    def filterWHTags(self, question):
        # ADD ALL VARIABLES TO THE FEATURE DICT WITH A WEIGHT OF 90
        matches = re.findall(r'(\[(.*?)\])', question)
        for match in matches:
            question = question.replace(match[0], '')

        question = re.sub('[^a-zA-Z0-9]', ' ', question)

        # PRE-PROCESSING: TOKENIZE SENTENCE, AND LOWER AND STEM EACH WORD
        words = nltk.word_tokenize(question)
        words = [word.lower() for word in words if '[' and ']' not in word]

        filtered_words = self.get_lemmas(words)

        question_tags = nltk.pos_tag(filtered_words)
        question_tags = [
            tag for tag in question_tags if self.is_wh_word(tag[1])]
        return question_tags

    def validate_WH(self, test_question, predicted_question):
        """
        Assumes that only 1 WH word exists
        Returns True if the WH word in the test question equals the
        WH word in the predicted question
        """

        test_tags = self.filterWHTags(test_question)
        predicted_tags = self.filterWHTags(predicted_question)

        # Uncomment these lines below to see
        # print("Test")
        # print(test_tags)
        # print()

        # print("Predicted")
        # print(predicted_tags)
        # print()

        # Compares all WH words in the tags array and returns False if one doesn't match
        min_tag_len = min(len(test_tags), len(predicted_tags))
        wh_match = True
        i = 0
        while (wh_match and i < min_tag_len):
            wh_match = wh_match and (test_tags[i][0] == predicted_tags[i][0])
            i += 1
        return wh_match

    def classify_question(self, test_question):
        """
        Match a user query with a question in the database based on the classifier we trained and overall features we calculated.
        Return relevant question.
        """
        if self.classifier is None:
            raise ValueError("Classifier not initialized")

        if self.use_new:
            test_features = self.get_question_features(test_question)
        else:
            test_features = self.get_question_features_old_algorithm(
                test_question)
        
        print(test_features)
            
        test_vector = dict.fromkeys(self.overall_features, 0)
        
        for key in test_features:
            if key in test_vector:
                test_vector[key] = test_features[key]
            else:
                # IF A WORD IS NOT IN THE EXISTING FEATURE SET, IT MAY BE A QUESTION WE CANNOT ANSWER.
                test_vector["not related"] += 250
        test_vector = np.array(list(test_vector.values()))
        test_vector = test_vector.reshape(1, len(test_vector))
        min_dist = np.min(self.classifier.kneighbors(
            test_vector, n_neighbors=1)[0])
        if min_dist > 150:
            return "I don't think that's a Statistics related question! Try asking something about the STAT curriculum."

        predicted_question = self.classifier.predict(test_vector)[0]

        wh_words_match = self.validate_WH(test_question, predicted_question)
        # Uncomment to print whether the WH words match
        # print("WH Words Match?:", wh_words_match)

        if (not wh_words_match):
            return "WH Words Don't Match"

        return predicted_question

if __name__ == '__main__':
    while True:
        question = input("Enter a question: ")
        answer = NIMBUS_NLP.predict_question(question)
        print(answer)
