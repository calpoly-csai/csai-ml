import re
import nltk
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def get_question_features(question):
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
    porter_stemmer = nltk.stem.porter.PorterStemmer()
    filtered_words = [porter_stemmer.stem(word) for word in words]

    # ADD FIRST WORD AND NON-STOP WORDS TO FEATURE DICT
    features[filtered_words[0]] = 30
    filtered_words = [word for word in filtered_words if word not in nltk.corpus.stopwords.words('english')]
    for word in filtered_words:
        features[word] = 30
    return features

def build_question_classifier(questions):
    """
    Build overall feature set for each question based on feature vectors of individual questions.
    Train KNN classification model with overall feature set.
    """

    # READ QUESTIONS
    questions = pd.read_csv('question_set_clean.csv')
    questions['features'] = questions['questionFormat'].apply(get_question_features)
    question_features = questions['features'].values.tolist()

    # BUILD OVERALL FEATURE SET FROM INDIVIDUAL QUESTION FEATURE VECTORS
    overall_features = {}
    for feature in question_features:
        for key in feature:
            if key not in overall_features:
                overall_features[key] = 0
    overall_features["not related"] = 0
    vectors = []
    for feature in question_features:
        vector = dict.fromkeys(overall_features, 0)
        for key in feature:
            vector[key] = feature[key]
        vectors.append(np.array(list(vector.values())))

    y_train = questions['questionFormat']
    vectors = np.array(vectors)
    y_train = np.array(y_train)
    classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    classifier.fit(vectors, y_train)
    return classifier, overall_features

def classify_question(test, overall_features, classifier):
    """
    Match a user query with a question in the database based on the classifier we trained and overall features we calculated.
    Return relevant question.
    """
    test_features = get_question_features(test)
    test_vector = dict.fromkeys(overall_features, 0)
    for key in test_features:
        if key in test_vector:
            test_vector[key] = test_features[key]
        else:
            # IF A WORD IS NOT IN THE EXISTING FEATURE SET, IT MAY BE A QUESTION WE CANNOT ANSWER.
            test_vector["not related"] += 250
    test_vector = np.array(list(test_vector.values()))
    test_vector = test_vector.reshape(1, len(test_vector))
    min_dist = np.min(classifier.kneighbors(test_vector, n_neighbors=1)[0])
    if min_dist > 150:
        return "I don't think that's a Statistics related question! Try asking something about the STAT curriculum."
    return classifier.predict(test_vector)[0]




questions = pd.read_csv('question_set_clean.csv')
classifier, features = build_question_classifier(questions)
test = "Who teaches [COURSE]?"
print(classify_question(test, features, classifier))

