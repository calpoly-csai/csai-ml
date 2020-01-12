import re
import nltk
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def get_question_features(question):
    features = {}
    matches = re.findall(r'(\[(.*?)\])', question)
    for match in matches:
        question = question.replace(match[0], '')
        features[match[0]] = 90
    question = re.sub('[^a-zA-Z0-9]', ' ', question)
    words = nltk.word_tokenize(question)
    words = [word.lower() for word in words if '[' and ']' not in word]
    porter_stemmer = nltk.stem.porter.PorterStemmer()
    filtered_words = [porter_stemmer.stem(word) for word in words]
    features[filtered_words[0]] = 30
    filtered_words = [word for word in filtered_words if word not in nltk.corpus.stopwords.words('english')]
    for word in filtered_words:
        features[word] = 30
    return features

def build_question_classifier():
    questions = pd.read_csv('questions.csv')
    questions['features'] = questions['questionFormat'].apply(get_question_features)
    question_features = questions['features'].values.tolist()

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
    test_features = get_question_features(test)
    test_vector = dict.fromkeys(overall_features, 0)
    for key in test_features:
        if key in test_vector:
            test_vector[key] = test_features[key]
        else:
            test_vector["not related"] += 250
    test_vector = np.array(list(test_vector.values()))
    test_vector = test_vector.reshape(1, len(test_vector))
    min_dist = np.min(classifier.kneighbors(test_vector, n_neighbors=1)[0])
    if min_dist > 150:
        return "I don't think that's a Statistics related question! Try asking something about the STAT curriculum."
    return classifier.predict(test_vector)[0]


test = "Who is teaching [COURSE_NUM]?"
classifier, overall_features = build_question_classifier()
match = classify_question(test, overall_features, classifier)
print(match)
