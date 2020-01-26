import re
import nltk
import spacy
import numpy as np
import sklearn.neighbors
import pandas as pd

# TODO: move the functionality in this module into class(es), so that it can be more easily used as a dependency

# LOAD SPACY ENGLISH NLP MODEL
nlp = spacy.load('en_core_web_sm')


def get_question_features(question):
    """
    Method to extract features from each individual question.
    """
    features = {}

    # Extract the main verb from the question before additional processing
    main_verb = str(extract_main_verb(question))

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

    # ADD THE STEMMED MAIN VERB TO THE FEATURE SET WITH A WEIGHT OF 60
    stemmed_main_verb = porter_stemmer.stem(main_verb)
    features[stemmed_main_verb] = 60

    # TAG WORDS' PART OF SPEECH, AND ADD ALL WH WORDS TO FEATURE DICT
    # WITH WEIGHT 60
    words_pos = nltk.pos_tag(filtered_words)
    for word_pos in words_pos:
        if is_wh_word(word_pos[1]):
            features[word_pos[0]] = 60

    # ADD FIRST WORD AND NON-STOP WORDS TO FEATURE DICT
    filtered_words = [word for word in filtered_words if word not in nltk.corpus.stopwords.words('english')]
    for word in filtered_words:
            # ADD EACH WORD NOT ALREADY PRESENT IN FEATURE SET WITH WEIGHT OF 30
            if word not in features:
                features[word] = 30

    return features

# EXTRACTS THE MAIN VERB FROM A QUESTION USING THE DEPENDENCY TREE
# THE MAIN VERB OF THE QUESTION SHOULD BE THE ROOT OF THE TREE
# Note: this method of extracting the main verb is not perfect, but
# for single sentence questions that should have no ambiguity about the main verb,
# it should be sufficient.
def extract_main_verb(question):
    doc = nlp(question)
    sents = list(doc.sents)
    if len(sents) == 0:
        raise ValueError("Empty question")

    return sents[0].root


# The possible WH word tags returned through NLTK part of speech tagging
WH_WORDS = {'WDT', 'WP', 'WP$', 'WRB'}


def is_wh_word(pos):
    return pos in WH_WORDS


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
print(get_question_features("What are Foaad Khosmood's office hours?"))
print(get_question_features("Does Foaad Khosmood have office hours?"))
print(get_question_features("Who teaches CSC 480"))
print(get_question_features("CSC 480 is taught by who?"))
print(get_question_features("Khosmood teaches CSC 480?"))
print(get_question_features("Whose office hours are between 1 and 2 pm?"))
print(get_question_features("Where is Franz Kurfess' office?"))
print(get_question_features("This is a normal sentence."))
print(get_question_features("[COURSE] is taught by who?"))
print(get_question_features("How do I register for classes?"))

while True:
    test = input("Ask me a question: ")
    print(classify_question(test, features, classifier))

