import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)

questions = []
file1 = open("questions.txt","r")
for i in file1.readlines():
    questions.append(i.split("|")[1].strip())

print(questions)

asked = ["How many units do I need?"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions) # remove stop words
clf = MultinomialNB().fit(X, questions)
knn = KNeighborsClassifier(n_neighbors=1).fit(X, questions)
Y = vectorizer.transform(asked)
predicted = clf.predict(Y)
print(predicted)
predicted = knn.predict(Y)
print(predicted)