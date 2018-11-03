import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Natural-Language-Processing/Natural_Language_Processing/Restaurant_Reviews.tsv',  delimiter = '\t', quoting =3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

corpus = []

for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ',data['Review'][i])
# This removes all characters except alphabet.
    review = review.lower()
# Split breaks the words in the string to a list of words.
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
# Stemmers remove morphological affixes from words, leaving only the word stem. eg. running- run.
    review = ' '.join(review)
    corpus.append(review)

'''Creating bag of words model'''

'''In X, each column corresponds to a word i.e 1 if word is present else 0.'''


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
# Max features to remove least occuring words. we can create a max. features we want.
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.80, random_state = 0)

'''Naive Bayes Classifier(Bayes Theorem) '''

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

'''Predicting the test set result'''
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)