# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:27:10 2018

@author: Anurag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv(r'C:\Users\Anurag\OneDrive\NLP\news_articles.tsv', delimiter='\t', quoting=3, encoding="latin1")
X = dataset.iloc[:,1]
y=dataset.iloc[:,2]

classifier = CountVectorizer()
X_vec = classifier.fit_transform(X)
print(X_vec)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
#onehotencoder = OneHotEncoder(categorical_features = [0])
#y = onehotencoder.fit_transform(y.reshape(y.size,-1)).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size = 0.2, random_state = 0)

#classifier = svm.SVC(class_weight='balanced')
#classifier.fit(X_train, y_train)
#
#classifier.score(X_test, y_test)

classifier = LogisticRegression()
#classifier = svm.SVC(class_weight='balanced')
classifier.fit(X_train, y_train)

classifier.score(X_test, y_test)

from sklearn import metrics, cross_validation
predicted=cross_validation.cross_val_predict(classifier, X_train, y_train, cv=10)
print (metrics.accuracy_score(y_train,predicted))
print (metrics.classification_report(y_train,predicted))