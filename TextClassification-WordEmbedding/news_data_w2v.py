# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:59:21 2018

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
from gensim.models import Word2Vec

from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split


dataset = pd.read_csv(r'C:\Users\Anurag\OneDrive\NLP\news_articles.tsv', delimiter='\t', quoting=3, encoding="latin1")
X = dataset.iloc[:,1]
y=dataset.iloc[:,2]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        #self.dim = len(word2vec.itervalues().next())
        self.dim = len(word2vec)
    def fit(self, X, y):
        return self

    def transform(self, X):
        return (np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X)
                  
word_tokens=[]
X_w2v=[]
for i in range(X.size):
    word_tokens.append(word_tokenize(X[i]))
    model=Word2Vec(word_tokens, size=100,min_count=1)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    meanEmbedVec= MeanEmbeddingVectorizer(w2v)
    X_w2v.append(meanEmbedVec.transform(w2v))
    word_tokens.clear()
etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
#X_w2v=meanEmbedVec.transform(w2v)
X_train, X_test, y_train, y_test = train_test_split(X_w2v, y, test_size = 0.2, random_state = 0)
   
etree_w2v.fit(X_train,y_train)

etree_w2v.score(X_test, y_test)
