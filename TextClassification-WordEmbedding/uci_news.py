# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 08:27:25 2018

@author: Anurag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gensim.models import Word2Vec


dataset=pd.read_csv(r'C:\Users\Anurag\OneDrive\NLP\NewsDataFinal.tsv', delimiter='\t', quoting=3, encoding="ISO-8859-1")


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range (0,1690):
    review=re.sub('[^a-zA-Z]',' ', dataset['News Data'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#from sklearn.feature_extraction.text import CountVectorizer
#cv=CountVectorizer(max_features=1500)
#X=cv.fit_transform(corpus).toarray()

model=Word2Vec(corpus, min_count=1)
X=model[model.wv.vocab]
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


kmeans=KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)

