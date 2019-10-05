# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:39:01 2018

@author: Anurag
"""

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import normalize

dataset = pd.read_csv(r'C:\Users\Anurag\OneDrive\NLP\uci_news_data.tsv', delimiter='\t', quoting=3, encoding='ISO-8859-1')
article = dataset.iloc[:,1]
topic = dataset.iloc[:,2]

master_word_list=[]
master_article=''
for j in range(len(article)):
    master_article+=article[j]

stop_words=set(stopwords.words('english'))
master_article=master_article.replace('',"'")
master_word_list=master_article.lower().split()
word_tokens=word_tokenize(master_article.lower())
master_filtered_sentence=[w for w in word_tokens if not w in stop_words]
master_tagged_words=pos_tag(master_filtered_sentence)
master_nouns=[word for word, pos in master_tagged_words\
              if(pos == 'NN' or pos == 'NNP' or pos == 'NNS')]
master_nouns[:] = [noun for noun in master_nouns if (noun != '\x94' and noun !='%')]
master_word_freq=[]

for k in range(len(master_nouns)):
        master_word_freq.append(master_word_list.count(master_nouns[k]))
 
dict={
      'data':master_nouns,
      'freq':master_word_freq
     }  
columns=['data','freq']     
master_set=pd.DataFrame(dict, columns=columns)
master_set=master_set.loc[master_set['freq']!=0]
master_set=master_set.drop_duplicates().reset_index()
master_set=master_set.drop(['index'],axis=1)
master_freq=master_set.iloc[:,1].values.astype(np.float64)
tot=np.sum(master_freq)



def embed_word(category, article_data):
#    article=dataset.loc[dataset['Category']==category].iloc[split_start:split_end,:]
    article_split=article_data.loc[dataset['Category']==category].reset_index()
    wordlist=[]
    split_article=''
    for i in range(len(article_split)):
        split_article+=article_split.iloc[i,1]
    stop_words=set(stopwords.words('english'))
    split_article=split_article.replace('',"'")
    wordlist=split_article.lower().split()
    word_tokens=word_tokenize(split_article.lower())
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    tagged_word=pos_tag(filtered_sentence)
    nouns=[word for word,pos in tagged_word\
           if (pos == 'NN' or pos == 'NNP' or pos == 'NNS')]
    nouns[:] = [noun for noun in nouns if (noun != '\x94' and noun !='%')]
    word_freq = []
    for i in range(len(nouns)):
        word_freq.append(wordlist.count(nouns[i]))
    #word_freq[:]=[freq for freq in word_freq if freq != 0]
    dict={
            'data': nouns,
            'freq': word_freq
        }
    columns=['data','freq']
    X_article=pd.DataFrame(dict, columns=columns)
    X_article=X_article.loc[X_article['freq']!=0]
    article_freq=X_article.iloc[:,1].values
    article_freq=article_freq.astype(np.float64)
    article_prob=[]
    #tot=np.sum(biz_prob)
    for freq in range(len(article_freq)):
        sum=0
        for i in range(len(article_freq)):
            if(i!=freq):
    #            prob_con_cen=biz_freq[i]/biz_freq[freq]
    #            prob_cen=biz_freq[freq]/tot
    #            prob_con=biz_freq[i]/tot
    #            sum+=(prob_con_cen*prob_cen)/prob_con
                prob=np.log(article_freq[freq]/article_freq[i])
                if(prob<=1):
                    sum+=prob
                else:
                    sum+=1
            else:
                continue
        article_prob.append(float(sum/len(article_freq)))
    X_article=X_article.assign(prob=article_prob,topic=category)
    return X_article

from sklearn.cross_validation import train_test_split
article_train, article_test, topic_train, topic_test = train_test_split(article, topic, test_size = 0.2, random_state = 0)

biz_train=embed_word('b',article_train)
biz_test=embed_word('b',article_test)
e_train=embed_word('e',article_train)
e_test=embed_word('e',article_test)
m_train=embed_word('m',article_train)
m_test=embed_word('m',article_test)
t_train=embed_word('t',article_train)
t_test=embed_word('t',article_test)
Train=pd.concat([biz_train,e_train,m_train,t_train],ignore_index=True)
Test=pd.concat([biz_test,e_test,m_test,t_test],ignore_index=True)

X_train=Train.iloc[:,[0,2]]
y_train=Train.iloc[:,-1]
X_test=Test.iloc[:,[0,2]]
y_test=Test.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X_train.iloc[:,0] = labelencoder.fit_transform(X_train.iloc[:,0])
y_train= labelencoder.fit_transform(y_train)
X_test.iloc[:,0] = labelencoder.fit_transform(X_test.iloc[:,0])
y_test= labelencoder.fit_transform(y_test)

#normalize=Normalizer(norm='l2')
#train_data = normalize(X_train.iloc[:,0].values.reshape(1,-1),axis=0)
#X_test = normalize.fit_transform(X_test)

#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#train_data=sc_X.fit_transform(X_train.iloc[:,0].reshape(1,-1))
classifier = LogisticRegression()
#classifier = svm.SVC(class_weight='balanced')
classifier.fit(X_train, y_train)

classifier.score(X_test, y_test)

from sklearn import metrics, cross_validation
predicted=cross_validation.cross_val_predict(classifier, X_train, y_train, cv=10)
print (metrics.accuracy_score(y_train,predicted))
print (metrics.classification_report(y_train,predicted))





    



