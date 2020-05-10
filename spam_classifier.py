# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:16:48 2020

@author: degananda.reddy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
df=pd.read_csv(r"spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.shape
df['class'].value_counts()
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
df.drop('class',axis=1,inplace=True)
X=df['message']
y=df['label']

msgs_list=X.tolist()
lem_msgs=[]
for i in msgs_list:
    words=nltk.word_tokenize(i)
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    lem_msgs.append(' '.join(words))
# SPAM CLASSIFIER USING BAGOFWORDS    
cv=CountVectorizer(max_features=5000)
vec_data=cv.fit_transform(lem_msgs).toarray()
cv.get_feature_names()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vec_data,y, test_size=0.20, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
clf.score(X_test,y_test)
from sklearn.metrics import accuracy_score,confusion_matrix
BOW_accuracy=accuracy_score(y_test,pred)
confusion_matrix(y_test,pred)
------------------------------------------------------------------------------
# SPAM CLASSIFIER USING TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
tfidf_data = tfidf.fit_transform(lem_msgs).toarray()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_data,y, test_size=0.20, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
clf.score(X_test,y_test)
from sklearn.metrics import accuracy_score,confusion_matrix
TFIDF_accuracy=accuracy_score(y_test,pred)
confusion_matrix(y_test,pred)
print(BOW_accuracy,TFIDF_accuracy)
