# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:37:19 2018

@author: Jérôme
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.metrics import classification_report

work_dir="Desktop/Books/" # Contient les fichiers csv, le corpus et les databases train et test

## récupération des data sets Train, Test et List. Changement du nom des colonnes ##
df_train = pd.read_csv(work_dir+"book30-listing-train.csv",engine = "python", header = None)
df_test = pd.read_csv(work_dir+"book30-listing-test.csv",engine = "python", header = None)
#df_List = pd.read_csv(work_dir+"book32-listing.csv",engine = "python", header = None)
Col_names=["Amazon_index","Filename","Image_url","Title","Author","Category_id","Category"]
df_train.columns=Col_names
df_test.columns=Col_names
#df_List.columns=Col_names


def read_corpus(corpus_path):
    df=pd.read_csv(corpus_path,encoding="latin1",header=None)
    series=df[0]
    return series.fillna("")



def Corpus_dropna(Corpus):    
    Test=pd.DataFrame()
    Test["Text"]=Corpus
    Test_dropna=pd.DataFrame()
    Test_dropna["Text"]=Test.Text[Test.Text!=""]
    Test_dropna["Category"]=df_test.Category[Test.Text!=""]
    return Test_dropna




######### Processing ###################

def stop_words_filtering(liste,stop_words):
   filtered=[]
   for word in liste:
       if word not in stop_words: filtered.append(word)
   return(filtered)

def stemming(mots,stemmer):
    stems=[]
    for mot in mots:
        stems.append(stemmer.stem(mot))
    return stems

def text_processer(list_of_sentences,stop_words=None,stemmer=None):
    sortie=[]
    for sentence in list_of_sentences:
        tokens=word_tokenize(sentence.lower(),language="english")
        if(stop_words!=None):
            filtered=stop_words_filtering(tokens,stop_words)
        else:
            filtered=tokens
        if(stemmer!=None):
            stemmed=stemming(filtered,stemmer)
        else:
            stemmed=filtered
        sortie.append(" ".join(stemmed))
    return sortie



stop_words=joblib.load(work_dir+"stopwords")
countv=joblib.load(work_dir+"countvectorizer")
tformer=joblib.load(work_dir+"tfidf_transformer")
clf=joblib.load(work_dir+"clf_final")

Corpus_final=read_corpus(work_dir+"Corpus_final.csv")
df_corpus=Corpus_dropna(Corpus_final)
Filtered_test=text_processer(df_corpus.Text,stop_words=stop_words,stemmer=None)
Test_count=countv.transform(Filtered_test)
Xtest=tformer.transform(Test_count)
ytest=df_corpus.Category

ypred=clf.predict(Xtest)
print(classification_report(ytest,ypred))
print(clf.score(Xtest,ytest))

