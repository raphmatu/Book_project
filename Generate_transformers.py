# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:17:02 2018

@author: Jérôme
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib


work_dir="Desktop/Books/" # Contient les fichiers csv, le corpus et les databases train et test

## récupération des data sets Train, Test et List. Changement du nom des colonnes ##
df_train = pd.read_csv(work_dir+"book30-listing-train.csv",engine = "python", header = None)
df_test = pd.read_csv(work_dir+"book30-listing-test.csv",engine = "python", header = None)
#df_List = pd.read_csv(work_dir+"book32-listing.csv",engine = "python", header = None)
Col_names=["Amazon_index","Filename","Image_url","Title","Author","Category_id","Category"]
df_train.columns=Col_names
df_test.columns=Col_names
#df_List.columns=Col_names

## ouverture des fichiers images avec un iterator et un image generator pour le redimensionnement.
## Le redimensionnement varie selon le modèle de DL utilisé
train_path = work_dir+'Database_train'
test_path = work_dir+'Database_test'

## Ajout d'une colonne filepath

filepath=list()
for i in range(len(df_test.Filename)):
    filepath.append(test_path+"/"+df_test.Category[i]+"/"+df_test.Filename[i])
df_test["Filepath"]=filepath

filepath=list()
for i in range(len(df_train.Filename)):
    filepath.append(train_path+"/"+df_train.Category[i]+"/"+df_train.Filename[i])
df_train["Filepath"]=filepath


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



stop_words=set(stopwords.words("english"))
Filtered_train=text_processer(df_train.Title,stop_words=stop_words,stemmer=None)
countv=CountVectorizer().fit(Filtered_train)

Train_count=countv.transform(Filtered_train)
tformer=TfidfTransformer(use_idf=True).fit(Train_count)

########## Sauvegarde des transformeurs entrâinés ###############

joblib.dump(stop_words,work_dir+"stopwords")
joblib.dump(countv,work_dir+"countvectorizer")
joblib.dump(tformer,work_dir+"tfidf_transformer")



