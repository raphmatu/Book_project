# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:13:32 2018

@author: Jérôme
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

############# A éxécuter la première fois pour télécharger les stopwords#####
### import nltk
###nltk.download('stopwords')


import itertools
from PIL import Image
from time import time
import cv2 as cv
import pytesseract
import matplotlib.pyplot as plt
import csv
from sklearn.externals import joblib


# Nécessaire pour utiliser pytesseract sous Windows (chemin d'accès au fichier tesseract-data)
pytesseract.pytesseract.tesseract_cmd='C:/Program Files (x86)/Tesseract-OCR/tesseract'
########### Variable à adapter selon l'utilisateur #####################

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


##### Extraction du texte à partir des images #################


def Final_textreader(filepath):
    #t0=time()
    Corpus=[]
    for i in range(len(filepath)):
        img=plt.imread(filepath[i])
        if(len(img.shape)==3):
            if(img.shape[2]==3):
                img=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
            else:
                img=cv.cvtColor(img,cv.COLOR_RGBA2GRAY)
        Text=[]
        IMG0=Image.fromarray(img)
        Text.append(pytesseract.image_to_string(IMG0).replace("\n"," ").split())
        ret,thresh1 = cv.threshold(img,200,255,cv.THRESH_BINARY_INV)
        IMG1=Image.fromarray(thresh1)
        Text.append(pytesseract.image_to_string(IMG1).replace("\n"," ").split())
        ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
        IMG2=Image.fromarray(thresh2)
        Text.append(pytesseract.image_to_string(IMG2).replace("\n"," ").split())
        if(Text==[[],[],[]]):
            ret,thresh3 = cv.threshold(img,225,255,cv.THRESH_BINARY_INV)
            IMG3=Image.fromarray(thresh3)
            Text.append(pytesseract.image_to_string(IMG3).replace("\n"," ").split())
        Text=list(itertools.chain.from_iterable(Text))
        Text=list(set(Text))
        Corpus.append(" ".join(Text))
        #print("Images traitées : ",i)
    #t1=time()-t0
    #print(t1)
    return Corpus

Corpus_final=Final_textreader(df_test.Filepath)

######## Enregistrement du Corpus #################
## Si problème, rajouter encoding="utf-8" 

def write_corpus(name,Corpus):
    with open(name,'w',encoding="utf-8") as resultFile:
        wr = csv.writer(resultFile)
        for item in Corpus:
            wr.writerow([item,])
        resultFile.close()
        
write_corpus(work_dir+"Corpus_final.csv",Corpus_final)



##### Chargement du corpus#############

def read_corpus(corpus_path):
    df=pd.read_csv(corpus_path,encoding="latin1",header=None)
    series=df[0]
    return series.fillna("")

Corpus_final=read_corpus(work_dir+"Corpus_final.csv")


###### Check Corpus stats ##########################"""

def flatten(l):
  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out

def word_splitter(corpus):
    out=[]
    for line in corpus:
        splitted=line.replace("\n"," ").split()
        out.append(splitted)
    return out

def count_words(corpus):
    return len(flatten(word_splitter(corpus)))

def count_empty_lines(corpus):
    empties=0
    for line in corpus:
        if(line==""): empties+=1
    return empties


count_words(Corpus_final)
count_empty_lines(Corpus_final)


###### Turn Corpus into a dataframe without empty text #######

def Corpus_dropna(Corpus):    
    Test=pd.DataFrame()
    Test["Text"]=Corpus
    Test_dropna=pd.DataFrame()
    Test_dropna["Text"]=Test.Text[Test.Text!=""]
    Test_dropna["Category"]=df_test.Category[Test.Text!=""]
    return Test_dropna

df_corpus=Corpus_dropna(Corpus_final)


####### Preprocess Text for classifying ###################""

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

def Text_data_processor(Train,Test,stopwords=None,stemmer=None,tfidf=False):
    Filtered_train=text_processer(Train,stopwords,stemmer)
    Filtered_test=text_processer(Test,stopwords,stemmer)
    countv=CountVectorizer().fit(Filtered_train)
    Train_count=countv.transform(Filtered_train)
    Test_count=countv.transform(Filtered_test)
    if(tfidf==True):
        tformer=TfidfTransformer(use_idf=True).fit(Train_count)
        Train_idf=tformer.transform(Train_count)
        Test_idf=tformer.transform(Test_count)
        return Train_idf,Test_idf
    else:
        return(Train_count,Test_count)

       
stop_words=set(stopwords.words("english"))
stemmer=EnglishStemmer()
Xtrain,Xtest=Text_data_processor(df_train.Title,df_corpus.Text,stopwords=stop_words,stemmer=None,tfidf=True)
ytrain=df_train.Category
ytest=df_corpus.Category

######## Apply classification ###############

clf=MultinomialNB(alpha=1.4).fit(Xtrain,ytrain)
ypred=clf.predict(Xtest)
print(classification_report(ytest,ypred))
print(clf.score(Xtest,ytest))


################ Sauvegarde du modèle ###########

joblib.dump(clf,work_dir+"clf_Final")

##############  Chargement du modèle #########

clf_loaded=joblib.load(work_dir+"clf_Final")