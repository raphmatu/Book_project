# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 18:11:27 2018

@author: Jérôme
"""

#######################  Packages pour extraction de texte #################


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

########### Variable à adapter selon l'utilisateur #####################

work_dir="Desktop/Books/" # Contient les fichiers csv, le corpus et les databases train et test
df_Corpus=pd.read_csv(work_dir+"Corpus1.csv",encoding="latin1",header=None) # Corpus de textes extraits des images

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

## Creation d'une liste de textes extraits depuis les images
Corpus=df_Corpus[0]
Corpus=Corpus.fillna("")


## Elimination des individus sans texte (à traiter par CNN uniquement)
df_y=pd.DataFrame()
df_y["Text"]=Corpus
df_y["Category"]=df_test.Category

Test=pd.DataFrame()
Test["Text"]=df_y.Text[df_y.Text!=""]
Test["Category"]=df_y.Category[df_y.Text!=""]

Train=pd.DataFrame()
Train["Text"]=df_train.Title
Train["Category"]=df_train.Category

####### Fonctions de préprocessing du texte ###################

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

########### Processing et classification du texte ############

        
stop_words=set(stopwords.words("english"))
stemmer=EnglishStemmer()
        
Xtrain,Xtest=Text_data_processor(Train.Text,Test.Text,stopwords=stop_words,tfidf=True)
ytrain=Train.Category
ytest=Test.Category

clf_NB=MultinomialNB(alpha=1.7).fit(Xtrain,ytrain)
NB_pred=clf_NB.predict(Xtest)
print(classification_report(NB_pred,ytest))

############# Sauvegarde du modèle ############################"

from sklearn.externals import joblib
joblib.dump(clf_NB,work_dir+"clf_NB_trained")

########### Chargement du modèle pr-entrainé #################

loaded_clf=joblib.load(work_dir+"clf_NB_trained")
Loaded_pred=loaded_clf.predict(Xtest)
print(classification_report(Loaded_pred,ytest))

