

######################### Import des packages nécéssaires #########################

import os
import keras
import itertools
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix

%matplotlib inline

from keras import callbacks
from keras.optimizers import Adam
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model, load_model, model_from_json
from keras_applications.inception_v3 import InceptionV3, preprocess_input

#######################  Packages pour extraction de texte #################

import itertools
from PIL import Image
import time
import cv2 as cv
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pytesseract

# Nécessaire pour utiliser pytesseract sous Windows (chemin d'accès au fichier tesseract-data)
pytesseract.pytesseract.tesseract_cmd='C:/Program Files (x86)/Tesseract-OCR/tesseract'

######################### Definition des fonctions utilisées dans le script #########################


def plot(ims, figsize=(20,10), titles=None):
    # fonction permettant d'afficher les images avec leur categorie en titre
    # à consolider
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        fig = f.add_subplot(2,5,i+1)
        fig.axis('off')
        if titles is not None:
            fig.set_title(titles[i].argmax(), fontsize=10)
        plt.imshow(ims[i])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    # Matrice de confusion. script récupéré sur Github
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.figure(figsize=(len(classes),len(classes)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


######################### Code principal #########################


## Import de mon modèle de Deep learning ##
inception = keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3), weights='imagenet')


## récupération des data sets Train, Test et List. Changement du nom des colonnes ##
work_dir="Desktop/Books/"
df_train = pd.read_csv(work_dir+"book30-listing-train.csv",engine = "python", header = None)
df_test = pd.read_csv(work_dir+"book30-listing-test.csv",engine = "python", header = None)
df_List = pd.read_csv(work_dir+"book32-listing.csv",engine = "python", header = None)
Col_names=["Amazon_index","Filename","Image_url","Title","Author","Category_id","Category"]
df_train.columns=Col_names
df_test.columns=Col_names
df_List.columns=Col_names

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

################ Extraction de texte (pre-alpha) ###################"

# Essai d'extraction de texte. Beaucoup trop lent pour le moment

def textreader(filepath):
    import cv2 as cv
    Corpus=[]
    for i in range(len(filepath)):
        img=plt.imread(filepath[i])
        if(len(img.shape)==3):
            if(img.shape[2]==3):
                img=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
            else:
                img=cv.cvtColor(img,cv.COLOR_RGBA2GRAY)
        ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
        ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
        ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
        ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
        ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
        IMG1=Image.fromarray(thresh1)
        IMG2=Image.fromarray(thresh2)
        IMG3=Image.fromarray(thresh3)
        IMG4=Image.fromarray(thresh4)
        IMG5=Image.fromarray(thresh5)
        Text=[]
        Text.append(pytesseract.image_to_string(IMG1).replace("\n"," ").split())
        Text.append(pytesseract.image_to_string(IMG2).replace("\n"," ").split())
        Text.append(pytesseract.image_to_string(IMG3).replace("\n"," ").split())
        Text.append(pytesseract.image_to_string(IMG4).replace("\n"," ").split())
        Text.append(pytesseract.image_to_string(IMG5).replace("\n"," ").split())
        Text=list(itertools.chain.from_iterable(Text))
        Text=list(set(Text))
        Corpus.append(" ".join(Text))
        print("Images traitées : ",i)
    return Corpus

Corpus=textreader(list(df_test.Filepath[567:698]))
ytest=df_test.Category[567:698]

############ Modèle de classification de texte 'premier essai' ###

countv=CountVectorizer()
xtrain_counts=countv.fit_transform(df_train.Title)
tformeridf=TfidfTransformer(use_idf=True)
xtrain_tfidf=tformeridf.fit_transform(xtrain_counts)
ytrain=df_train.Category
clf=MultinomialNB().fit(xtrain_tfidf,ytrain)


xtest_counts=countv.transform(Corpus)
xtest_tfidf=tformeridf.transform(xtest_counts)
predicted=clf.predict(xtest_tfidf)
accuracy=(ytest==predicted).sum()/len(ytest)

print("Accuracy : ",accuracy)
print(classification_report(predicted,ytest))

####### probas

Pred_text=clf.predict_proba(xtest_tfidf)
cat = pd.DataFrame(df_train['Category'].value_counts())
classe = list(cat.index.values)
classe.sort()
Pred_df=pd.DataFrame(Pred_text,columns=classe)

########################################################"

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.inception_v3.preprocess_input).flow_from_directory(
    train_path, target_size=(299,299))
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.inception_v3.preprocess_input).flow_from_directory(
    test_path, target_size=(299,299), shuffle =False) # shuffle = False pour la matrice de confusion


## affichage des images et de leur catégorie
imgs, labels = next(train_batches)
plot(imgs, titles=labels)

## construction du CNN
inception.summary()

x = inception.layers[311].output #récupération des 311 couches sur 312
last_layer = Dense(30, activation='softmax')(x)
new_inception = Model(inputs=inception.input, outputs=last_layer) # nouveau modèle avec l'output modifié
new_inception.summary()

for layer in new_inception.layers[:-222]:# choix du nombre de couche à ré-entrainer
        layer.trainable=False

## entrainement du CNN avec des callbacks en cas d'overfitting
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience =3, verbose = 1)
callbacks_list = [early_stop]

new_inception.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics = ['accuracy'])
new_inception.fit_generator(train_batches, validation_data=test_batches, epochs=10,verbose=1, callbacks=callbacks_list)


## sauvegarde ou chargement du modèle
new_inception.save_weights('model.h5')
model_json = new_inception.to_json()
with open('inception_simple.json', "w") as json_file:
    json_file.write(model_json)
json_file.close()


json_file = open("inception_simple_13epochs.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
new_inception = model_from_json(loaded_model_json)
new_inception.load_weights("model.h5")


## Prédiction et affichage de la matrice de Confusion
cat = pd.DataFrame(df_train['Category'].value_counts())
classe = list(cat.index.values)
classe.sort()# je récupère le nom de mes catégorie dans l'ordre

predictions = new_inception.predict_generator(test_batches, verbose =1)
test_labels = test_batches.classes
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_confusion_matrix(cm, classe)


## Récupération des top1, top3 et top5 accuracy
pred = pd.DataFrame(np.transpose(predictions))
top = pd.DataFrame(columns=['top1','top2','top3','top4','top5'])
resultats = pd.DataFrame(index = classe, columns=['top1','top3','top5'])
average = pd.DataFrame(index = ['Average'], columns=['top1','top3','top5'])
test_labels_df = pd.DataFrame((test_labels), columns = ['cat'])
valid_top1 = 0
valid_top3 = 0
valid_top5 = 0



for i in range(pred.shape[1]):# récupération des 5 meilleures prédictions
    maximum = pred[i].sort_values(ascending = False)
    top.loc[i] = maximum.index[0:5]


for k in range(len(classe)):# stats sur les top1 top3 top5 (à optimiser)
    liste = test_labels_df[(test_labels_df.cat==k)]
    liste_index = list(liste.index)
    for i in liste_index:
        if test_labels_df.loc[i, 'cat'] == top.loc[i, 'top1']:
            valid_top1 = valid_top1 + 1
            valid_top3 = valid_top3 + 1
            valid_top5 = valid_top5 + 1
        elif ((test_labels_df.loc[i, 'cat'] == top.loc[i, 'top2']) or
                (test_labels_df.loc[i, 'cat'] == top.loc[i, 'top3'])):
            valid_top3 = valid_top3 + 1
            valid_top5 = valid_top5 + 1
        elif ((test_labels_df.loc[i, 'cat'] == top.loc[i, 'top4']) or
                (test_labels_df.loc[i, 'cat'] == top.loc[i, 'top5'])):
            valid_top5 = valid_top5 + 1
    resultats.loc[classe[k]]['top1'] = valid_top1/len(liste_index)*100
    resultats.loc[classe[k]]['top3'] = valid_top3/len(liste_index)*100
    resultats.loc[classe[k]]['top5'] = valid_top5/len(liste_index)*100
    valid_top1 = 0
    valid_top3 = 0
    valid_top5 = 0

average['top1'] = np.mean(resultats.top1)
average['top3'] = np.mean(resultats.top3)
average['top5'] = np.mean(resultats.top5)
resultats = pd.concat([resultats,average], axis = 0)
print(resultats)


import csv

with open(work_dir+"Corpus1.csv",'w') as resultFile:
    wr = csv.writer(resultFile)
    for item in Corpus:
        wr.writerow([item,])
resultFile.close()

df_Corpus=pd.read_csv(work_dir+"Corpus1.csv",encoding="latin1",header=None)