

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
df_train = pd.read_csv("D:/Boulot_Raph/2018_06_Formation DATA Scientist/projets/books/book30-listing-train.csv",
                       engine = "python", header = None)
df_test = pd.read_csv("D:/Boulot_Raph/2018_06_Formation DATA Scientist/projets/books/book30-listing-test.csv",
                       engine = "python", header = None)
df_List = pd.read_csv("D:/Boulot_Raph/2018_06_Formation DATA Scientist/projets/books/book32-listing.csv",
                       engine = "python", header = None)

df_train.rename(columns={0:"Amazon_index",1:"Filename",2:"Image_url",3:"Title",4:"Author",5:"Category_id",6:"Category"},
                inplace = True)
df_test.rename(columns={0:"Amazon_index",1:"Filename",2:"Image_url",3:"Title",4:"Author",5:"Category_id",6:"Category"},
                inplace = True)
df_List.rename(columns={0:"Amazon_index",1:"Filename",2:"Image_url",3:"Title",4:"Author",5:"Category_id",6:"Category"},
                inplace = True)

## ouverture des fichiers images avec un iterator et un image generator pour le redimensionnement.
## Le redimensionnement varie selon le modèle de DL utilisé
train_path = 'D:/Boulot_Raph/2018_06_Formation DATA Scientist/projets/books/images/Database_train'
test_path = 'D:/Boulot_Raph/2018_06_Formation DATA Scientist/projets/books/images/Database_test'

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
resultats = pd.DataFrame(index = range(30),columns=['top1','top3','top5'])
valid_top1 = 0
valid_top3 = 0
valid_top5 = 0
test_labels_df = pd.DataFrame((test_labels), columns = ['cat'])


for i in range(pred.shape[1]):# récupération des 5 meilleures prédictions
    maximum = pred[i].sort_values(ascending = False)
    top.loc[i] = maximum.index[0:5]


for k in range(len(classe)):# stats sur les top1 top3 top5 (à optimiser)
    liste = test_labels_df[(test.cat==k)]
    liste_index = list(liste.index)
    for i in liste_index:
        if test.loc[i, 'cat'] == top.loc[i, 'top1']:
            valid_top1 = valid_top1 + 1
        elif ((test.loc[i, 'cat'] == top.loc[i, 'top2']) or
                (test.loc[i, 'cat'] == top.loc[i, 'top3'])):
            valid_top3 = valid_top3 + 1
        elif ((test.loc[i, 'cat'] == top.loc[i, 'top4']) or
                (test.loc[i, 'cat'] == top.loc[i, 'top5'])):
            valid_top5 = valid_top5 + 1
    resultats.loc[k]['top1'] = valid_top1/len(liste_index)*100
    resultats.loc[k]['top3'] = valid_top3/len(liste_index)*100
    resultats.loc[k]['top5'] = valid_top5/len(liste_index)*100
    valid_top1 = 0
    valid_top3 = 0
    valid_top5 = 0

print(resultats)
