

            #######################################################
            ################ IMPORT DES PACKAGES ##################
            #######################################################

import os
import keras
import itertools
%matplotlib inline
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pylab as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
############# A éxécuter la première fois pour télécharger les stopwords#####
### import nltk
###nltk.download('stopwords')


from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from keras import callbacks
from keras.preprocessing import image
from keras.optimizers import Adam ,SGD
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, GlobalAveragePooling2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras_applications.inception_v3 import InceptionV3, preprocess_input



             #######################################################
             #################### FONCTIONS ########################
             #######################################################


####### Fonctions de préprocessing du texte ###################

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


####### Fonctions pour le mdèle Inception ###################

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

def extract_features_keras(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    model = Model(inputs=new_inception.input, outputs=new_inception.get_layer('avg_pool').output)

    for i, image_path in enumerate(list_images):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)
        features[i,:]=np.squeeze(predictions)
    return features

def new_inception_training(trainable_layers = -100, save_as = 'model'):

    inception = keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3), weights='imagenet')

    train_batches = ImageDataGenerator(preprocessing_function=keras.applications.inception_v3.preprocess_input).flow_from_directory(
        train_path, target_size=(299,299))
    test_batches = ImageDataGenerator(preprocessing_function=keras.applications.inception_v3.preprocess_input).flow_from_directory(
        test_path, target_size=(299,299), shuffle =False) # shuffle = False pour la matrice de confusion

    ## construction du CNN
    inception.summary()

    x = inception.layers[-2].output #suppression de la dernière couche

    x = Dense(1024, activation='sigmoid')(x)
    x = Dropout(0.5)(x)
    x = Dense(30, activation='softmax')(x)
    new_inception = Model(inputs=inception.input, outputs=x) # nouveau modèle avec l'output modifié, ainsi que le Dropout

    for layer in new_inception.layers[:trainable_layers]:# choix du nombre de couche à ré-entrainer
            layer.trainable=False
    new_inception.summary()

    ## entrainement du CNN avec des callbacks en cas d'overfitting
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience =5, verbose = 1)
    callbacks_list = [early_stop]

    new_inception.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics = ['accuracy'])
    new_inception.fit_generator(train_batches, validation_data=test_batches, epochs=10,verbose=1, callbacks=callbacks_list)

    ## Prédiction et affichage de la matrice de Confusion
    cat = pd.DataFrame(df_train['Category'].value_counts())
    classe = list(cat.index.values)
    classe.sort()# je récupère le nom de mes catégorie dans l'ordre

    predictions = new_inception.predict_generator(test_batches, verbose =1)
    test_labels = test_batches.classes
    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
    plot_confusion_matrix(cm, classe)
    top_table(predictions)

    ## sauvegarde ou chargement du modèle
    new_inception.save_weights(model'.h5')
    model_json = new_inception.to_json()
    with open(model, "w") as json_file:
        json_file.write(model_json)
    json_file.close()

    return new_inception

def SVM_new_inception:

    df_path_cat_test = pd.DataFrame({'path': df_test.Filepath, 'Category': df_test.Category})
    list_images_test = df_path_cat_test.path
    labels_test = df_path_cat_test.Category

    df_path_cat_train = pd.DataFrame({'path': df_train.Filepath, 'Category': df_train.Category})
    list_images_train = df_path_cat_train.path
    labels_train = df_path_cat_train.Category



    extraction des 2048 features de la couches Avg_pool de new_inception
    features_test = extract_features(list_images_test)
    features_train = extract_features(list_images_train)

    features_test = pd.DataFrame(features_test)
    features_train = pd.DataFrame(features_train)
    features_test.to_csv("2048_features_test_forSVM")
    print('sauvegarde des 2048 features_test_forSVM')
    features_train.to_csv("2048_features_train_forSVM")
    print('sauvegarde des 2048 features_train_forSVM')


    pca = PCA(n_components=0.9)
    features_train_pca = pca.fit_transform(features_train)
    features_test_pca = pca.transform(features_test)


    ## utilisation d'une SVM pour séparer les catégories

    clf = SVC()
    parametres = { 'C' : [0.1,1,10], 'kernel': ['rbf', 'linear','poly'], 'gamma' : [0.001, 0.1, 0.5]}
    grid_clf = model_selection.GridSearchCV(clf, param_grid = parametres)

    grid_clf.fit(features_train_pca, labels_train)
    pred = grid_clf.predict(features_test_pca)

    test_labels = test_batches.classes
    cm = confusion_matrix(labels_test, pred)
    plot_confusion_matrix(cm, classe)

    return grid_clf


def top_table(pred):
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
    resultats.to_csv("resultats_prédiction.csv")
    print(resultats)
    return resultats

             #######################################################
             ############## ENVIRONNEMENT A ADAPTER ################
             ################## SELON UTILISATEUR ##################
             #######################################################



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



             #######################################################
             ################### CODE PRINCIPAL ####################
             #######################################################



if os.path.isfile('best_results_inception_180905.json') == True:
    print("Chargement du modèle new_inception pré-entrainé...")
    json_file = open("best_results_inception_180905.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    new_inception = model_from_json(loaded_model_json)
    new_inception.load_weights("best_results_inception_180905.h5")
    print("Modèle de prédiction new_inception chargé")
else:
    print("Le modèle de prédiction \"new_inception\" n'a pas été chargé")

## Import de mon modèle de Deep learning ##

if os.path.isfile('clf_SVM_inception.h5')==True:
    print('chargement du modèle SVM_new_inception...')
    clf_SVM_inception = open('clf_SVM_inception.h5')
    print("Modèle de prédiction SVM_new_inception chargé")
else:
    print("Le modèle de prédiction \"SVM_new_inception\" n'a pas été chargé")


if os.path.isfile('clf_NB_trained.h5')==True:
    print('chargement du modèle de Text Mining...')
    clf_TextMining=open("clf_NB_trained")
    print("Modèle de Text Mining chargé")
else:
    print("Le modèle de Text Mining \"clf_TextMining\" n'a pas été chargé")


if textreader(input) ==





 #######################################################
 ####################ANALYSE DE TEXTE###################
 #######################################################

df_Corpus=pd.read_csv(work_dir+"Corpus1.csv",encoding="latin1",header=None) # Corpus de textes extraits des images

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
