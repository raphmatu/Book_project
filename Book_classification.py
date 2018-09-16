## The objective of the program is to categorize books according to their cover.

## We used a dataset available at this address :
## https://github.com/uchidalab/book-dataset

##  3 data sets were available :
## - Data set Train ==> 51 300 images
## - Data set test ==> 5 700 images
## - Data set Final ==> 207 000 images where we removed the 57 000 images of
##                     train and test

## We trained our models on train and used test to validate it.
## You can simply run the program. It will provide our results on Final Data set
## and give you the possibility to upload your own image to test it.

## We used inceptionV3 CNN and Text mining to classify covers.

## Inception network is very powerful, we changed the last layer and replace it
## with 3 layers (Two dense layers and one Dropout). We tried many modifications
## but it appears that overfitting was very difficult to limit due to the high
## covers variablity. We choose to re-train the last 100 layers of the new
## inception model. The others were pre-trained with imageNet data.

## Text mining...................




## IMPORTANT : to run properly the program, you have to download :
##
##              - all the pre-trained models
##              - the 2 csv files train and test
##              - image data base (if you want to re-train models)
##              - Final corpus file
##
## You have to put all of them in one folder :

work_dir='D:\\Boulot_Raph\\2018_06_Formation DATA Scientist\\projets\\books\\Final\\'
#work_dir='Desktop/Books/'


            #######################################################
            ####################### PACKAGES ######################
            #######################################################

import os
import csv
import keras
import nltk
import pickle
import itertools
import cv2 as cv
#%matplotlib inline
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from time import time
from joblib import load, dump
import matplotlib.pylab as plt
from IPython.display import clear_output

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
nltk.download('punkt')
nltk.download('stopwords')

from bokeh.plotting import figure
from bokeh.models.tools import HoverTool
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import push_notebook,output_notebook, show
output_notebook()

from sklearn.svm import SVC
from sklearn.externals import joblib
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


# To use pytesseract with Windows OS (Filepath to tesseract-data)
pytesseract.pytesseract.tesseract_cmd='C:/Program Files (x86)/Tesseract-OCR/tesseract'


#######################################################
############ PRE-TRAINED MODEL LOADING ################
#######################################################


## pre-trained model loading

if os.path.isfile(work_dir + 'new_inception.json') == True and \
   os.path.isfile(work_dir + 'new_inception.h5') == True:

    print("New_inception model loading...")
    json_file = open(work_dir + "new_inception.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    new_inception = model_from_json(loaded_model_json)
    new_inception.load_weights(work_dir + "new_inception.h5")
    print("New_inception model loaded")
else:
    print("New_inception model can not be loaded, please check the file name or the filepath")


if os.path.isfile(work_dir + 'SVM_new_inception') == True and \
   os.path.isfile(work_dir + 'pca_pre_SVM') == True:
    print('SVM_new_inception model loading...')
    pca_pre_svm = load(work_dir + 'pca_pre_SVM')
    clf_SVM_new_inception = load(work_dir + 'SVM_new_inception')
    print("SVM_new_inception model loaded")
else:
    print("SVM_New_inception model can not be loaded, please check the file name or the filepath")


if os.path.isfile(work_dir + 'clf_textmining') == True and \
   os.path.isfile(work_dir + 'stopwords') == True and \
   os.path.isfile(work_dir + 'countvectorizer') == True and \
   os.path.isfile(work_dir + 'tfidf_transformer') == True:
    print('Text Mining model loading...')

    stop_words=load(work_dir + "stopwords")
    countv=load(work_dir + "countvectorizer")
    tformer=joblib.load(work_dir + "tfidf_transformer")
    clf_TextMining=joblib.load(work_dir + "clf_textmining")

    print("Text Mining model loaded")
else:
    print("Text Mining model can not be loaded, please check the file name or the filepath")

Choice_matrix=pd.read_csv(work_dir + "Choice_matrix.csv", sep=";",index_col=0)


#######################################################
####################### VARIABLES #####################
#######################################################

##  3 data sets were available :
## - Data set Train ==> 51 300 images
## - Data set test ==> 5 700 images
## - Data set Final ==> 207 000 images where we removed the 57 000 images of train and test


## Loading of dataframe train and test.
df_train = pd.read_csv(work_dir+"book30-listing-train.csv",engine = "python", header = None)
df_test = pd.read_csv(work_dir+"book30-listing-test.csv",engine = "python", header = None)
df_list = pd.read_csv(work_dir+"book32-listing.csv",engine = "python", header = None)

Col_names=["Amazon_index","Filename","Image_url","Title","Author","Category_id","Category"]
df_train.columns=Col_names
df_test.columns=Col_names
df_list.columns=Col_names

## Filepath of database Train and test
train_path = work_dir+'Database_train'
test_path = work_dir+'Database_test'
list_path=work_dir+"BigData"
## Adding filepath column to dataframes
filepath=list()
for i in range(len(df_test.Filename)):
    filepath.append(test_path+"/"+df_test.Category[i]+"/"+df_test.Filename[i])
df_test["Filepath"]=filepath

filepath=list()
for i in range(len(df_train.Filename)):
    filepath.append(train_path+"/"+df_train.Category[i]+"/"+df_train.Filename[i])
df_train["Filepath"]=filepath

filepath=list()
for i in range(len(df_list.Filename)):
    filepath.append(list_path+"/"+df_list.Category[i]+"/"+df_list.Filename[i])
df_list["Filepath"]=filepath

## storage of category names
cat = pd.DataFrame(df_train['Category'].value_counts())
classe=['Arts & Photography',
 'Biographies & Memoirs',
 'Business & Money',
 'Calendars',
 "Children's Books",
 'Comics & Graphic Novels',
 'Computers & Technology',
 'Cookbooks, Food & Wine',
 'Crafts, Hobbies & Home',
 'Christian Books & Bibles',
 'Engineering & Transportation',
 'Health, Fitness & Dieting',
 'History',
 'Humor & Entertainment',
 'Law',
 'Literature & Fiction',
 'Medical Books',
 'Mystery, Thriller & Suspense',
 'Parenting & Relationships',
 'Politics & Social Sciences',
 'Reference',
 'Religion & Spirituality',
 'Romance',
 'Science & Math',
 'Science Fiction & Fantasy',
 'Self-Help',
 'Sports & Outdoors',
 'Teen & Young Adult',
 'Test Preparation',
 'Travel']



             #######################################################
             #################### FUNCTIONS ########################
             #######################################################

########################## LeNet model  ############################

def LeNet_model(df_train=df_train, df_test=df_test):

    # Create input array for CNN learning
    def generate_image_data(images,size=28):
       tensor=list()

    # images = list of all images filepath ex: test_data.path
    # size = int, wanted picture size

       for i in range(len(images)):
    # Image reading
           img=plt.imread(images[i])
    # Image resizing
           resized=cv2.resize(img,dsize=(size,size),interpolation=cv2.INTER_CUBIC)
    # Detecting whether image is in greyscale
           if(len(resized.shape)==2):
               resized=grey2rgb(resized)
    # Detecting whether image is in rgba
           if(resized.shape[2]==4):
               resized=(rgba2rgb(resized)*255).astype("int")
    # Adding resized image to the list
           tensor.append(resized)
    # Transforming list of images into a single array
       tensor=np.asarray(tensor)
       return tensor


    # Generating Train/Test datasets
    Xtest=generate_image_data(df_test.path,size=28)
    Xtrain=generate_image_data(df_train.path,size=28)
    ytrain=df_train.Category_id
    ytest=df_test.Category_id

    ytest=np_utils.to_categorical(df_test.Category_id)
    ytrain=np_utils.to_categorical(df_train.Category_id)


    ###LeNet model
    model = Sequential()
    model.add(Conv2D(filters=30, kernel_size=(5,5), border_mode='valid', input_shape=(28,28,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model = Sequential()
    model.add(Conv2D(filters=30, kernel_size=(5,5), border_mode='valid', input_shape=(28,28,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(filters=15, kernel_size=(3,3), border_mode='valid', input_shape=(28,28,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    #model training
    model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest),epochs=50, batch_size=300, verbose=1,shuffle=True)

    #prediction
    test_pred = model.predict(Xtest)
    print(test_pred)

    pred = np.around(test_pred,decimals=8).argmax(axis=1)
    scores = model.evaluate(Xtest, ytest, verbose=2)
    print("Perte: %.2f Erreur: %.2f%%" % (scores[0], 100-scores[1]*100))
    y = ytest.argmax(axis=1)

    print(y.shape)
    print(pred.shape)


################### Text preprocessing functions ###################

def Final_textreader(filepath):
    ## Image reading and text extraction. 3 thresholds are used :
    ## - THRESH_BINARY_INV
    ## - THRESH_TRUNC
    ## - THRESH_BINARY_IN
    ## After extraction, word are fused into a corpus
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

def write_corpus(name,Corpus):
    ## Save the corpus in a csv file

    with open(name,'w',encoding="utf-8") as resultFile:
        wr = csv.writer(resultFile)
        for item in Corpus:
            wr.writerow([item,])
        resultFile.close()

def read_corpus(corpus_path):
    ## Read the corpus in a pandas Dataframe

    df=pd.read_csv(corpus_path,encoding="latin1",header=None)
    series=df[0]
    return series.fillna("")

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

def Corpus_dropna(Corpus):
    ## Turn Corpus into a dataframe without empty text

    Test=pd.DataFrame()
    Test["Text"]=Corpus
    Test_dropna=pd.DataFrame()
    Test_dropna["Text"]=Test.Text[Test.Text!=""]
    Test_dropna["Category"]="unknown"
    #Test_dropna["Category"]=df_test.Category[Test.Text!=""]
    return Test_dropna

def stop_words_filtering(liste,stop_words):
    ## remove stop_words from text
    ## see https://pythonspot.com/nltk-stop-words/

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
    ## Process text to separate words, stem them and remove stop words

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
    ## Text processing using text_processer function (stopword, stemming)
    ## Text elements are also transformed into numeric value

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

def Text_mining(df_train, df_test, save = False):
    ## Main text analysis function
    ## Text extraction, Text processing, Text transforming
    ##  MultinomialNB model training and saving

    Corpus_final=Final_textreader(df_test.Filepath)
    #write_corpus(work_dir+"Corpus_final.csv",Corpus_final)
    #Corpus_final=read_corpus(work_dir+"Corpus_final.csv")
    count_words(Corpus_final)
    count_empty_lines(Corpus_final)
    df_corpus=Corpus_dropna(Corpus_final)


    stop_words=set(stopwords.words("english"))
    Filtered_train=text_processer(df_train.Title,stop_words=stop_words,stemmer=None)
    countv=CountVectorizer().fit(Filtered_train)
    Train_count=countv.transform(Filtered_train)
    tformer=TfidfTransformer(use_idf=True).fit(Train_count)


    stemmer=EnglishStemmer()
    Xtrain,Xtest=Text_data_processor(df_train.Title,df_corpus.Text,stopwords=stop_words,stemmer=None,tfidf=True)
    ytrain=df_train.Category_id
    ytest=df_corpus.Category_id

    clf=MultinomialNB(alpha=1.4).fit(Xtrain,ytrain)
    ypred=clf.predict(Xtest)
    print(classification_report(ytest,ypred))
    print(clf.score(Xtest,ytest))

    if save == True:
        joblib.dump(stop_words,work_dir+"stopwords")
        joblib.dump(countv,work_dir+"countvectorizer")
        joblib.dump(tformer,work_dir+"tfidf_transformer")
        joblib.dump(clf,work_dir+"clf_Textmining")
    #clf_loaded=joblib.load(work_dir+"clf_Textmining")

    return clf


################### inception model functions ###################


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',
                          cmap=plt.cm.Blues):

    # Script from github
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
    ## Extract the transfer values of the last Avg_pool layer of new_inception
    ## model. Weights are kept from new inception model
    ## A new model is made with Avg_pool as the last layer. Image are processed
    ## in the CNN and features extracted

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

def inception_one_image(image_path):

    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = new_inception.predict(x)
    return predictions

def new_inception_training(train_path, test_path, classe=classe, save = False):
    ## Main function of new_inception model training
    ## Inception model is customed by removing last layer and adding 3 layers
    ## The last 100 layers are set as trainable
    ## new_inception model is trained on train data set and saved

    inception = keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3), weights='imagenet')

    train_batches = ImageDataGenerator(preprocessing_function=keras.applications.inception_v3.preprocess_input).flow_from_directory(
        train_path, target_size=(299,299))
    test_batches = ImageDataGenerator(preprocessing_function=keras.applications.inception_v3.preprocess_input).flow_from_directory(
        test_path, target_size=(299,299), shuffle =False) # shuffle = False  for the confusion matrix at the end

    ## CNN building
    inception.summary()

    x = inception.layers[-2].output #last layer removing

    x = Dense(1024, activation='sigmoid')(x)
    x = Dropout(0.5)(x)
    x = Dense(30, activation='softmax')(x)
    new_inception = Model(inputs=inception.input, outputs=x)

    for layer in new_inception.layers[:-100]:# layers to train
            layer.trainable=False
    new_inception.summary()

    ## CNN training with callbacks to prevent overfitting
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience =5, verbose = 1)
    callbacks_list = [early_stop]

    new_inception.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics = ['accuracy'])
    new_inception.fit_generator(train_batches, validation_data=test_batches, epochs=20,verbose=1, callbacks=callbacks_list)

    ## Prediction and confusion matrix

    predictions = new_inception.predict_generator(test_batches, verbose =1)
    test_labels = test_batches.classes
    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
    plot_confusion_matrix(cm, classe)
    top_table(predictions)

    ## model saving
    if save == true:
        new_inception.save_weights(work_dir + 'model_inception.h5')
        model_json = new_inception.to_json()
        with open(work_dir + 'model_inception', "w") as json_file:
            json_file.write(model_json)
        json_file.close()

    return new_inception

def SVM_new_inception(df_train, df_test, classe=classe, save=False):
    ## First, PCA is used to reduced dimension of extracted features. We keep
    ## 90% of explained variability with 229 features.
    ## Then, SVM model is trained on reduced features to predict labels of images.
    ## We used a GridSearch to determine best parameters

    df_path_cat_test = pd.DataFrame({'path': df_test.Filepath, 'Category_id': df_test.Category_id})
    list_images_test = df_path_cat_test.path
    labels_test = df_path_cat_test.Category_id
    df_path_cat_train = pd.DataFrame({'path': df_train.Filepath, 'Category_id': df_train.Category_id})
    list_images_train = df_path_cat_train.path
    labels_train = df_path_cat_train.Category_id

    ## new_inception features extraction (last Avg_pool layer ==> 2048 features)
    features_test = extract_features_keras(list_images_test)
    features_train = extract_features_keras(list_images_train)

    features_test.shape
    features_train.shape

    ## features saving
    if save == True:
        features_test = pd.DataFrame(features_test)
        features_train = pd.DataFrame(features_train)
        features_test.to_csv(work_dir + "2048_features_test_forSVM")
        print('sauvegarde des 2048 features_test_forSVM')
        features_train.to_csv(work_dir + "2048_features_train_forSVM")
        print('sauvegarde des 2048 features_train_forSVM')

    ##Features loading
    #features_test = pd.read_csv(work_dir + "features_test_forSVM")
    #features_train = pd.read_csv(work_dir + "features_train_forSVM")
    #features_train.drop('Unnamed: 0', axis = 1, inplace=True)
    #features_test.drop('Unnamed: 0', axis = 1, inplace=True)

    pca = PCA(n_components=0.9)
    features_train_pca = pca.fit_transform(features_train)
    features_test_pca = pca.transform(features_test)

    if save == True:
        joblib.dump(pca,work_dir + 'pca_pre_SVM')

    clf = SVC(verbose = True, probability= True)
    parametres = { 'C' : [0.1,1,10], 'kernel': ['rbf', 'linear','poly'], 'gamma' : [0.001, 0.1, 0.5]}
    grid_clf = model_selection.GridSearchCV(clf, param_grid = parametres)

    grid_clf.fit(features_train_pca, labels_train)
    if save == True:
        joblib.dump(grid_clf,"SVM_new_inception")

    pred_proba = grid_clf.predic_proba(features_test_pca) ## for top_table function
    pred = grid_clf.predict(features_test_pca) ## for confusion matrix

    test_labels = test_batches.classes
    cm = confusion_matrix(labels_test, pred)
    plot_confusion_matrix(cm, classe)

    return grid_clf, pred_proba


################### Analysis functions ###################


def top_table(pred, ytest, label):
    ## datatable of top_results
    yt=ytest
    yt.index=range(len(yt))
    pred = pd.DataFrame(np.transpose(pred))
    top = pd.DataFrame(columns=['top1','top2','top3','top4','top5'])
    resultats = pd.DataFrame(index = label, columns=['top1','top3','top5'])
    average = pd.DataFrame(index = ['Average'], columns=['top1','top3','top5'])
    test_labels_df=pd.DataFrame()
    test_labels_df["cat"]=list(label)
    test_labels_df["cat_id"]=list(range(len(label)))
    valid_top1 = 0
    valid_top3 = 0
    valid_top5 = 0

    for i in range(pred.shape[1]):# récupération des 5 meilleures prédictions
        maximum = pred[i].sort_values(ascending = False)
        top.loc[i] = maximum.index[0:5]


    for k in range(len(label)):# stats sur les top1 top3 top5 (à optimiser)
        liste_index = yt.index[yt==k]
        for i in liste_index:
            if test_labels_df.loc[k, 'cat_id'] == top.loc[i, 'top1']:
                valid_top1 = valid_top1 + 1
                valid_top3 = valid_top3 + 1
                valid_top5 = valid_top5 + 1
            elif ((test_labels_df.loc[k, 'cat_id'] == top.loc[i, 'top2']) or
                    (test_labels_df.loc[k, 'cat_id'] == top.loc[i, 'top3'])):
                valid_top3 = valid_top3 + 1
                valid_top5 = valid_top5 + 1
            elif ((test_labels_df.loc[k, 'cat_id'] == top.loc[i, 'top4']) or
                    (test_labels_df.loc[k, 'cat_id'] == top.loc[i, 'top5'])):
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
    resultats.to_csv(work_dir + "resultats_prédiction.csv")
    print(resultats)
    return resultats

def classement_predictions(predictions, classe=classe):
    pred = pd.DataFrame(np.transpose(predictions))
    maximum = pred.sort_values(by=0, ascending = False)
    max_3 = maximum.index[0:3]
    classe_3 = []
    for i in max_3:
        classe_3.append(classe[i])
    label = []
    for i in maximum.index:
        label.append(classe[i])

    return classe_3, maximum, label,

def prediction(img, classe=classe, stopwords= stop_words, clf_Text = clf_TextMining,
            new_inception = new_inception, clf_SVM_new_inception = clf_SVM_new_inception,
            pca = pca_pre_svm):

    image_path = list([img])
    text_img = Final_textreader(image_path)
    pred_clf_inception = inception_one_image(img)
    #pred_clf_inception = pre_pred_clf_inception.argmax(axis=1)

    features_img = extract_features_keras(image_path)
    features_img_pca = pca.transform(features_img)
    pred_clf_svm_inception = clf_SVM_new_inception.predict_proba(features_img_pca)
    if text_img == ['']:
        print('Text Mining classifier did not find any text on the image')
        total_pred = pd.DataFrame(index=['prédiction 1', 'prédiction 2', 'prédiction 3'],
                            columns=['inception', 'SVM_inception'])
        total_pred.inception = classement_predictions(pre_pred_clf_inception)
        total_pred.SVM_inception = classement_predictions(pred_svm_inception)
    else :
        print('Some text has been found on the image')
        df_text_img=Corpus_dropna(text_img)
        Filtered_text_img=text_processer(df_text_img.Text,stop_words=stop_words,stemmer=None)
        Text_img_count=countv.transform(Filtered_text_img)
        text_img_to_pred=tformer.transform(Text_img_count)

        pred_clf_textmining = clf_Text.predict_proba(text_img_to_pred)

        total_pred = pd.DataFrame(index=['prédiction 1', 'prédiction 2', 'prédiction 3'],
                            columns=['Text', 'inception', 'SVM_inception'])

        total_pred.Text, proba_textmining, label_proba_textmining = classement_predictions(pred_clf_textmining)
        total_pred.inception, proba_clf_inception, label_proba_clf_inception = classement_predictions(pred_clf_inception)
        total_pred.SVM_inception, proba_clf_svm_inception, label_proba_clf_svm_inception = classement_predictions(pred_clf_svm_inception)
        affichage_proba(proba_textmining=proba_textmining, proba_clf_inception=proba_clf_inception,
                        proba_clf_svm_inception = proba_clf_svm_inception, label_proba_textmining = label_proba_textmining,
                          label_proba_clf_inception = label_proba_clf_inception, label_proba_clf_svm_inception= label_proba_clf_svm_inception)
    return total_pred

def affichage_proba(proba_textmining, proba_clf_inception, proba_clf_svm_inception,
                    label_proba_textmining, label_proba_clf_inception, label_proba_clf_svm_inception):

    proba_et_label_text = pd.DataFrame({'proba_Text' : proba_textmining[0],'label_Text' : label_proba_textmining})
    proba_et_label_inception = pd.DataFrame({'proba_inception' : proba_clf_inception[0],'label_inception' : label_proba_clf_inception})
    proba_et_label_svm_inception = pd.DataFrame({'proba_SVM_inception' : proba_clf_svm_inception[0],'label_SVM_inception' : label_proba_clf_svm_inception})

    text = ColumnDataSource(proba_et_label_text)
    inception = ColumnDataSource(proba_et_label_inception)
    svm_inception = ColumnDataSource(proba_et_label_svm_inception)

    hover_text = HoverTool(tooltips=[("probabilité ", "@proba_Text")])
    hover_inception = HoverTool(tooltips=[("probabilité ", "@proba_inception")])
    hover_SVM_inception = HoverTool(tooltips=[("probabilité ", "@proba_SVM_inception")])

    fig1 = figure(plot_width =1000, plot_height=400, x_range = label_proba_textmining)
    fig1.vbar(x='label_Text', top= 'proba_Text', source = text, width=0.5, fill_color = '#45A7E2', line_color = '#45A7E2')
    fig1.xaxis.major_label_orientation = 0.7
    fig1.add_tools(hover_text)
    tab1 = Panel(child = fig1, title='Text_mining_proba')

    fig2 = figure(plot_width =1000, plot_height=400, x_range = label_proba_clf_inception)
    fig2.vbar(x='label_inception', top= 'proba_inception', source = inception, width=0.5, fill_color = '#E74C3C', line_color = '#E74C3C')
    fig2.xaxis.major_label_orientation = 0.7
    fig2.add_tools(hover_inception)
    tab2 = Panel(child = fig2, title='Inception_proba')

    fig3 = figure(plot_width =1000, plot_height=400, x_range = label_proba_clf_svm_inception)
    fig3.vbar(x='label_SVM_inception', top= 'proba_SVM_inception', source = svm_inception, width=0.5, fill_color = '#2ECC71', line_color = '#2ECC71')
    fig3.xaxis.major_label_orientation = 0.7
    fig3.add_tools(hover_SVM_inception)
    tab3 = Panel(child = fig3, title='SVM_inception_proba')


    onglet = Tabs(tabs=[tab1,tab2,tab3])
    show(onglet)



             #######################################################
             ################### MAIN CODE #########################
             #######################################################

clear_output()
print('******************************************************************')
print('******************************************************************')
print('***************** Welcome in a Book Classifier *******************')
print('********* You can upload a book cover in the function ************')
print('******************************************************************')
print('**************** prediction() ** or ** best_pred()****************')
print('******************************************************************')
print('************ if you want to see our statistics on ****************')
print('******************************************************************')
print('***************** -- Inception CNN model *************************')
print('**************************** or **********************************')
print('******************* -- Text Mining model *************************')
print('******************************************************************')
print('************ use see_statistics() without any arguments **********')
print('******************************************************************')
print('********************** Please enjoy ! *****************************')
print('******************************************************************')



def best_pred(img,choice_matrix=Choice_matrix, classe=classe, stopwords=stop_words, clf_Text = clf_TextMining,
            new_inception = new_inception, clf_SVM_new_inception = clf_SVM_new_inception,
            pca = pca_pre_svm):
    pred=prediction(img, classe=classe, stopwords=stopwords, clf_Text = clf_Text,
            new_inception = new_inception, clf_SVM_new_inception = clf_SVM_new_inception,
            pca = pca)
    if(len(pred.columns)==3):
        resultats=pd.DataFrame(columns=['Best Predictions'],
                            index=['Top 1', 'Top 2', 'Top 3'])
        Top1f=choice_matrix[str(pred.Text[0])][pred.SVM_inception[0]]
        if(pred.Text[0]!=pred.SVM_inception[0]):
            if(Top1f!=classe[pred.Text[0]]):
                 Top2f=classe[pred.Text[0]]
            else:
                Top2f=classe[pred.SVM_inception[0]]
            Top2Text=classe[pred.Text[1]]
            Top2SVM=[pred.SVM_inception[1]]
            if(Top2Text!=Top1f and Top2Text!=Top2f):
                if(Top2SVM!=Top1f and Top2SVM!=Top2f):
                    Top3f=choice_matrix[str(pred.Text[1])][pred.SVM_inception[1]]
                else:
                    Top3f=Top2Text
            else:
                if(Top2SVM!=Top1f and Top2SVM!=Top2f):
                    Top3f=Top2SVM
                else:
                    Top3f=choice_matrix[str(pred.Text[2])][pred.SVM_inception[2]]
        else:
            Top2f=choice_matrix[str(pred.Text[1])][pred.SVM_inception[1]]
            if(pred.Text[1]!=pred.SVM_inception[1]):
                if(Top2f!=classe[pred.Text[1]]):
                    Top3f=classe[pred.Text[1]]
                else:
                    Top3f=classe[pred.SVM_inception[1]]
            else:
                Top3f=choice_matrix[str(pred.Text[2])][pred.SVM_inception[2]]
        resultats["Best Predictions"][0]=Top1f
        resultats["Best Predictions"][1]=Top2f
        resultats["Best Predictions"][2]=Top3f
        return resultats




## end
