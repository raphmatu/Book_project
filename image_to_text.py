import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#import the metrics class for the performance measurement
from sklearn import metrics


# Selecting working directory
work_dir="/Users/adrianivasiuc/Data Science/Formation Data Scientist/Projet/Book_project/"


# Opening csv files (must be in working directory)
column_names=["AMAZON_INDEX","FILENAME","IMAGE_URL","TITLE","AUTHOR","CATEGORY_ID","CATEGORY"]
test_data=pd.read_csv(work_dir+"book30-listing-test.csv",encoding="latin1",sep=",",header=None)
test_data.columns=column_names
train_data=pd.read_csv(work_dir+"book30-listing-train.csv",encoding="latin1",sep=",",header=None)
train_data.columns=column_names
# Creating a column with filepath to all images
# Absolute path to the folders contatining image categories
test_dir=work_dir+"Database_test"
train_dir=work_dir+"Database_train"
# Generating list of filepaths
filepath=list()
for i in range(len(test_data.FILENAME)):
    filepath.append(os.path.join(test_dir,test_data.CATEGORY[i],test_data.FILENAME[i]))
test_data["path"]=filepath
filepath=list()
for i in range(len(train_data.FILENAME)):
    filepath.append(os.path.join(train_dir,train_data.CATEGORY[i],train_data.FILENAME[i]))
train_data["path"]=filepath

############################################################################

############################################################################
# Create a list of strings 
def generate_text_from_image(images):
    tensor=list()
    
# images = list of all images filepath ex: test_data.path
    
    for i in range(len(images)):
# Image reading       
        image=cv2.imread(images[i])
# We check if the image is NoneType before doing any action on it
        if image is not None:
# Image transformation in grayscale
            if (len(image.shape)!=2):
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else :
                gray_image = image
# Extracting the words from the image
        result = pytesseract.image_to_string(gray_image)
# Adding the string result to the list
        tensor.append(result)
    
    return tensor


############################################################################
#                         DATA EXTRACTION
############################################################################


# Generating Train/Test datasets 
# X_test and X_train are data frames with a single column containing the strings from the images

# It takes a lot of time to have the X_test and X_train dataframes
X_test=pd.DataFrame({'Texte sur les images test':generate_text_from_image(test_data.path)})
X_train=pd.DataFrame({'Texte sur les images train':generate_text_from_image(train_data.path)})

y_test=test_data.CATEGORY_ID
y_train=train_data.CATEGORY_ID


############################################################################
#                         DATA PREPARATION
############################################################################


# We add y_train and y_test to the dataframes in order to drop NAs
X_train["y_train"] = y_train
X_test["y_test"] = y_test
X_train = X_train.dropna()
X_test = X_test.dropna()

# then we refit y_train and y_test ...
y_train = X_train.y_train
y_test = X_test.y_test

# ... and the train and test datasets
X_train = X_train.drop('y_train',axis=1)
X_test = X_test.drop('y_test',axis=1)

# we split the characters of the datasets into a list of characters
# in order to have each word apart
X_test = X_test.applymap(lambda x: x.split())
X_train = X_train.applymap(lambda x: x.split())

# function which converts a list of text in text
def TextList_to_Text(liste):
    texte = ""
    for element in liste:
        texte = texte+" "+element
    return texte

X_test = X_test.applymap(lambda x: TextList_to_Text(x))
X_train = X_train.applymap(lambda x: TextList_to_Text(x))
# Conversion of the two datasets in order to apply CountVectorizer()
# ( which can used only on text)

# Transformation of the datsets in order to apply classification models
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train["Texte sur les images train"])
X_test_counts = count_vect.transform(X_test["Texte sur les images test"])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


############################################################################
#                         DATA MODELISATION
############################################################################


# We will test 4 different models ...
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# With cross validation
CV = 5

# Initialisation of a dataframe 
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
# We fill a list with the model name with its accuracy
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, X_train_tfidf, y_train, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

# We fit the dataframe with the list filled before    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# Plot of the different models with their scores
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

# scores of the 4 models
print(cv_df.groupby('model_name').accuracy.mean())

# Display of a confusion matrix with our best model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
