########### On remplit X_train par le texte extrait des images d'entrainement
########### On remplit X_test par le texte extrait des images test



import pytesseract
import cv2
import numpy as np
import pandas as pd
import os


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

# Create a list of strings 
def generate_text_from_image(images):
    tensor=list()
    
# images = list of all images filepath ex: test_data.path
# size = int, wanted picture size 
    
    for i in range(len(images)):
# Image reading       
        image=cv2.imread(images[i])
# Image transformation in grayscale      
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Extracting the words from the image
        result = pytesseract.image_to_string(gray_image)
# Adding the string result to the list
        tensor.append(result)
    
    return tensor


############################################################################


# Generating Train/Test datasets 
# X_test and X_train are data frames with a single column containing the strings from the images


X_test=pd.DataFrame({'Texte sur l\'image':generate_text_from_image(test_data.path)})
X_train=pd.DataFrame({'Texte sur l\'image':generate_text_from_image(train_data.path)})
y_test=np_utils.to_categorical(test_data.CATEGORY_ID)
y_train=np_utils.to_categorical(train_data.CATEGORY_ID)



############################################################################
