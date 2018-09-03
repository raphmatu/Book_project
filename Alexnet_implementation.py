import cv2
from skimage.color import rgba2rgb
from skimage.color import grey2rgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
%matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.utils import np_utils
from keras.applications.xception import Xception
from keras.models import Model

# Selecting working directory
work_dir="/home/derazejerome/Desktop/book_project/"

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

############################################################################
# Generating Train/Test datasets 
# Generating a list of 51300 arrays can be very long (and exceed available memory)
# Do not try it with size>28, see after Alexnet for loading smaller parts of the dataset
Xtest=generate_image_data(test_data.path,size=28)
Xtrain=generate_image_data(train_data.path,size=28)
ytest=np_utils.to_categorical(test_data.CATEGORY_ID)
ytrain=np_utils.to_categorical(train_data.CATEGORY_ID)

############################################################################

#  AlexNet implementation
Alexmodel = Sequential()

# 1st Convolutional Layer
Alexmodel.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
Alexmodel.add(Activation('relu'))
# Pooling
Alexmodel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
Alexmodel.add(BatchNormalization())

# 2nd Convolutional Layer
Alexmodel.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
Alexmodel.add(Activation('relu'))
# Pooling
Alexmodel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
Alexmodel.add(BatchNormalization())

# 3rd Convolutional Layer
Alexmodel.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
Alexmodel.add(Activation('relu'))
# Batch Normalisation
Alexmodel.add(BatchNormalization())

# 4th Convolutional Layer
Alexmodel.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
Alexmodel.add(Activation('relu'))
# Batch Normalisation
Alexmodel.add(BatchNormalization())

# 5th Convolutional Layer
Alexmodel.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
Alexmodel.add(Activation('relu'))
# Pooling
Alexmodel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
Alexmodel.add(BatchNormalization())

# Passing it to a dense layer
Alexmodel.add(Flatten())
# 1st Dense Layer
Alexmodel.add(Dense(4096, input_shape=(224*224*3,)))
Alexmodel.add(Activation('relu'))
# Add Dropout to prevent overfitting
Alexmodel.add(Dropout(0.4))
# Batch Normalisation
Alexmodel.add(BatchNormalization())

# 2nd Dense Layer
Alexmodel.add(Dense(4096))
Alexmodel.add(Activation('relu'))
# Add Dropout
Alexmodel.add(Dropout(0.4))
# Batch Normalisation
Alexmodel.add(BatchNormalization())

# 3rd Dense Layer
Alexmodel.add(Dense(1000))
Alexmodel.add(Activation('relu'))
# Add Dropout
Alexmodel.add(Dropout(0.4))
# Batch Normalisation
Alexmodel.add(BatchNormalization())

# Output Layer
Alexmodel.add(Dense(30))
Alexmodel.add(Activation('softmax'))

# Compile
Alexmodel.compile(loss='categorical_crossentropy', optimizer='adam',\
 metrics=['accuracy'])

Alexmodel.summary()

############################################################################

# Training Alexnet with the full 51300 dataset if Xtrain exceeds memory

def fraction_training(images,model,y,size=224,fraction_number=10,batch_size=200,epochs=1):
    
# images = list of all images filepath ex: test_data.path
# model = a CNN model to train
# y = an array of labels corresponding to the images (use np_utils.to_categorial)
# fraction_number = the number of subsets to partition the data into
# batch_size = number of images to use simultaneously for trainig
# epochs = number of iterations of the CNN
    for k in range(epochs):
        for j in range(fraction_number):
            tensor=[]
            for i in range(round(len(images)/fraction_number)*j,round(len(images)/fraction_number)*(j+1)):
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
            yfrac=y[round(len(images)/fraction_number)*j:round(len(images)/fraction_number)*(j+1),:]
            model.fit(tensor,yfrac,batch_size=batch_size,epochs=1)

############################################################################