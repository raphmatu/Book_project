# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:36:06 2018

@author: Raphaël
"""


import pandas as pd


# récupération des data sets Train et Test
df_train = pd.read_csv("chemin vers le fichier book30-listing-tain.csv", 
                       engine = "python", header = None)

df_test = pd.read_csv("chemin vers le fichier book30-listing-test.csv", 
                      engine = "python", header = None)

# Changement du nom des colonnes
df_train.rename(columns={0:"Amazon_index",1:"Filename",2:"Image_url",3:"Title",4:"Author",5:"Category_id",6:"Category"}, inplace = True)
df_test.rename(columns={0:"Amazon_index",1:"Filename",2:"Image_url",3:"Title",4:"Author",5:"Category_id",6:"Category"}, inplace = True)

df_train.columns



