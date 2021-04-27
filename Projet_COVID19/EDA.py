# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:04:43 2021

@author: Pirlouit
"""
################################################# IMPORTATION BIBLIOTHEQUES ################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
################################################# EXPLORATORY DATA ANALYSIS ################################################

#### Analyse de la forme ####
pd.set_option('display.max_row',111) #permet d'afficher toutes les lignes dataset 
#pd.set_option('display.max_column',4) #permet d'afficher toutes les columns du dataset 

data = pd.read_excel('dataset.xlsx')
df = data.copy()
#df['SARS-Cov-2 exam result'].replace(('negative','positive'),(0,1),inplace=True) #on remplace negatif par 0/pos par 1
print(df.head())   #target = SARS-Cov-2 exam result
print(df.shape)  #5644lignes et 111 colonnes


#IDENTIFICATION TYPES DE VARIABLES
#print(df.dtypes) #affiche les types des variables
#print(df.dtypes.value_counts()) #compte les differents types de variable

#IDENTIFICATION DES VALEURS MANQUANTES

plt.figure(figsize=(20,20))
sns.heatmap(df.isna(),cbar=False)  #permet de visualiser l'ensemble du dataset avec les données manquantes(en blanc)
plt.imshow(df.isna())
#print(df.isna().sum()/df.shape[0])  #calcule pourcentage de donnee manquante pour chaque colonne
#print((df.isna().sum()/df.shape[0]).sort_values())  #trie les colonnes selon le pourcentage de donnees manquantes(croissant)


#### Analayse du fond ####
#On nettoie rapidement notre dataset
# print((df.isna().sum()/df.shape[0] < 0.90))  #True pour valeurs ou il manque moins de 90% des donnees, False sinon
# print(df.columns[(df.isna().sum()/df.shape[0] < 0.90)]) #Selectionne uniquement les colonnes qui ont moins de 0,9 de NaN
# print(df[df.columns[(df.isna().sum()/df.shape[0] < 0.90)]]) #Affiche le nouveau tableau contenant seulement les colonnes choisies
# df = df[df.columns[(df.isna().sum()/df.shape[0] < 0.90)]] #on applique le changement au dataset
# df = df.drop('Patient ID',axis=1) #on supprime la colonne PatientID (car inutile ici)
# sns.heatmap(df.isna(),cbar=False)  #permet de visualiser l'ensemble du dataset avec les données manquantes(en blanc)

#VISUALISATION DE LA TARGET
# print(df['SARS-Cov-2 exam result'].value_counts(normalize=True)) #normalise = True permet de convertir en pourcentage

#SIGNIFICATION DES VARIABLES
#Histogramme des variables continues
# for col in df.select_dtypes('float'):  #select_dtypes permet de selectionner le type de variable qu'on veut
#     plt.figure()
#     sns.distplot(df[col])
# plt.figure()
# sns.distplot(df['Patient age quantile'])
# print(df['Patient age quantile'].value_counts())

#Variables qualitatives 
#Il faut voir les differentes categories qu'il y a dans chaque variable

# print(df['SARS-Cov-2 exam result'].unique()) #retourne un tableau qui précise les differentes categories presentes pour cette variable
# for col in df.select_dtypes('object'):
#     print(f'{col :-<50} {df[col].unique()}')  #syntaxe pour un affichage plus agreable
# #Il faut maintenant compter le nb de valeurs dans chaque categorie

# for col in df.select_dtypes('object'):
#     plt.figure()
#     df[col].value_counts().plot.pie()


#Relation Target/Variables
#Création des sous ensemble positif et negatif

positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']
negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']

#Creation des sous ensemble Blood et Viral

missing_rate = df.isna().sum()/df.shape[0] 
#print(df.columns[(missing_rate < 0.9) & (missing_rate > 0.88) ])
blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate > 0.88) ]
viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75) ]

#On visualise les differentes relations entre Target/Features
#TARGET/BLOOD

# for col in blood_columns:
#     plt.figure()
#     sns.distplot(positive_df[col],label='positive')
#     sns.distplot(negative_df[col],label='negative')
#     plt.legend()

#TARGET/AGE

# plt.figure()
# sns.distplot(positive_df['Patient age quantile'],label='positive')
# sns.distplot(negative_df['Patient age quantile'],label='negative')
# plt.legend()
# plt.figure()

# sns.countplot(x='Patient age quantile',hue='SARS-Cov-2 exam result',data=df)
# plt.legend()


#TARGET/VIRAL

# print(pd.crosstab(df['SARS-Cov-2 exam result'], df['Influenza A']))
# for col in viral_columns:
#     plt.figure()
#     sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')

####ANALYSE PLUS DETAILLEE####

#RELATION VARIABLE/VARIABLE

#Relations blood/blood

#sns.pairplot(df[blood_columns])
# #Méthode plus rapide pour tout visualiser:
# sns.heatmap(df[blood_columns].corr())
# #Ou bien quasi pareil : 
# sns.clustermap(df[blood_columns].corr())

#Relation blood/age

# for col in blood_columns:
#     plt.figure()
#     sns.lmplot(x='Patient age quantile',y=col,hue='SARS-Cov-2 exam result',data=df)

#print(df.corr()['Patient age quantile'].sort_values())

#Relation viral/viral
#Test fiabilite du rapid test
#pd.crosstab(df['Influenza A'], df['Influenza A, rapid test'])
#pd.crosstab(df['Influenza B'], df['Influenza B, rapid test'])
   
#Relation Viral/sanguin
#Creation d'une nouvelle variable 'est malade'
df['est malade'] = np.sum(df[viral_columns[:-2]] == 'detected',axis=1) >= 1  #on enleve les deux dernieres colonnes qui correspondent au rapidtest

#On cree des nouveaux dataset pour separer les malades des non malades
malade_df = df[df['est malade']== True]
non_malade_df = df[df['est malade']== False]

# for col in blood_columns:
#     plt.figure()
#     sns.distplot(malade_df[col],label='malade')
#     sns.distplot(non_malade_df[col],label='non malade')
#     plt.legend()



#Relation hospitalation/est malade


# def hospitalisation(df):
#     if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
#         return 'surveillance'
#     elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
#         return 'soins semi-intensives'
#     elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
#         return 'soins intensifs'
#     else:
#         return 'inconnu'

# df['statut'] = df.apply(hospitalisation,axis=1)
# print(df.head())

# for col in blood_columns:
#     plt.figure()
#     for cat in df['statut'].unique():
#         sns.distplot(df[df['statut']==cat][col], label=cat)
#     plt.legend()

# df1 = df[viral_columns[:-2]]
# df1['covid'] = df['SARS-Cov-2 exam result']
# df1.dropna()['covid'].value_counts(normalize=True)

# df2 = df[blood_columns]
# df2['covid'] = df['SARS-Cov-2 exam result']
# df2.dropna()['covid'].value_counts(normalize=True)



#####TEST DES HYPOTHESES ######
#TEST DE STUDENT
from scipy.stats import ttest_ind
#Pour utiliser ce Test, il faut que la proportion negatif/positif soit equilibre ce n'est pas le cas ici
#on doit utiliser une technique d'enchantillonage
#print(positive_df.shape)
#print(negative_df.shape)
# balanced_neg = negative_df.sample(positive_df.shape[0]) #car on prend autant de cas negatif qu'on en a de positif

# def t_test(col):
#     alpha = 0.02
#     stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
#     if p < alpha:
#         return 'H0 Rejetée'  #dans ce cas le taux sanguin est different pour les personnes positives et negatives
#     else :
#         return 0
    
# for col in blood_columns:
#     print(f'{col :-<50} {t_test(col)}')
    
