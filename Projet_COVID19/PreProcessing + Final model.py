# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:24:38 2021

@author: Pirlouit
"""

################################################# IMPORTATION BIBLIOTHEQUES ################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
#############################################################################################################
################################################# PREPROCESSING ################################################
#############################################################################################################

#Objectif : 1) Mettre les donnees dans un format propice au ML

data = pd.read_excel('dataset.xlsx')
df = data.copy()
print(df.shape)
#####Nettoyage rapide du df comme dans la partie EDA#####


missing_rate = df.isna().sum()/df.shape[0] #variable qui calcule le taux de valeur manquante dans une colonne
#On cree des ensembles de colonnes comme en EDA
blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate >0.88)])
viral_columns = list(df.columns[(missing_rate < 0.80) & (missing_rate > 0.75)])
key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result'] #On modifie le df en gardant uniquement ces colonnes
df = df[blood_columns + viral_columns + key_columns]
# print(df.head())


####TrainSet/Nettoyage/Encodage####
#Creation trainset/testset
from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(df,test_size=0.2,random_state=0) #on cree un train et test sets

#On verifie que les proportions positif/negatif sont respectes(comme dans la dataset d origine)

print(trainset['SARS-Cov-2 exam result'].value_counts())
print(testset['SARS-Cov-2 exam result'].value_counts())

#ENCODAGE
#On ne touche plus au testset dans cette partie, on joue uniquement avec le trainset
#On a vu dans le EDA qu'il y avait seulement 4 categories (positif,negatif,detected,not detected)
#On cree un dictionnaire pour associer chaque valeur à un nombre


# code = {'negative' : 0,
#         'positive' : 1,
#         'detected' : 1,
#         'not_detected' :0
#         }

# for col in df.select_dtypes('object') :
#     df[col] = df[col].map(code)  #on utilise la fonction map de pandas pour appliquer le dictionnaire a chaque colonne de type object
    
#Pour faciliter son utilisation, on va creer une fonction encodage qui fait la meme chose
def encodage(df):
    code = {'negative':0,
            'positive':1,
            'not_detected':0,
            'detected':1}
    
    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)
        
    return df
# print(df.dtypes.value_counts()) #on verifie qu'il ne reste plus que des valeurs numeriques

#De la meme facon on cree une fonction pour gerer les valeurs manquantes

def imputation(df) :
    #df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leukocytes'].isna())

   # return df.fillna(-999)
   df =df.dropna(axis=0)
   return df


def feature_engineering(df):
    #on cree une variable est malade comme dans le EDA pour savoir si la personne est teste positive a au moins un autre virus que le covid

    df['est malade'] = df[viral_columns].sum(axis=1) >= 1
    df = df.drop(viral_columns, axis=1)
    return df
def preprocessing(df):
    
    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)
    
    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']
    
    print(y.value_counts())
    
    return X, y

X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)

####MODELISATION####
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#model = DecisionTreeClassifier(random_state=0)
model_1 = RandomForestClassifier(random_state=0)
model_2 = make_pipeline(PolynomialFeatures(2,include_bias=False),SelectKBest(f_classif,k=10),RandomForestClassifier(random_state=0))
#mettre le include bias = False pour eviter les messages d'erreurs avec le selectKBest
#Procedure d'evaluation
from sklearn.metrics import f1_score, confusion_matrix, classification_report #le metrics f1 permet de voir les faux positif/faux negatif diagnostique par notre modele
from sklearn.model_selection import learning_curve  #pour detecter over/under fitting

# def evaluation(model) : 
#     model.fit(X_train,y_train)
#     y_pred = model.predict(X_test)
    
#     print(confusion_matrix(y_test,y_pred))
#     print(classification_report(y_test,y_pred))
    
    
    
# evaluation(model)
#On constate que le modele est peu fiable, on va donc ajouter les learning curve pour voir si il y a over/underfitting

def evaluation(model) : 
    model.fit(X_train ,y_train)
    y_pred = model.predict(X_test)
    
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, train_sizes = np.linspace(0.1,1,10), scoring='f1' )
    
    plt.figure(figsize = (12,8))
    plt.plot(N,train_score.mean(axis=1),label='train score')
    plt.plot(N,val_score.mean(axis=1),label='validation score')
    plt.legend()
    
#evaluation(model_2) 
#print(testset.dtypes.value_counts()) #on verifie qu'il ne reste plus que des valeurs numeriques
#print(testset)
#On voit que le trainscore est a 100% et le valscore est bcp moins bon -> OVER FITTING
#On va d'abord tenter de fournir plus de données a la machine, donc on modifie la fonction imputation
#On voit qu avec fillna le score est encore moins bon !
#on va faire du feature selection
#print(model.feature_importances_)

# print(pd.DataFrame(model.feature_importances_,index=X_train.columns))
#pd.DataFrame(model_2[2].feature_importances_, index=X_train.columns).plot.bar(figsize=(12,8))
#on se rend compte que bcp de donnees ont peu d dimportance 
#premiere idee: on supprime les donnees viral et on test, on obtient pas de meilleurs resultats

#On peut changer de modele en prenzant un RandomForest qui est efficace contre l'overfitting
#On cree une pipeline et on joue rapidement avec les parametres et on obtient des resultats interessant avec un f1 score de 0.40

#############################################################################################################
##################################### MODELE DE MACHINE LEARNING#############################################
#############################################################################################################

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
preprocesor = make_pipeline(PolynomialFeatures(3,include_bias=False),SelectKBest(f_classif,k=26))
AdaBoost = make_pipeline(preprocesor, AdaBoostClassifier(random_state=0) )
SVM = make_pipeline(preprocesor, StandardScaler(),SVC(random_state=0) )
KNN = make_pipeline(preprocesor,StandardScaler(), KNeighborsClassifier() )
RandomForest = make_pipeline(preprocesor, RandomForestClassifier(random_state=0) )

# model_dict = {'AdaBoost' : AdaBoost,
#             'SVC' : SVC,
#             'KNN' : KNN,
#             'RandomForest' : RandomForest
#             }
     
# for name,model in model_dict.items():
#     print('--------',name,'----------')
#     evaluation(model)
   
#on peut ainsi determiner quel modele est le plus adapte 
#on regarde aussi les courbes de recall/score pour savoir quel modele est le plus prometteur

####ON CHOISIT D OPTIMISER LE MODELE D ADABOOST###
#au depart on a precision=0.64 , recall=0.44 et f1-score=0.52
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# print(AdaBoost)  
# hyper_params = {'adaboostclassifier__base_estimator' : [ExtraTreesClassifier()],
#                 'adaboostclassifier__n_estimators' : range(125,145),
#                 'adaboostclassifier__learning_rate' : [0.9,1],
#                 'adaboostclassifier__algorithm' : ['SAMME'],
#                 'pipeline__polynomialfeatures__degree':[3],
#                 'pipeline__selectkbest__k': range(25, 30)}

hyper_params = {'adaboostclassifier__base_estimator' : [ExtraTreesClassifier()],
                'adaboostclassifier__n_estimators' : [126],
                'adaboostclassifier__learning_rate' : [1],
                'adaboostclassifier__algorithm' : ['SAMME'],
                'pipeline__polynomialfeatures__degree':[3],
                'pipeline__selectkbest__k': [28]}

# best_hyper_params = {'pipeline__selectkbest__k': range(28),
#                      'pipeline__polynomialfeatures__degree': 3,
#                      'adaboostclassifier__n_estimators': 126,
#                      'adaboostclassifier__learning_rate': 1,
#                      'adaboostclassifier__base_estimator': ExtraTreesClassifier(),
#                      'adaboostclassifier__algorithm': 'SAMME'}

grid = RandomizedSearchCV(AdaBoost, hyper_params, scoring='recall', cv=4,
                          n_iter=1)
grid.fit(X_train, y_train)

# print(grid.best_params_)

# y_pred = grid.predict(X_test)

# print(classification_report(y_test, y_pred))

#evaluation(grid.best_estimator_)
#on obtient en best estimatr :'pipeline__selectkbest__k': 396, 'pipeline__polynomialfeatures__degree': 3, 'adaboostclassifier__n_estimators': 260, 'adaboostclassifier__learning_rate': 0.8, 'adaboostclassifier__algorithm': 'SAMME'}
#pipeline__selectkbest__k': 36, 'pipeline__polynomialfeatures__degree': 3, 'adaboostclassifier__n_estimators': 135, 'adaboostclassifier__learning_rate': 0.8, 'adaboostclassifier__algorithm': 'SAMME'} -> recall 0.44 f1=0.52
#print(AdaBoost.get_params().keys())

#{'pipeline__selectkbest__k': 28, 'pipeline__polynomialfeatures__degree': 3, 'adaboostclassifier__n_estimators': 126, 'adaboostclassifier__learning_rate': 1, 'adaboostclassifier__base_estimator': ExtraTreesClassifier(), 'adaboostclassifier__algorithm': 'SAMME'}
#recall de 0.50 et f1 de 0.55



#### PRECISION RECALL CURVE ####

# preprocesor = make_pipeline(PolynomialFeatures(3,include_bias=False),SelectKBest(f_classif,k=28))
# AdaBoost_f = make_pipeline(preprocesor, AdaBoostClassifier(random_state=0,n_estimators=26,learning_rate=1,base_estimator=ExtraTreesClassifier(),algorithm='SAMME') )
# AdaBoost_f.fit(X_train,y_train)
# evaluation(AdaBoost_f)
evaluation(grid.best_estimator_)
from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_train, grid.best_estimator_.decision_function(X_train))
plt.figure()
plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')

plt.legend()


#ON DEFINIT LE MODEL FINAL#
def model_final(model, X, threshold):
    return model.decision_function(X) > threshold

from sklearn.metrics import recall_score
y_predict = model_final(grid.best_estimator_, X_test, threshold=-0.5)
print(f1_score(y_test, y_predict) )
print(recall_score(y_test,y_predict))

#pas de courbe de precision recall pour le modele AdaBoost, on optimies le modele de SVM qui etait prometteur 
#au vu des learning curves
SVM = make_pipeline(preprocesor, StandardScaler(), SVC(random_state=0))
hyper_params = {'svc__gamma':[1e-3, 1e-4, 0.0005],
                'svc__C':[1, 10, 100, 1000, 3000], 
               'pipeline__polynomialfeatures__degree':[2, 3],
               'pipeline__selectkbest__k': range(45, 60)}


grid = RandomizedSearchCV(SVM, hyper_params, scoring='recall', cv=4,
                          n_iter=20)

grid.fit(X_train, y_train)

print(grid.best_params_)

y_pred = grid.predict(X_test)


y_pred = model_final(grid.best_estimator_, X_test, threshold=-1)
print(classification_report(y_test, y_pred))

from sklearn.metrics import recall_score
print('le recall final est : ',recall_score(y_test, y_pred))
evaluation(model_final(grid.best_estimator_, X_test, threshold=-1))





