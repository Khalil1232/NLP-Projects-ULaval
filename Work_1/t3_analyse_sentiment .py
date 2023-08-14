#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import json
import nltk

from nltk import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")

reviews_dataset = {
    'train_pos_fn' : "./senti_train_positive.txt",
    'train_neg_fn' : "./senti_train_negative.txt",
    'test_pos_fn' : "./senti_test_positive.txt",
    'test_neg_fn' : "./senti_test_negative.txt"
}


def load_reviews(filename):
    with open(filename, 'r') as fp:
        reviews_list = json.load(fp)
    return reviews_list



def display_confusion_matrix(confusion_matrix, classes):
    
    df_cm = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    f, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.ylabel('Vrai étiquette ')
    plt.xlabel('Étiquette prédite')
    

    
def stem(string):
        porter_stemmer = nltk.stem.porter.PorterStemmer()
        tokenizedStr = word_tokenize(string)
        stemsStr = list(map(lambda word: porter_stemmer.stem(word), tokenizedStr))
        space = " "
        return space.join(stemsStr)

def lemm(string):    
        wordnet_lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        tokenizedStr = word_tokenize(string)
        lemsStr = []
        
        for token in tokenizedStr:
            lemmatized_word = wordnet_lemmatizer.lemmatize(token)
            lemsStr.append(lemmatized_word)
       
        return " ".join(lemsStr)


def train_and_test_classifier(dataset, model='NB', normalization='words'):
    
    """
    :param dataset: un dictionnaire contenant le nom des 4 fichiers utilisées pour entraîner et tester les classificateurs. Voir variable reviews_dataset.
     :param model: le type de classificateur. NB = Naive Bayes, LR = Régression logistique.
    :param normalization: le prétraitement appliqué aux mots des critiques (reviews)
                 - 'word': les mots des textes sans normalization.
                 - 'stem': les racines des mots obtenues par stemming.
                 - 'lemma': les lemmes des mots obtenus par lemmatisation.
    :return: un dictionnaire contenant 3 valeurs:
                 - l'accuracy à l'entraînement (validation croisée)
                 - l'accuracy sur le jeu de test
                 - la matrice de confusion calculée par scikit-learn sur les données de test
    """

    # Votre code...
    
    X = np.append(partitions['train_pos_fn'],partitions['train_neg_fn'])        
    y = np.append(np.ones(len(partitions['train_pos_fn'])), np.zeros(len(partitions['train_neg_fn'])))
    
    X_test= np.append(partitions['test_pos_fn'],partitions['test_neg_fn']) 
    y_test = np.append(np.ones(len(partitions['test_pos_fn'])), np.zeros(len(partitions['test_neg_fn'])))
    
    vectorizer = CountVectorizer(stop_words = stop, tokenizer = word_tokenize,lowercase=True)
    if normalization=='words':
        
        XVectorised = vectorizer.fit_transform(X)
        X_test_vectorized = vectorizer.transform(X_test)
    elif normalization=='stem':
        XStem=[]
        X_test_Stem=[]
        
        for i in range(len(X)):
            XStem.append(stem(X[i]))
        for i in range(len(X_test)):
            X_test_Stem.append(stem(X_test[i]))
        
        XVectorised = vectorizer.fit_transform(XStem)
        X_test_vectorized = vectorizer.transform(X_test_Stem)
    else:
        Xlemm=[]
        X_test_lemm=[]
        
        for i in range(len(X)):
            Xlemm.append(lemm(X[i]))
        for i in range(len(X_test)):
            X_test_lemm.append(lemm(X_test[i]))
        
        
        XVectorised = vectorizer.fit_transform(Xlemm)
        X_test_vectorized = vectorizer.transform(X_test_lemm)
    
        
    
    
    
    
    
    if(model == 'NB'):
        nb_classifier = MultinomialNB()
          
    else:
        nb_classifier = LogisticRegression(solver = 'lbfgs')
             
    #Apprentissage final sur tout le jeu de donnée
    nb_classifier.fit(XVectorised, y)
    
    
    # Evaluation de l'algorithme sur X_test vectorized
    y_pred = nb_classifier.predict(X_test_vectorized)
    
    
    
    
    # Matrice de confusion 
    cm = confusion_matrix(y_test, y_pred)
    
    
    classes = nb_classifier.classes_
    display_confusion_matrix(cm, classes)
    
    

    # Les résultats à retourner 
    results = dict()
    
    results['accuracy_train'] = np.mean(cross_val_score(nb_classifier, XVectorised, y))
    results['accuracy_test'] = accuracy_score(y_test, y_pred)
    results['confusion_matrix'] = cm  # la matrice de confusion obtenue de Scikit-learn
    return results


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme vous le souhaitez.
    # Contenu des fichiers de données
    splits = ['train_pos_fn', 'train_neg_fn', 'test_pos_fn', 'test_neg_fn']
    stop = set(stopwords.words('english'))
    print("Taille des partitions du jeu de données")
    partitions = dict()
    for split in splits:
        partitions[split] = load_reviews(reviews_dataset[split])
        print("\t{} : {}".format(split, len(partitions[split])))

    # Entraînement et évaluation des modèles
    results = train_and_test_classifier(reviews_dataset, model='NB', normalization='lemm')
    print("Accuracy - entraînement: ", results['accuracy_train'])
    print("Accuracy - test: ", results['accuracy_test'])
    print("Matrice de confusion: ", results['confusion_matrix'])
    print("\n\nVersion graphique de la matrice de confusion") 


# In[ ]:





# In[ ]:





# In[ ]:




