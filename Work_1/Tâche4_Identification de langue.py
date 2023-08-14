#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import glob
import os
import string
import unicodedata
import json

# import ajoutes

import math
from numpy import prod

datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
test_filename = './data/test_names.txt'  # le fichier contenant les données de test pour évaluer vos modèles

names_by_origin = {}  # un dictionnaire qui contient une liste de noms pour chaque langue d'origine
all_origins = []  # la liste des 18 langues d'origines de noms

BOS = "~"  # character used to pad the beginning of a name
EOS = "!"  # character used to pad the end of a name

modele_uni = {}
modele_bi = {}
modele_tri = {}


def find_files(path):
    """Retourne le nom des fichiers contenus dans un répertoire.
       glob fait le matching du nom de fichier avec un pattern - par ex. *.txt"""
    return glob.glob(path)


def get_origin_from_filename(filename):
    """Passe-passe qui retourne la langue d'origine d'un nom de fichier.
       Par ex. cette fonction retourne Arabic pour "./data/names/Arabic.txt". """
    return os.path.splitext(os.path.basename(filename))[0]


def unicode_to_ascii(s):
    """Convertion des caractères spéciaux en ascii. Par exemple, Hélène devient Helene.
       Tiré d'un exemple de Pytorch. """
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_names(filename):
    """Retourne une liste de tous les noms contenus dans un fichier."""
    with open(filename, encoding='utf-8') as f:
        names = f.read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def load_names():
    """Lecture des noms et langues d'origine d'un fichier. Par la suite,
       sauvegarde des noms pour chaque origine dans le dictionnaire names_by_origin."""
    for filename in find_files(datafiles):
        origin = get_origin_from_filename(filename)
        all_origins.append(origin)
        names = read_names(filename)
        names_by_origin[origin] = names


def train_classifiers():
    load_names()
    # Vous ajoutez à partir d'ici tout le code dont vous avez besoin
    # pour construire vos modèles N-grammes pour chacune des langues d'origines.
    #
    # Vous pouvez ajouter au fichier toutes les fonctions que vous jugerez nécessaire.
    # Merci de ne pas modifier les fonctions présentes dans ce fichier.
    #
    # À compléter - Fonction pour la construction des modèles unigrammes, bigrammes et trigrammes.
    #
    # Votre code à partir d'ici...
    
    
    for langue in all_origins:
        
        modele_uni[langue] = {}
        modele_bi[langue] = {}
        modele_tri[langue] = {}
        
        for name in names_by_origin[langue]:
            
            lettres = []
            
            lettres.extend(unicode_to_ascii(name))
            
            for uni in lettres:
                
                if uni in modele_uni[langue].keys():
            
                    modele_uni[langue][uni] += 1
                
                else:
                    
                    modele_uni[langue][uni] = 1
                    
            lettres.insert(0, BOS)
            lettres.append(EOS)
            
            for i in range(1, len(lettres)):
                
                bi = lettres[i-1] + lettres[i]
                
                if bi in modele_bi[langue].keys():
            
                    modele_bi[langue][bi] += 1
                
                else:
                    
                    modele_bi[langue][bi] = 1
                           
            lettres.insert(0, BOS)
            lettres.append(EOS)
            
            for i in range(2, len(lettres)):
                
                tri = lettres[i-2] + lettres[i-1] + lettres[i]
                
                if tri in modele_tri[langue].keys():
            
                    modele_tri[langue][tri] += 1
                
                else:
                    
                    modele_tri[langue][tri] = 1


def most_probable_origin(name, n=3):
    # Retourne la langue d'origine la plus probable du nom.
    # n désigne la longueur des N-grammes. Par ex n=3 --> trigramme
    # À compléter...
    
    langue_origine = None
    
    prob = 0
    
    
    for langue in all_origins:
    
        prob_temp = logprob(name, langue, n)
        
        if langue_origine == None or prob_temp > prob:
            
            langue_origine = langue
            
            prob = prob_temp
            
    '''
    
    for langue in all_origins:
        
        prob_temp = perplexity(name, langue, n)
        
        if langue_origine == None or prob_temp < prob:
            
            langue_origine = langue
            
            prob = prob_temp
    '''
    
    return langue_origine


def probUni(uni, origin):

    denom = sum(modele_uni[origin].values()) + len(modele_uni[origin].keys())
    
    if uni in modele_uni[origin].keys():
        
        num = modele_uni[origin][uni] + 1
    
    else:
        
        num = 1
    
    return num/denom

def probBi(bi, origin):
    
    lettres = []
    lettres.extend(bi)
    
    if lettres[0] in modele_uni[origin].keys():
        
        denom = modele_uni[origin][lettres[0]]

    elif lettres[0] == BOS:
    
        denom = len(names_by_origin[origin])
        
    else:
        
        denom = 1
        
    denom += len(modele_uni[origin].keys())
    
    if bi in modele_bi[origin].keys():
        
        num = modele_bi[origin][bi] + 1
        
    else:
        
        num = 1
        
    return num/denom

def probTri(tri, origin):
    
    lettres = []
    lettres.extend(tri)
    
    if lettres[0] + lettres[1] in modele_bi[origin].keys():
        
        denom = modele_bi[origin][lettres[0] + lettres[1]]
    
    elif lettres[0] + lettres[1] == BOS + BOS:
        
        denom = len(names_by_origin[origin])
    
    else:
        
        denom = 1
        
    denom += len(modele_uni[origin].keys())
        
    if tri in modele_tri[origin].keys():
        
        num = modele_tri[origin][tri] + 1
        
    else:
        
        num = 1
        
    return num/denom

def logprob(name, origin, n=3):
    # Retourne la valeur du logprob d'un nom étant donné une origine
    # Utilisez une fonction logarithme en base 2.
    # À compléter...
    
    lettres = []
            
    lettres.extend(unicode_to_ascii(name))
    
    log_probs = []
    
    if n == 1:
        
        for uni in lettres:
            
            log_probs.append(math.log2(probUni(uni, origin)))
        
    lettres.insert(0, BOS)
    lettres.append(EOS)
        
    if n == 2:
        
        for i in range(1, len(lettres)):
            
            bi = lettres[i-1] + lettres[i]
            
            log_probs.append(math.log2(probBi(bi, origin)))
        
    lettres.insert(0, BOS)
    lettres.append(EOS)
        
    if n == 3:
        
        for i in range(2, len(lettres)):
            
            tri = lettres[i-2] + lettres[i-1] + lettres[i]
            
            log_probs.append(math.log2(probTri(tri, origin)))
    
    resultat = math.exp(sum(log_probs))
            
    return resultat


def perplexity(name, origin, n=3):
    # Retourne la valeur de perplexité d'un nom étant donné une origine
    # À compléter...
    
    lettres = []
            
    lettres.extend(unicode_to_ascii(name))
    
    probs = []
    
    if n == 1:
        
        for uni in lettres:
            
            probs.append(1/probUni(uni, origin))
        
        resultat = math.pow(prod(probs), 1/len(lettres))
        
    lettres.insert(0, BOS)
    lettres.append(EOS)
        
    if n == 2:
        
        for i in range(1, len(lettres)):
            
            bi = lettres[i-1] + lettres[i]
            
            probs.append(1/probBi(bi, origin))
        
        resultat = math.pow(prod(probs), 1/len(lettres))
        
    lettres.insert(0, BOS)
    lettres.append(EOS)
        
    if n == 3:
        
        for i in range(2, len(lettres)):
            
            tri = lettres[i-2] + lettres[i-1] + lettres[i]
            
            probs.append(1/probTri(tri, origin))
        
        resultat = math.pow(prod(probs), 1/len(lettres))
            
    
    return resultat


def load_test_names(filename):
    """Retourne un dictionnaire contenant les données à utiliser pour évaluer vos modèles.
       Le dictionnaire contient une liste de noms (valeurs) et leur origine (clé)."""
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data


def evaluate_classifier(filename, n=3):
    """Fonction utilitaire pour évaluer vos modèles. Aucune contrainte particulière.
       Je n'utiliserai pas cette fonction pour l'évaluation de votre travail. """
    test_data = load_test_names(filename)
    # À compléter - Fonction pour l'évaluation des modèles N-grammes.
    # ...
    
    print('Classification via logprob et avec n = {}'.format(n))

    nb_nom_bien_classes = 0
    
    nb_nom = 0
    
    for org, name_list in test_data.items():
        
        compte_bien_classes = 0
        
        for name in name_list:
            
            m_p_o = most_probable_origin(name, n)
            
            if m_p_o == org:
                
                compte_bien_classes += 1
                
        print('{}/{} pour la langue {}'.format(compte_bien_classes, len(name_list), org))
        
        nb_nom_bien_classes += compte_bien_classes
        
        nb_nom += len(name_list)
        
    print('{}/{} pour tout les noms'.format(nb_nom_bien_classes, nb_nom))
    print('Précision : {}'.format(nb_nom_bien_classes/nb_nom))

if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    load_names()
    print("Les {} langues d'origine sont:".format(len(all_origins)))
    print(all_origins)
    print("Les noms chinois sont:")
    print(names_by_origin["Chinese"])

    train_classifiers()
    some_name = "Lamontagne"
    some_origin = most_probable_origin(some_name)
    logprob_temp = logprob(some_name, some_origin)
    perplexity_temp = perplexity(some_name, some_origin)
    print("\nLangue d'origine de {}: ".format(some_name), some_origin)
    print("logprob({}, {}):".format(some_name, some_origin), logprob_temp)
    print("perplexity({}, {}):".format(some_name, some_origin), perplexity_temp)

    test_names = load_test_names(test_filename)
    print("\nLes données pour tester vos modèles sont:")
    for org, name_list in test_names.items():
        print("\t", org, name_list)
    evaluate_classifier(test_filename, 3)

