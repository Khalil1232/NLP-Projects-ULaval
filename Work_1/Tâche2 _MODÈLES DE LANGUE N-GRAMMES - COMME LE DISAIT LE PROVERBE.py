#!/usr/bin/env python
# coding: utf-8

# In[17]:


import json
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.lm.models import Laplace
import numpy as np
from nltk.util import pad_sequence 

BOS = '<BOS>'
EOS = '<EOS>'
proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes.txt"


def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data

def build_vocabulary(text_list):
    all_unigrams = list()
    for sentence in text_list:
        word_list = word_tokenize(sentence.lower())
        all_unigrams = all_unigrams + word_list
    voc = set(all_unigrams)
    voc.add(BOS)
    voc.add(EOS)
    return list(voc)


def get_ngrams(text_list, n=2):
    all_ngrams = list()
    for sentence in text_list:
        tokens = word_tokenize(sentence.lower())
        padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=n))
        all_ngrams = all_ngrams + list(ngrams(padded_sent, n=n))      
    return all_ngrams


dicti={}
def train_models(filename):
    proverbs = load_proverbs(filename)
    """ Vous ajoutez à partir d'ici tout le code dont vous avez besoin
        pour construire les différents modèles N-grammes.
        Voir les consignes de l'énoncé du travail pratique concernant les modèles à entraîner.

        Vous pouvez ajouter au fichier les classes, fonctions/méthodes et variables que vous jugerez nécessaire.
        Il faut au minimum prévoir une variable (par exemple un dictionnaire) 
        pour conserver les modèles de langue N-grammes après leur construction. 
        Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    """

    # Votre code à partir d'ici...
    #dicti={}
    for i in range(1,4):
        vocabulary = build_vocabulary(proverbs)
        corpus_ngrams = get_ngrams(proverbs, n=i)
        model = Laplace(i)
        model.fit([corpus_ngrams], vocabulary_text=vocabulary)
        dicti[i]=model
        

def cloze_test(incomplete_proverb, choices, n=3, criteria="perplexity"):
    """ Fonction qui complète un texte à trous (des mots masqués) en ajoutant le bon mot.
        En anglais, on nomme ce type de tâche un "cloze test".

        Le paramètre criteria indique la mesure qu'on utilise pour choisir le mot le plus probable: "logprob" ou "perplexity".
        La valeur retournée est l'estimation sur le proverbe complet (c.-à-d. toute la séquence de mots du proverbe).

        Le paramètre n désigne le modèle utilisé.
        1 - unigramme NLTK, 2 - bigramme NLTK, 3 - trigramme NLTK
    """
    prev={}
    train_models(proverbs_fn)
    for choice in choices:
        Kab=incomplete_proverb.replace("***",choice)
        kab=Kab.split()
        Sec=get_ngrams(kab, n)
        if criteria=="perplexity":
            c=dicti[n].perplexity(Sec)
        else :
            sek=incomplete_proverb.split()
            Sek=get_ngrams(sek, n)
            c=dicti[n].logscore(Sek,[choice])
        prev[c]=Kab
    scores=list(prev.keys())
    
    
    
    # Votre code à partir d'ici.Vous pouvez modifier comme bon vous semble.
    logprob_value = np.max(scores)
    perplexity_value = np.min(scores)
    #result = "qui vivra verra"  # modifier

    if criteria == "perplexity":
        score = perplexity_value
        result=prev[score]
    else:
        score = logprob_value
        result=prev[score]
    return result, score

train_models(proverbs_fn)

if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    proverbs = load_proverbs(proverbs_fn)
    print("\nNombre de proverbes pour entraîner les modèles : ", len(proverbs))
    train_models(proverbs_fn)

    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")
    for partial_proverb, options in test_proverbs.items():
        solution, valeur = cloze_test(partial_proverb, options, n=1, criteria="perplexity")
        print("\n\tProverbe incomplet: {} , Options: {}".format(partial_proverb, options))
        print("\tSolution = {} , Valeur = {}".format(solution, valeur))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




