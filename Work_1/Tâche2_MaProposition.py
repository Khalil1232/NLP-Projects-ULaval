#!/usr/bin/env python
# coding: utf-8

# In[17]:


import json
import nltk
import collections
import time
import math
from collections import defaultdict
from itertools import permutations
from nltk.tokenize import word_tokenize, RegexpTokenizer
from collections import Counter
#from utility import get_complet_path, get_file_content, bcolors, get_file_content_with_br
from math import log
from nltk.util import pad_sequence
# Create the ngram tuples
from nltk.util import ngrams
import os
import sys
BOS = '<BOS>'
EOS = '<EOS>'


# In[18]:


# https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# In[19]:


def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data


# In[20]:


def get_ngram(tokens, ngram_count):
    for i in range(len(tokens) - ngram_count+1):
        tuple = ()
        for ngram in range(ngram_count):
            tuple = tuple + (tokens[i+ngram],)
        yield tuple


# In[21]:


proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes.txt"


# In[22]:


def tokenize_text(text_list):
    all_unigrams = list()
    for sentence in text_list:
        word_list = word_tokenize(sentence.lower())
        all_unigrams = all_unigrams + word_list
    voc = set(all_unigrams)
    voc.add(BOS)
    voc.add(EOS)
    return list(voc)


# In[23]:


def probability_of_sentence(tokens, sentence, laplace=0):

    table_prob = create_table_prob(tokens, 2, laplace)
    tokens_of_sentence = tokenize_text(sentence)
    count_tokenized_sentence = len(tokens_of_sentence)

    probability = 0
    for i in range(count_tokenized_sentence-2):

        search_tuple = (tokens_of_sentence[i], tokens_of_sentence[i+1])
        bigram_prob = next((x for x in table_prob if x[0][0] == search_tuple[0] and x[0][1] == search_tuple[1]), ("UNK",0.0001))

        if (probability == 0):
            probability = math.log(bigram_prob[1])
        else:
            probability += math.log(bigram_prob[1])

    return probability


# In[24]:


def logprob_perplexity_of_sentence(sentence, tokens, laplace=0):

    #print("\n{}## PERPLEXITY OF: {} {}".format(bcolors.HEADER, sentence, bcolors.ENDC))

    n = len(collections.Counter(tokens))
        
    probability = probability_of_sentence(tokens, sentence, laplace)
    perplexity = math.pow(2, probability)

    print("# Log Prob is {}{}{} - Perplexity is {}{}{}".format(
        bcolors.OKBLUE, probability, bcolors.ENDC,
        bcolors.OKBLUE, perplexity, bcolors.ENDC))


# In[25]:


# Create a table of probabilities with unigram/bigram to be used in the predictions as a 
# progressive way to calculate the probability
def create_table_prob(tokens, ngram_number, laplace):

    start_time = time.time()

    # Trying to create an table with all possibly combinations according with the ngram, but it's to slow, 
    # can't make it to finish
    # for group in permutations(tokens, ngram_number):
    #    quantity = 0
    #    if (laplace > 0):
    #        quantity += laplace
    #    table_prob.append((group, quantity))
    ngram = list(get_ngram(tokens, ngram_number))
    
    ngram_counter = collections.Counter(ngram)
    total = len(tokens)

    if (laplace > 0):
        unique = collections.Counter(tokens)
        total = total + len(unique)

    table_prob = []
    for gram in ngram_counter.most_common():
        quantity = gram[1]
        
        if (ngram_number == 1):
            elem_count = total
        else:
            elem_count = tokens.count(gram[0][0])

        if (laplace > 0):
            quantity += laplace
            elem_count += laplace

        table_prob.append((gram[0], quantity / elem_count))

    print("{}# {:.2f} seconds to create table prod with {} elements{}".format(bcolors.WARNING, (time.time() - start_time), len(table_prob), bcolors.ENDC))

    return table_prob


# In[26]:


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
    tokens =  tokenize_text(proverbs)
    tokens.append("UNK")
    #table_prob = create_table_prob(tokens, ngram_number, laplace)
    return tokens


# In[27]:


# Complet phrase with most possible word accordingly to the table of probabilities,
# Unigram and Bigrams (and more grams) are calculate differently
def cloze_test(tokens, proverbs_to_test, ngram_number, laplace=0):
    print("\n{}## NGRAM({}) GUESS | Laplace {}{}".format(bcolors.HEADER, ngram_number, laplace, bcolors.ENDC))

    real_ngram_number = ngram_number
    # Force always to 2 to make a progressive probability, instead of full probability, like done
    # like the complet_proverbe_with_trigram function, for exemple, otherwise probs are too small
    if (ngram_number > 2):
        ngram_number = 2 

    table_prob = create_table_prob(tokens, ngram_number, laplace)

    if (ngram_number == 1):

        for proverb in proverbs_to_test:
            guess_words = proverbs_to_test[proverb]
            
            filtered = [x for x in table_prob if x[0][0] in guess_words]
            filtered.sort(key=lambda elem: elem[1], reverse=True)

            r=(proverb.replace("***",  "{}{}{}") + " | Prob {}{}{}").format(
                    bcolors.OKBLUE, 
                    filtered[0][0][0], 
                    bcolors.ENDC,
                    bcolors.OKGREEN,
                    filtered[0][1],
                    bcolors.ENDC)
            print(r)
            logprob_perplexity_of_sentence(r, tokens, 0)
    else:

        for proverb in proverbs_to_test:
            guess_words = proverbs_to_test[proverb]

            guess_word_prob = dict()
            for guess_word in guess_words:

                for i in range(real_ngram_number-1):

                    tokenized_proverb = tokenize_text(proverb[0:proverb.find("***") + 3])
                    count_tokenized_proverb = len(tokenized_proverb) -1

                    search_tuple = (tokenized_proverb[count_tokenized_proverb-i-1], tokenized_proverb[count_tokenized_proverb-i].replace("***", guess_word))

                    bigram_prob = next((x for x in table_prob if x[0][0] == search_tuple[0] and x[0][1] == search_tuple[1]), ("UNK",0.0000001))

                    if  (bigram_prob[0] == "UNK"):
                        # trying to take the real probability of UNK coming after some word, but it never finds :/, so we stay with the real 
                        # small probability for the moment, just in case we don't find anything else
                        #bigram_prob = [x for x in table_prob if x[0][0] == search_tuple[0] and x[0][1] == "UNK"]
                        guess_word = "UNK"

                    if guess_word in guess_word_prob:
                        guess_word_prob[guess_word] *= bigram_prob[1]
                    else:
                        guess_word_prob[guess_word] = bigram_prob[1]
            
            ordered_probs = sorted(guess_word_prob, key=guess_word_prob.get, reverse=True)

            r=(proverb.replace("***",  "{}{}{}") + " | Prob {}{}{}").format(
                    bcolors.OKBLUE, 
                    ordered_probs[0], 
                    bcolors.ENDC,
                    bcolors.OKGREEN,
                    guess_word_prob[ordered_probs[0]],
                    bcolors.ENDC)
            print(r)
            logprob_perplexity_of_sentence(r, tokens, 0)


# In[29]:


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    proverbs = load_proverbs(proverbs_fn)
    print("\nNombre de proverbes pour entraîner les modèles : ", len(proverbs))
    tokens=train_models(proverbs_fn)

    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")
    

    cloze_test(tokens, test_proverbs, ngram_number=1, laplace=1)
    cloze_test(tokens, test_proverbs, ngram_number=2, laplace=0)
    
    cloze_test(tokens, test_proverbs, ngram_number=3, laplace=0)
    cloze_test(tokens, test_proverbs, ngram_number=3, laplace=10)


# In[ ]:




def main():
    corpus = load_proverbs("proverbes.txt")
    tests = load_tests("test_proverbes.txt")

    tokens =  tokenize_text(corpus)
    tokens.append("UNK")

    #complet_proverbe_with_trigram(tokens, tests)

    # First solution
   # complet_proverbe_with_unigram(tokens, tests)
    #complet_proverbe_with_bigram(tokens, tests)
    #complet_proverbe_with_trigram(tokens, tests)
    
    # Second solution with laplace
    complet_with_ngram(tokens, tests, ngram_number=1, laplace=1)
    complet_with_ngram(tokens, tests, ngram_number=2, laplace=0)
    
    complet_with_ngram(tokens, tests, ngram_number=3, laplace=0)
    complet_with_ngram(tokens, tests, ngram_number=3, laplace=10)

    # Perplexity
    #logprob_perplexity_of_sentence("bon ouvrier ne querelle pas ses voisins", tokens, 0)
    #logprob_perplexity_of_sentence("something not trained in our model", tokens, 0)

if __name__ == '__main__':  
   main()


# In[ ]:




