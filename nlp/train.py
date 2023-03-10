from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import movie_reviews
import nltk
import random
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
import string
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import word_tokenize
from random import shuffle
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.corpus import wordnet
import os
import re

import pickle


def class_names_long():
    classes_long = ['Limitation of liability', 'Choice of law', 'Unilateral change', 'Use by contract',
                    'Unilateral terminaison', 'Juridiction', 'Arbitration']
    return dict(zip(class_names(), classes_long))


def class_names():
    return ["ltd", "law", "ch", "use", "ter", "j", "a"]


def train_model(sentences_path: str, tags_path: str):
    fichiers = [f for f in listdir(sentences_path) if isfile(join(sentences_path, f))]

    sentences = []
    tags = []
    # instancier un Tokeniser 
    tokenizer = TreebankWordTokenizer()

    # parcourir les fichiers
    for items in fichiers:
        i = 0
        # lire les phrases  
        data_extracted = sentence_read(os.path.join(sentences_path, items))[0]
        # les classes associées à chaque phrase
        data_tags_extracted = sentence_read(os.path.join(tags_path, items))[1]

        for lines in data_tags_extracted:
            i += 1
            if lines != "\n":
                # associer à chaque sentences[i] sont tag[i]
                sentences.append(data_extracted[i - 1].strip())
                tags.append(lines.strip())
                # print(lines,data_extracted[i-1]) " lier chaque text à sa classe "
            if lines == "\n":
                sentences.append(data_extracted[i - 1].strip())
                tags.append("other")

                # tableau de data augmentation pour chaque classe
    aug_set = []
    # niveau de gravité, le niveau " " est celui des données récoltées pour l'augmentation
    levels = [" ", "1", "2", "3"]
    # classes à augmenter 
    aug_class_list = ["law", "a", "use", "ter", "j"]

    # pour chaque classe, doubler le dataset en y ajoutant des synonymes 
    # rassembler également tous les niveaux d'intensité de chaque classe
    # exemple : rassembler : law1, law2, law3 en law 
    for aug_class in aug_class_list:
        for i in range(len(sentences)):

            for j in range(len(levels)):
                if tags[i] == (aug_class + levels[j]).lstrip():
                    aug_set.append((synonym_replacement(sentences[i]), aug_class))

    # effectuer la tokenisation de chaque phrase (clause)
    for i in range(len(sentences)):
        sentences[i] = tokenizer.tokenize(sentences[i])

    # fusionner chaque clause et sa classe associée dans un seul set (docs)
    docs = list(zip(sentences, tags))

    # ajoutée à docs la data augmentée 
    docs = docs + aug_set
    # print(len(docs))

    a_set, ch_set, cr_set, j_set, law_set, ltd_set, ter_set, use_set = [], [], [], [], [], [], [], []

    for i in range(len(docs)):
        use_set.append([bag_of_words(docs[i][0]), re.sub(r'\d', '', docs[i][1])])

    train_set = (law_set[0:60] + ltd_set[0:60] + ter_set[0:60] + use_set[0:60] +
                 j_set[0:60] + a_set[0:60] + ch_set[0:60])

    # le reste sera pour test_set 
    test_set = (law_set[60:72] + ltd_set[61:72] + ter_set[61:72] + use_set[61:72]
                + j_set[61:72] + a_set[61:72] + ch_set[0:72])

    # mélanger les données
    random.shuffle(train_set)
    random.shuffle(test_set)

    # Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    print(accuracy)

    return classifier


def doc_analysis(file_name, classifier):
    sentences = open(file_name, "r")
    sentences = sentences.readlines()
    for i in range(len(sentences)):
        # enlever le caractère "\n" de chaque phrase 
        sentences[i] = sentences[i].strip()
        # effectuer la tokenisation de chaque phrase
        sentences[i] = word_tokenize(sentences[i])
        # effectuer un TF_IDF de chaque phrase puis déterminer si les mots de la phrase sont 
        # présent dans les mots les plus cités 
        sentences[i] = bag_of_words(sentences[i])
        # classifier la phrase
        prob_result = classifier.prob_classify(sentences[i])
        # afficher la probabilité d'appartenance à la classe de la phrase
        print("max proba prédiction for clause number :", i + 1, "is",
              custom_classify(prob_list_generator(prob_result))[0])
        # afficher la classe prédite de la phrase 
        print("prediction for clause numbre :", i + 1, "is", custom_classify(prob_list_generator(prob_result))[1])

    return sentences


# générer la liste des probabilités d'appartenance à une phrase 
def prob_list_generator(prob_result):
    # liste de toutes les classes 
    class_list = class_names() + ["other"]
    prob_list = []
    for item in class_list:
        prob_list.append([prob_result.prob(item), item])

    return prob_list


# bad idea (cette fonction qui fait de la data augmentation par 
# suppression aléatoire de mots n'a pas été utilisé dans le cadre de ce projet )
def random_deletion(words, p):
    words = words.split()

    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    sentence = ' '.join(new_words)

    return sentence


# fonction qui retourne la probabilité la plus haute d'appartenance d'une phrase à une clsse 
def custom_classify(proba_list):
    # initialiser max_proba
    max_proba = -1

    # itérer sur la liste des probabilités retourné par le classifieur 
    for i in range(len(proba_list)):
        # déterminer max_proba de la liste 
        if proba_list[i][0] > max_proba:
            # retourner la valeur de max_proba et sa classe associée 
            max_proba = proba_list[i][0]
            prediction = proba_list[i][1]

    return max_proba, prediction


# fonction pour trouver le synonymes de mots 
def get_synonyms(word):
    synonyms = set()

    # itérer dans wordnet pour chercher le synonyme de chaque word
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)


def bag_of_words(words):
    words_clean = []
    stopwords_english = stopwords.words('english')

    for word in words:
        # obtenir tous les mots en minuscule
        word = word.lower()
        # si le mot n'est pas une ponctuation ou n'est pas dans stopwords_english 
        # stowords_english sont tout les mots dans l'ensemble " was, the , he , she..etc"
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
    # words_dictionary contient tout les mots qui ne sont pas dans la ponctuation ou stopwords
    words_dictionary = dict([word, True] for word in words_clean)

    return words_dictionary


# fonction pour remplacer des mots par leurs synonymes
def synonym_replacement(words):
    words = words.split()
    copy = words
    # si la taille du mot est suppérieur à 4, remplacer par un synonyme 
    for i in range(len(words)):
        if (len(words[i])) > 4:
            syn = get_synonyms(words[i])

            # si la liste de sinonymes est non vide, en choisir un au hasard 
            if syn:
                # print(len(syn))
                # print(random.choice(syn))
                copy[i] = random.choice(syn)

    # retourner la phrase remplacée par des synonymes aléatoirement 
    return copy


# lire tout les fichiers de notre data_set
# les phrases sont dans le dossier /sentences 
# les tags associés à chaque phrase est dans le dossier /tags 
def sentence_read(path):
    # lire les fichiers de sentences
    sentence_dir = path
    # lire les fichiers des tags
    tags_dir = path

    # ouvrir sentences en mode lecture
    data = open(sentence_dir, "r", encoding='utf-8')
    data_tags = open(tags_dir, "r", encoding='utf-8')

    # lire chaque fichiers phrase par phrase 
    data_extracted = data.readlines()
    data_tags_extracted = data_tags.readlines()

    # retourner chaque phrase avec sa classe 
    return data_extracted, data_tags_extracted


doc_analysis(os.path.join('nlp', "phrase.txt"),
             train_model(os.path.join('nlp', 'sentences'), tags_path=os.path.join('nlp', 'tags')))
