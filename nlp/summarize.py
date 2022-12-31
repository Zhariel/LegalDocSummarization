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
from nlp.train import bag_of_words, class_names

def summarize_contract(model: NaiveBayesClassifier, document: str):
    letters_sub = r'[^a-zA-Z.]'
    text = re.sub(letters_sub, ' ', document)
    text = re.sub(' +', ' ', text)
    text = text.split('.')
    text = [t for t in text if len(t) > 2]
    tokenizer = TreebankWordTokenizer()
    classes = class_names()

    kept_sentences = []

    for sentence in text:
        processed_sentence = tokenizer.tokenize(sentence)
        bag = bag_of_words(processed_sentence)
        predicted_class = model.classify(bag)
        if predicted_class in classes:
            kept_sentences.append((predicted_class, sentence))

    return kept_sentences