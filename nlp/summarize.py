from nltk.tokenize import TreebankWordTokenizer
from nltk import NaiveBayesClassifier
import re
from nlp.train import bag_of_words, class_names, class_names_long

def summarize_contract(model: NaiveBayesClassifier, document: str):
    letters_sub = r'[^a-zA-Z.]'
    text = re.sub(letters_sub, ' ', document)
    text = re.sub(' +', ' ', text)
    text = text.split('.')
    text = [t for t in text if len(t) > 2]
    tokenizer = TreebankWordTokenizer()
    classes = class_names()
    classes_long = class_names_long()

    kept_sentences = []

    for sentence in text:
        processed_sentence = tokenizer.tokenize(sentence)
        bag = bag_of_words(processed_sentence)
        predicted_class = model.classify(bag)
        if predicted_class in classes:
            kept_sentences.append((classes_long[predicted_class], sentence))

    return kept_sentences

