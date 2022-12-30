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




# fonction pour analyser et classifier chaque phrase d'un fichier contenant les clauses 
def doc_analysis(file_name,classifier) : 
    sentences = open(file_name,"r") 
    sentences = sentences.readlines()
    for i in range(len(sentences)) :
        # enlever le caractère "\n" de chaque phrase 
        sentences[i]= sentences[i].strip() 
        # effectuer la tokenisation de chaque prase 
        sentences[i] = word_tokenize(sentences[i]) 
        # effectuer un TF_IDF de chaque phrase puis déterminer si les mots de la phrase sont 
        # présent dans les mots les plus cités 
        sentences[i] = bag_of_words(sentences[i]) 
        #classifier la phrase 
        prob_result = classifier.prob_classify(sentences[i])
        # afficher la probablités d'appartenance à la classe de la phrase 
        print("max proba prédiction for clause number :",i+1,"is", custom_classify(prob_list_generator(prob_result))[0])
        # afficher la classe prédite de la phrase 
        print("prediction for clause numbre :", i+1,"is" ,custom_classify(prob_list_generator(prob_result))[1])

    return sentences 


# générer la liste des probabilités d'appartenance à une phrase 
def prob_list_generator(prob_result):  
    # liste de toutes les classes 
    class_list = ["ltd","law","ch","use","ter","j","a"] 
    prob_list=[]
    for item in class_list :  
        # afficher la probabilités retournée pour chaque classe 
        prob_list.append([prob_result.prob(item),item])

    return prob_list



# bad idea (cette fonction qui fait de la data augmentation par 
# suppression aléatoire de mots n'a pas été utilisé dans le cadre de ce projet )
def random_deletion(words, p):

    words = words.split()
    
    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    sentence = ' '.join(new_words)
    
    return sentence


# fonction qui retourne la probabilité la plus haute d'appartenance d'une phrase à une clsse 
def custom_classify(proba_list):  
    # initialiser max_proba  
    max_proba = -1  

    # itérer sur la liste des probabilités retourné par le classifieur 
    for i in range(len(proba_list)) : 
        # déterminer max_proba de la liste 
        if (proba_list[i][0] > max_proba) : 
            # retourner la valeur de max_proba et sa classe associée 
            max_proba= proba_list[i][0]
            prediction = proba_list[i][1]

    return max_proba , prediction 



# fonction pour trouver le synonymes de mots 
def get_synonyms(word):
    synonyms = set()
    
    #itérer dans wordnet pour chercher le synonyme de chaque word 
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)


# fonction pour remplacer des mots par leurs synonymes 
def synonym_replacement(words):
    
    words = words.split()
    copy = words
    # si la taille du mot est suppérieur à 4, remplacer par un synonyme 
    for i in range(len(words)):  
        if (len(words[i]))> 4 :
            syn= get_synonyms(words[i])

            # si la liste de sinonymes est non vide, en choisir un au hasard 
            if syn != []: 
                #print(len(syn))
                #print(random.choice(syn)) 
                copy[i]= random.choice(syn)

    # retourner la phrase remplacée par des synonymes aléatoirement 
    return copy


# lire tout les fichiers de notre data_set  
# les phrases sont dans le dossier /sentences 
# les tags associés à chaque phrase est dans le dossier /tags 
def sentence_read(file_name) : 

    # lire les fichiers de sentences 
    sentence_dir = "sentences/"+file_name  
    #lire les fichiers des tags 
    tags_dir= "tags/"+file_name

    #ouvrir sentences en mode lecture 
    data = open(sentence_dir, "r") 
    data_tags = open(tags_dir, "r") 

    # lire chaque fichiers phrase par phrase 
    data_extracted= data.readlines()  
    data_tags_extracted= data_tags.readlines()  

    # retourner chaque phrase avec sa classe 
    return data_extracted, data_tags_extracted



monRepertoire = "sentences"
fichiers = [f for f in listdir(monRepertoire) if isfile(join(monRepertoire, f))]


sentences = [] 
tags = []
# instancier un Tokeniser 
tokenizer = TreebankWordTokenizer()

for items in fichiers :
    i=0  
    # lire les phrases 
    data_extracted= sentence_read(items)[0] 
    # les les classes associés à chaque phrase 
    data_tags_extracted= sentence_read(items)[1]

    for lines in data_tags_extracted : 
        i+= 1 
        if (lines !=  "\n") :   
            # associer à chaque sentences[i] sont tag[i]
            sentences.append(data_extracted[i-1].strip())
            tags.append(lines.strip())
            #print(lines,data_extracted[i-1]) " lier chaque text à sa classe "

# tableau de data augmentation pour chaque classe 
aug_set= []
# niveau de gravité, le niveau " " est celui des données récoltés pour l'augmentation 
levels = [" ","1","2","3"] 
# classes à augmenter 
aug_class_list = ["law","a","use","ter","j"] 

# pour chaque classe, doubler le dataset en y ajoutant des synonymes 
# rassembler également tout les niveau d'itensité de chaque classe 
# exemple : rassembler : law1, law2, law3 en law 
for aug_class in aug_class_list : 
    for i in range(len(sentences)):  

        for j in range(len(levels)): 
            if (tags[i] == (aug_class+levels[j]).lstrip()) :
                aug_set.append((synonym_replacement(sentences[i]),aug_class))     

#code plus explicite 
""" if ((tags[i]=="a") or (tags[i]=="a2") or (tags[i]=="a3")):
        a_aug.append((synonym_replacement(sentences[i]),"a"))


    if ((tags[i]=="use2") or (tags[i]=="use3")):
        use_aug.append((synonym_replacement(sentences[i]),"use"))

    if ((tags[i]=="ter2") or (tags[i]=="ter3")):
        ter_aug.append((synonym_replacement(sentences[i]),"ter"))


    if ((tags[i]=="ch2") or (tags[i]=="ch3")):
        ch_aug.append((synonym_replacement(sentences[i]),"ch"))


    if ((tags[i]=="j") or (tags[i]=="j2") or (tags[i]=="j3")):
        j_aug.append((synonym_replacement(sentences[i]),"j")) """


# effectuer la tokenisation de chaque phrase (clause)
for i in range(len(sentences)) : 
    sentences[i] = tokenizer.tokenize(sentences[i])

# fusionner chaque clause et sa classe associée dans un seul set (docs)
docs = list(zip(sentences, tags))

# ajoutée à docs la data augmentée 
docs = docs + aug_set
#print(len(docs))

stopwords_english = stopwords.words('english')

def bag_of_words(words):
    words_clean = []

    for word in words: 
        # obtenir tout les mots en minuscule 
        word = word.lower() 
        # si le mot n'est pas une ponctuation ou n'est pas dans stopwords_english 
        # stowords_english sont tout les mots dans l'ensemble " was, the , he , she..etc"
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
    # words_dictionary contient tout les mots qui ne sont pas dans la ponctuation ou stopwords
    words_dictionary = dict([word, True] for word in words_clean)
    
    return words_dictionary


all_classes_list = ["a","ch","j","law","ltd","ter","use"] 

a_set,ch_set,cr_set,j_set,law_set,ltd_set,ter_set,use_set =[],[],[],[],[],[],[],[]


# unifier chaque classe law1 law2 law3 en law  
"""
for i in range(len(docs)): 

    for j in range(len(levels)):
        if (docs[i][1]==("a"+levels[j]).lstrip()):  
            a_set.append([bag_of_words(docs[i][0]),"a"])
        if (docs[i][1]==("ch"+levels[j]).lstrip()):  
            ch_set.append([bag_of_words(docs[i][0]),"ch"])
        if (docs[i][1]==("j"+levels[j]).lstrip()):  
            j_set.append([bag_of_words(docs[i][0]),"j"])
        if (docs[i][1]==("law"+levels[j]).lstrip()):  
            law_set.append([bag_of_words(docs[i][0]),"law"])
        if (docs[i][1]==("ltd"+levels[j]).lstrip()):  
            ltd_set.append([bag_of_words(docs[i][0]),"ltd"])
        if (docs[i][1]==("ter"+levels[j]).lstrip()):  
            ter_set.append([bag_of_words(docs[i][0]),"ter"])
        if (docs[i][1]==("use"+levels[j]).lstrip()):  
            use_set.append([bag_of_words(docs[i][0]),"use"])

#code plus explicite 
"""
a1_set ,a2_set ,a3_set = [],[],[]
ch1_set, ch2_set ,ch3_set= [],[],[]
cr1_set, cr2_set , cr3_set= [],[],[]
j1_set, j2_set, j3_set = [],[],[]
law1_set, law2_set, law3_set = [],[],[]
ltd1_set,  ltd2_set , ltd3_set= [],[],[]
ter1_set, ter2_set , ter3_set = [],[],[]
use1_set, use2_set, use3_set= [],[],[]

for i in range(len(docs)): 
    if (docs[i][1]=="a"): 
        a1_set.append([bag_of_words(docs[i][0]),'a'])
    if (docs[i][1]=="a2"): 
        a2_set.append([bag_of_words(docs[i][0]),'a'])
    if (docs[i][1]=="a3"): 
        a3_set.append([bag_of_words(docs[i][0]),'a'])


    #if (docs[i][1]=="ch1"): 
        #ch1_set.append([bag_of_words(docs[i][0]),'ch1'])
    if (docs[i][1]=="ch2"): 
        ch2_set.append([bag_of_words(docs[i][0]),'ch'])
    if (docs[i][1]=="ch3"): 
        ch3_set.append([bag_of_words(docs[i][0]),'ch'])


    if (docs[i][1]=="j"): 
        j1_set.append([bag_of_words(docs[i][0]),'j'])
    if (docs[i][1]=="j2"): 
        j2_set.append([bag_of_words(docs[i][0]),'j'])
    if (docs[i][1]=="j3"): 
        j3_set.append([bag_of_words(docs[i][0]),'j'])


    if (docs[i][1]=="law"): 
        law1_set.append([bag_of_words(docs[i][0]),'law'])
    if (docs[i][1]=="law2"): 
        law2_set.append([bag_of_words(docs[i][0]),'law'])
    if (docs[i][1]=="law3"): 
        law3_set.append([bag_of_words(docs[i][0]),'law'])


    if (docs[i][1]=="ltd"): 
        ltd1_set.append([bag_of_words(docs[i][0]),'ltd'])
    if (docs[i][1]=="ltd2"): 
        ltd2_set.append([bag_of_words(docs[i][0]),'ltd'])
    if (docs[i][1]=="ltd3"): 
        ltd3_set.append([bag_of_words(docs[i][0]),'ltd'])


    if (docs[i][1]=="ter"): 
        ter1_set.append([bag_of_words(docs[i][0]),'ter'])
    if (docs[i][1]=="ter2"): 
        ter2_set.append([bag_of_words(docs[i][0]),'ter'])
    if (docs[i][1]=="ter3"): 
        ter3_set.append([bag_of_words(docs[i][0]),'ter'])


    if (docs[i][1]=="use"): 
        use1_set.append([bag_of_words(docs[i][0]),'use'])
    #if (docs[i][1]=="use1"): 
        #use1_set.append([bag_of_words(docs[i][0]),'use1'])
    if (docs[i][1]=="use2"): 
        use2_set.append([bag_of_words(docs[i][0]),'use'])
    if (docs[i][1]=="use3"): 
        use3_set.append([bag_of_words(docs[i][0]),'use'])
  


a_set = a1_set + a2_set + a3_set 
ch_set = ch1_set+ ch2_set + ch3_set 
cr_set = cr1_set+ cr2_set + cr3_set
j_set= j1_set+ j2_set+ j3_set 
law_set = law1_set+  law2_set+ law3_set 
ltd_set = ltd1_set+  ltd2_set + ltd3_set
ter_set= ter1_set+  ter2_set + ter3_set
use_set = use1_set+  use2_set+ use3_set


# pour train_set, la même taille de données pour chaque classe sera selectionné, à savoir 60 clauses
train_set=(law_set[0:60] + ltd_set[0:60] + ter_set[0:60]+ use_set[0:60] + 
              j_set[0:60]+ a_set[0:60]+ ch_set[0:60])

# le reste sera pour test_set 
test_set= (law_set[60:72]+ ltd_set[61:72] + ter_set[61:72]+ use_set[61:72] 
            +j_set[61:72] + a_set[61:72] + ch_set[0:72])


# mélanger les données 
random.shuffle(train_set)
random.shuffle(test_set)

print("size of use dataSet",len(use_set)) #120
print("size of ter dataSet", len(ter_set)) #174
print("size of law dataSet",len(law_set)) #105 # revoir law augmentation 
print("size of ltd dataSet",len(ltd_set)) #218
print("size of ch  dataSet",len(ch_set)) #76 
print("size of j dataSet",len(j_set)) # 168 
#rint(len(cr_set)) #0
print("size of a dataSet", len(a_set)) # 100

j = 3 
a = 10 

#Naive Bayse Classifier 
classifier1 = NaiveBayesClassifier.train(train_set)
accuracy = classify.accuracy(classifier1, test_set)
print("accuracy for NaiveBayes", accuracy) # Output: 0.7325


# Decision TReeClassifier 
classifier = nltk.classify.DecisionTreeClassifier.train( train_set, 
                                entropy_cutoff=0, support_cutoff=0)
accuracy = classify.accuracy(classifier, test_set)
print("Accuracy for DecisionTree \n\n", accuracy)

print("\n\n")



#print("max proba:", custom_classify(prob_list_generator(prob_result))[0])
#print("prediction:", custom_classify(prob_list_generator(prob_result))[1])


# fonction pour analyser un fichier de clause et afficher la classe de chaque clause 
sent =doc_analysis("phrase.txt",classifier1)
#print(stopwords_english)