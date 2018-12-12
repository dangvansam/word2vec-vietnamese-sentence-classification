from pyvi.ViTokenizer import tokenize
import re, os, string
import pandas as pd
import numpy as np

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

# list stopwords
filename = './data/stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords']

def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)

    return text2

def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents


def word_segment(sent):
    sent = tokenize(sent)
    return sent

#path_to_corpus = '../wikipediacorpus'
#load data from xlsx file
input=pd.read_excel('./data/data_goc.xlsx',encoding='utf-8')
listsent=input['text'].values

print("\nLoaded ",len(listsent),"sentences from data/data_goc.xlsx.")
#with open('text_goc.txt', 'w',encoding='utf-8') as f:
#    for item in listsent:
#        f.write("%s\n" % item)
#tokenize and clean text save to datatrain.txt
f_w = open('./data/datatrain.txt', 'w',encoding='utf-8')
dem=0;
listsent2=[]
for sent in listsent:
    sent= clean_text(sent)
    #sents = sentence_segment(content)
    if(sent != None):
        sent = word_segment(sent)
        sent = remove_stopword(normalize_text(sent))            
        #if(len(sent.split()) > 1):
        #     f_w.write("%s\n" % sent)
    listsent2=np.append(listsent2,sent)
print("\nProcessed ",len(listsent),"sentences")
with open('./data/datatrain.txt', 'w',encoding='utf-8') as f:
    for item in listsent2:
        f.write("%s\n" % item)
print("\nWrite to file: data/datatrain.txt")
f_w.close()