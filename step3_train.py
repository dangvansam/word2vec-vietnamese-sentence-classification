from pyvi.ViTokenizer import tokenize
import re, os, string
import pandas as pd
import numpy as np
import gensim
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Activation,Conv2D,Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
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
#PADDING AND EMBEDDING SENTENCES BY WORD EMBEDDING
def word_embed_sentences(sentences, max_length=50):
        i=1
        model = gensim.models.KeyedVectors.load_word2vec_format('./model/w2v.bin', binary=True, encoding='utf-8')
        embed_sentences = []
        vocablist=model.vocab
        for sent in sentences:
            if(i%20==0):
                print("Embedding sent:",i,"/",len(sentences))
            i=i+1
            embed_sent = []
            sent=sent.split()
            for word in sent:
               #print("\nword=",word)
               if(word) not in vocablist:
                  print("word not in vocab!")
               else:
                  embed_sent.append(model[word])
            if len(embed_sent) > max_length:
                  embed_sent = embed_sent[:max_length]
            elif len(embed_sent) < max_length:
                  embed_sent = np.concatenate((embed_sent, np.zeros(shape=(max_length - len(embed_sent),100), dtype=float)),axis=0)
            embed_sentences.append(embed_sent)
        return embed_sentences
#-----------------------MAIN---------------------
input=pd.read_excel('./data/data_goc.xlsx',encoding='utf-8') #load data for train
listsent=input['text'].values #load sentence
listlabel=input['label'].values #load label
listlabelsorted =np.sort(np.unique(listlabel))
index=np.arange(0,len(listlabelsorted))
#print(index)
onehot_label=np.eye(len(listlabelsorted))[index]
dict_label=dict(zip(listlabelsorted,onehot_label))
#(dict_label)
print("Loaded ",len(listsent),"sentences from data/data_goc.xlsx.Cleanning and tokenizing...")
listsent2=[] #store sentence cleaned
i=1
print("Preprocessing sentence:")
for sent in listsent:
    if(i%20==0):
        print("Processed sent:",i,"/",len(listsent))
    i=i+1
    sent= clean_text(sent)
    if(sent != None):
        sent = word_segment(sent)
        sent = remove_stopword(normalize_text(sent))            
    listsent2=np.append(listsent2,sent)
print("Processed ",len(listsent),"sentences")
df = pd.DataFrame({"label" : listlabel, "text" : listsent2})
df.to_csv("./data/data_goc_cleaned.csv", encoding='utf-8',index=False)
print("Write to file: data/data_goc_cleaned.csv")

#embedding sentence
#model_w2v = gensim.models.KeyedVectors.load_word2vec_format('./model/w2v.bin', binary=True, encoding='utf-8')
sen_embedded= word_embed_sentences(listsent2,max_length=50)
print("number of sent in listsent2=",len(listsent2))
print("number of sent converted to vec=",len(sen_embedded))
#for sen in listsent2:
#   print(sen," to vector:\n")
#   print(sen_embedded[i].shape)
#   i=i+1
#FINAL DATA
Y=listlabel
print("label: ",listlabel)
labels = []
#dict label in Y to onehot vector
for lab_ in Y:						
		if dict_label is None:
			labels.append(lab_)
		else:
			labels.append(dict_label[lab_])
#print(labels[1])
y_final=np.array(labels)
x_final=np.array(sen_embedded)
print("label: ",y_final[1])
print("X shape:",x_final.shape)
print("y shape:",y_final.shape)
#Build model
#input_dim= #input_dim: input dimension max_length x word_dim
word_dim =100 #dim of vector word
max_length = 50
n_epochs = 5
batch_size = 64
n_class =y_final.shape[1] #number of classes
model = Sequential()
#Model1
model.add(Conv2D(128,input_shape=(max_length,word_dim)))
model.add(Activation('relu'))
model.add(Conv2D(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(8)))
model.add(Conv2D(128))
model.add(Activation('relu'))
model.add(Conv2D(128))
model.add(Activation('relu'))
model.add(Conv2D(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(n_class, activation="softmax"))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#Model2

              
"""
Training with data x_final, y_final
x_final: 3D features array, number of samples x max length x word dimension
y_final: 2D labels array, number of samples x number of class
"""
X_train, X_test, y_train, y_test = train_test_split( x_final, y_final, test_size=0.33, random_state=42)
model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(X_test,y_test))
model.save_weights('./model/model.h5')