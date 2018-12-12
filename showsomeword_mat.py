import gensim
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt #%matplotlib inline

model = gensim.models.KeyedVectors.load_word2vec_format('model/w2v.bin', binary=True, encoding='utf-8')
print(model.vocab)
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    #string="tôm thẻ chân trắng đông lạnh"
    labels = ["cá","tôm","dầu","tươi", "tôm_hùm","tôm_sú", "trứng", "đông", ]
    tokens = []
    i=1;
    print("Step 1:Loading")
    for word in labels:
        print("",i,"/",len(labels)," vocab")
        i=i+1
        tokens.append(model[word])
        #labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=1)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    i=1
    for value in new_values:
        print("Step 2: ",i,"/",len(new_values))
        i=i+1
        x.append(value[0])
        y.append(value[1])
    j=1  
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        print("Step 3: Showing ",j,"/",len(x))
        j=j+1
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
#tsne_plot(model)