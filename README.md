input file: data_goc.xlsx
create word2vec model: 
	step 1: python step1_make_data.py #load text, label from excel file clean, tokenize,...
	output: /data/datatrain.txt # list one sentence in one line cleaned, tokenized for train word2vec model
	step 2: python step2_w2vtrain.py #load sentence from /data/datatrain.txt and use gensim create model
	output: model/w2v.bin (BIN file) and model/w2v.txt are model word2vec from data
train model text classification word embedding and keras:
	step 3: python step3_train.py #load text, label from /data/data_goc.xlsx file clean, tokenize,... same step 1
	encode label to onehot vector, converted sentence to vector with word2vec model in step 2 vector size[max_len_sentence,word2vec_dim], here is [50,100], after convert n sentence we have a maxtrix [number_sentence,max_len_sentence,word2vec_dim] is input of CNN network
	output: /model/model.h5 #model of sentence classification, can predict, test,...