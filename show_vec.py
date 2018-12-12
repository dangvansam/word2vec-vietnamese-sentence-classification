import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('./model/w2v.bin', binary=True, encoding='utf-8')
print(model.wv.vocab)
print(model["cá"])
print(model.most_similar("tôm"))