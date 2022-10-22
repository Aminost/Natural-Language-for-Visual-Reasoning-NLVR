# Opening JSON file
import json
from gensim.models import Word2Vec

sentence = []
f = open('preprocessed-dataset/preprocessed_train.json', 'r')
data = json.load(f)
word_list=[]
for i in  data:
    word_list.append(i["sentence"])

w = Word2Vec(word_list, min_count=1, sg=0, window=8)
print(w.wv.most_similar('black'))
print(w.wv["one"])
print(w.vector_size)
print(len(w.wv.index_to_key))