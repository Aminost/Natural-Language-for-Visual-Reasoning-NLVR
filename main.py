import nltk
import json

from nltk import jaccard_distance, ngrams, edit_distance
from num2words import num2words
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

train_file = open("D:/NLP_Project_/dataset/nlvr-master/nlvr/train/train.json")
dev_file = open("D:/NLP_Project_/dataset/nlvr-master/nlvr/dev/dev.json")
hidden_file = open("D:/NLP_Project_/dataset/nlvr-master/nlvr/hidden/hidden.json")
test_file = open("D:/NLP_Project_/dataset/nlvr-master/nlvr/test/test.json")

nltk.download('words')
nltk.download('stopwords')
from nltk.corpus import words

ps = PorterStemmer()  # reducing or chopping the words into their root forms
correct_words = words.words()  # correct words

data_train = train_file.read().split("\n")
train_json = []
tokenizer = RegexpTokenizer(r'\w+')

for line in data_train:
    line_json = json.loads(line)
    sentence = line_json["sentence"]
    print(sentence)

    # preprocess
    text = tokenizer.tokenize(sentence)
    py_nltk1 = stopwords.words('english')  # removing stop words
    text_nosp = [t for t in text if t not in py_nltk1]
    for i, word in enumerate(text_nosp):

        word = word.strip()
        word = word.lower()
        if word.isdigit():
            text_nosp[i] = num2words(word)
        else:

            temp = [(edit_distance(word, w), w) for w in correct_words if w[0] == word[0]]
            # print(len(temp))
            temp = sorted(temp)


            text_nosp[i] = temp[0][1]


        text_nosp[i] = ps.stem(text_nosp[i])

    train_json.append({"sentence": text_nosp, "label": line_json["label"], "identifier": line_json["identifier"]})
    print(text_nosp)
for line in train_json:
    print(line)





# @todo
"""
save in sajon file 
maybe make a script that do the preprocessing
"""
"""
data_dev = json.load(dev_json)

data_hidden=hidden_json.read()
data_hidden=data_hidden.split("\n")

data_test=test_json.read()
data_test=data_test.split("\n")
"""
