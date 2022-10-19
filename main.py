import nltk
import json
from nltk import edit_distance
from num2words import num2words
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('words')
nltk.download('stopwords')
from nltk.corpus import words

train_file = open("dataset/nlvr-master/nlvr/train/train.json")

ps = PorterStemmer()  # reducing or chopping the words into their root forms
correct_words = words.words()  # correct words
stopwords_list = stopwords.words('english')  # stopwords
tokenizer = RegexpTokenizer(r'\w+')  # punctuation removal and tokenization

data_train = train_file.read().split("\n")
train_json = []

# Preprocessing one line at a time and adding it to a list to create an actual json
lines_preprocessed = 0
for line in data_train:
    line_json = json.loads(line)  # reading the line as a json object
    sentence = line_json["sentence"]

    text = tokenizer.tokenize(sentence)  # removing punctuation and tokenizing
    text_nosp = [t for t in text if t not in stopwords_list]  # removing stopwords
    for i, word in enumerate(text_nosp):
        word = word.strip()  # removing extra spaces
        word = word.lower()
        if word.isdigit():
            text_nosp[i] = num2words(word)  # replacing numbers with words
        else:
            temp = [(edit_distance(word, w), w) for w in correct_words if w[0] == word[0]]
            temp = sorted(temp)
            text_nosp[i] = temp[0][1]  # correcting spelling mistakes

        text_nosp[i] = ps.stem(text_nosp[i])  # stemming

    # Formatting the data into a proper json array
    train_json.append({"sentence": text_nosp, "label": line_json["label"], "identifier": line_json["identifier"]})

    # Saving every 100 lines just in case
    lines_preprocessed += 1
    print(f"Line {lines_preprocessed}/{len(data_train)} preprocessed")
    if lines_preprocessed % 100 == 0:
        with open('preprocessed_train.json', 'w') as f:
            json.dump(train_json, f)

# Writing all the preprocessed data into a new json
with open('preprocessed_train.json', 'w') as f:
    json.dump(train_json, f)