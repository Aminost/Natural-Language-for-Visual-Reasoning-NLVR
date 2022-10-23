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

train_file = open("nlvr-dataset/nlvr/train/train.json")

ps = PorterStemmer()  # reducing or chopping the words into their root forms
correct_words = words.words()  # correct words
stopwords_list = stopwords.words('english')  # stopwords
tokenizer = RegexpTokenizer(r'\w+')  # punctuation removal and tokenization
saving_enabled = True  # Set that to true if we have to save the preprocessed data
shape_correction = {'Yellow': 'yellow', '#0099ff': 'blue', 'Black': 'black'}


def save_to_json(lines_number, data, file):
    if saving_enabled:
        with open(file, 'w') as f:
            json.dump(data, f)
        with open("lines_number_backup", 'w') as f: # in case of saving in multiple execution
            json.dump(lines_number, f)



data_train = train_file.read().split("\n")
train_json = []
saved_lines_amount = 0

# In case some lines have already been preprocessed
try:
    saved_file = open("preprocessed-dataset/preprocessed_train_backup.json", 'r')
    saved_lines = json.loads(saved_file.read())
    saved_file.close()
    saved_lines_file = open("lines_number_backup")
    saved_lines_amount = int(saved_lines_file.read())
    saved_lines_file.close()
    if saved_lines_amount != 0:
        train_json = saved_lines

except:
    pass

lines_preprocessed = 0
# Preprocessing one line at a time and adding it to a list to create an actual json
for line in data_train:
    lines_preprocessed += 1
    if lines_preprocessed < saved_lines_amount:
        continue
    line_json = json.loads(line)  # reading the line as a json object

    if line_json["label"] == "false":  # Only use tests labelled as "true"
        continue

    structured_rep = line_json["structured_rep"]
    for i, el in enumerate(structured_rep):
        for j, shape in enumerate(el):
            if shape['color'] in shape_correction.keys():
                shape['color'] = shape_correction[shape['color']]
            el[j] = shape
        structured_rep[i] = el

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
    train_json.append({"sentence": text_nosp, "structured_rep": structured_rep, "identifier": line_json["identifier"]})
    # Saving every 100 lines just in case
    print(f"Line {lines_preprocessed}/{len(data_train)} preprocessed")
    if lines_preprocessed % 100 == 0:
        save_to_json(lines_preprocessed, train_json, 'preprocessed-dataset/preprocessed_train_v2.json')

# Writing all the preprocessed data into a new json
save_to_json(lines_preprocessed, train_json, 'preprocessed-dataset/preprocessed_train_v2.json')
