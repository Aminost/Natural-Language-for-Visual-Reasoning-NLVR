from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from operator import itemgetter
from nltk.corpus import words as english_words
from nltk.corpus import stopwords
from nltk import edit_distance

tokenizer = RegexpTokenizer(r'\w+')  # punctuation removal and tokenization
ps = PorterStemmer()  # reducing or chopping the words into their root forms
correct_words = english_words.words()  # correct words
stopwords_list = stopwords.words('english')  # stopwords

sentence_correction = {'circl': 'circle', 'exactli': 'exact', 'squar': 'square', 'triangl': 'triangle', 'edg': 'edge',
                       'boxen': 'square', 'togeth': 'together', 'mani': 'many', 'ablest': 'least',
                       'multipl': 'multiple',
                       'singl': 'single', 'anoth': 'another', 'bellow': 'yellow', 'middl': 'middle',
                       'intrud': 'contain', 'everi': 'every', 'nearli': 'near', 'leat': 'least', 'onli': 'only',
                       'tringl': 'triangle', 'blick': 'black', 'leas': 'least', 'abox': 'square', 'rectangl': 'square',
                       'triabl': 'triangle', 'lest': 'least', 'directli': 'direct', 'theer': 'there', 'thee': 'three',
                       'above': 'top', 'grey': 'black', 'below': 'under', 'kind': 'shape', 'posit': 'position',
                       'roof': 'top', 'both': 'two', 'they': 'item', 'object': 'item', 'beneath': 'under',
                       'them': 'item', 'type': 'shape', 'over': 'on', 'line': 'edge', 'single': 'one', 'bottom': 'base',
                       'wall': 'edge', 'lot': 'many', 'multiple': 'many', 'rectangle': 'square', 'box': 'square',
                       'first': 'one', 'second': 'two', 'third': 'three', 'block': 'square', 'side': 'edge',
                       'total': 'all'}
sentence_removal = ['wah', 'iba', 'bae', 'saurel', 'e', 'oe', 'ad', 'opu', 'wir', 'of', 'abl', 'idl',
                    'carli', 'ae', 'i', 'b', 'u', 'adag', 'od', 'bo', 'eagl', 'tchast', 'in', 'trainless']
shape_correction = {'Yellow': 'yellow', '#0099ff': 'blue', 'Black': 'black'}
numbers = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight'}


def prepare_data(data, all=False):
    tuple_keys = ('sentence', 'structured_rep', 'label')
    get_keys = itemgetter(*tuple_keys)
    if all:
        data = [dict(zip(tuple_keys, get_keys(annotation))) for annotation in data]
    else:
        data = [dict(zip(tuple_keys, get_keys(annotation))) for annotation in data if
                annotation['label'] == 'true']
    return data


def preprocess(text):
    text = text.lower().strip()
    words = tokenizer.tokenize(text)  # removing punctuation and tokenizing
    words = [t for t in words if
                 t not in stopwords_list and t not in sentence_removal]  # removing stopwords and unwanted words
    for i, word in enumerate(words):
        word = word.strip()  # removing extra spaces
        word = word.lower()
        if word in sentence_correction:  # first manual correction
            word = sentence_correction[word]
        if word in numbers.keys():
            words[i] = numbers[word]
        # replacing numbers with words
        else:
            temp = [(edit_distance(word, w), w) for w in correct_words if w[0] == word[0]]
            temp = sorted(temp)
            temp = temp[0][1]  # correcting spelling mistakes
            temp = ps.stem(temp)  # stemming
            # applying another manual correction after the word has been processed
            if temp in sentence_correction:
                temp = sentence_correction[temp]
            words[i] = temp
    words = [t for t in words if
                 t not in stopwords_list and t not in sentence_removal]  # removing new stopwords
    # and unwanted corrections
    return ' '.join(words)
