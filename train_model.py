import json
import pickle
from operator import itemgetter
from keras import Input, Model
from keras.layers import LSTM, Dense, Dropout, Embedding, add
import re
import numpy as np
from keras.utils import pad_sequences
from keras_preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers


def Structured_representations_Features(Annotations):
    Features = []
    for Annot in Annotations:
        Regions = []
        for List in Annot['structured_rep']:
            Vec = [[0] * 5] * 8
            for i, O in enumerate(List):
                Object = []
                Object.append(Shapes[O['type']])
                Object.append(Colors[O['color']])
                Object.append(O['x_loc'])
                Object.append(O['y_loc'])
                Object.append(O['size'])
                Vec[i] = Object
            Regions += Vec
        Regions = np.array(Regions).flatten()
        Features.append(Regions)
    return np.array(Features)


def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq


def Index_to_words(Seq, corpus_tokenizer):
    tokens = []
    for i in Seq:
        if i > 0:
            tokens.append(corpus_tokenizer.index_word[i])
    return tokens


def create_sequences(tokenizer, max_length, desc, SR, vocab_size):
    X1, X2, y = list(), list(), list()
    seq = tokenizer.texts_to_sequences([desc])[0]
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]

        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

        X1.append(SR)
        X2.append(in_seq)
        y.append(out_seq)
    return X1, X2, y


def to_categorical(out_seq, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[out_seq]


def Rebuild_Dataset(descriptions, SRs, tokenizer, max_length, vocab_size):
    X1, X2, y = [], [], []
    for i, desc in enumerate(descriptions):
        SR, in_seq, out_seq = create_sequences(tokenizer, max_length, desc, SRs[i], vocab_size)

        X1.extend(SR)
        X2.extend(in_seq)
        y.extend(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


def Preprocessing(text):
    Numbers = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six'}

    text = re.sub('a sthe', 'as the', text)

    Corrections = {'ad': 'and', 'i': 'is', 's': 'is', 'ia': 'is a', 'adge': 'edge', 'al': 'at', 't': 'at',
                   'tleast': 'at least', 'atleast': 'at least', 'ablue': 'a blue', 'ans': 'and', 'bkack': 'black',
                   'bloxk': 'block', 'abox.': 'a box', 'bow': 'box', 'blicks': 'block', 'bo': 'box', 'boxes': 'box',
                   'ble': 'blue',
                   'blacks': 'black', 'blccks': 'black', 'back': 'black', 'bellow': 'below', 'contains': 'contain',
                   'containing': 'conatin', 'opis': 'is', 'isa': 'is a', 'exacrly': 'exactly', 'exacts': 'exactly',
                   'eactly': 'exactly', 'exacty': 'exactly', 'cirlce': 'circle', 'ciircles': 'circles',
                   'cirlce': 'circle',
                   'colour': 'color', 'colours': 'colors', 'coloured': 'color', 'colour.': 'color', 'colored': 'color',
                   'closely': 'close', 'ha': 'have', 'having': 'have',
                   'including': 'include', 'sqaures': 'squares', 'egde': 'edge',

                   'leats': 'least', 'lest': 'least', 'lease': 'least', 'squere': 'square', 'squares': 'square',
                   'touhing': 'touch', 'tocuhing': 'touch', 'traingle': 'triangle', 'traingles': 'triangle',
                   'trianlge': 'triangle',
                   'hte': 'the', 'thee': 'the', 'then,idle': 'the middle', 'theer': 'there', 'od': 'odd',
                   'objetcs': 'objects',
                   'tow': 'two', 'wirh': 'with', 'wwith': 'with', 'wth': 'with', 'wih': 'with',
                   'yelow': 'yellow', 'yelloe': 'yellow', 'yelllow': 'yellow'
                   }

    Corrections2 = {'above': 'top', 'blocks': 'block', 'items': 'item', 'objects': 'item', 'towers': 'tower',
                    'triangles': 'triangle', 'colors': 'color', 'squares': 'square', 'attached': 'attach',
                    'stack': 'stack', 'boxes': 'box', 'shapes': 'shape', 'numbers': 'number', 'corners': 'corner',
                    'positions': 'position', 'bases': 'base', 'kinds': 'shape', 'below': 'under', 'grey': 'black',

                    'object': 'item', 'it': 'item', 'underneath': 'under', 'roof': 'top', 'include': 'contain',
                    'both': 'two',
                    'they': 'item', 'objects': 'item', 'beneath': 'under', 'them': 'item', 'type': 'shape',
                    'over': 'on',
                    'line': 'edge', 'ones': 'item', 'stacked': 'stack', 'single': 'one', 'corners': 'corner',
                    'attached': 'attach',
                    'touching': 'touch', 'bottom': 'base', 'alternately': 'different', 'odd': 'different',
                    'wall': 'edge',
                    'smaller': 'small', 'lot': 'many', 'multiple': 'many', 'none': 'no', 'rectangle': 'square',
                    # 'box':'square',
                    'even': 'same', 'first': 'one', 'second': 'two', 'third': 'three', 'traingles': 'triangle',
                    'circles': 'circle', 'block': 'square', 'total': 'all', 'side': 'edge',
                    }

    text = text.lower()

    words = [word if not word in Numbers.keys() else Numbers[word] for word in text.split(' ')]
    words = [word if not word in Corrections.keys() else Corrections[word] for word in words]
    # words = [word if not word in Corrections2.keys() else Corrections2[word] for word in words]

    text = ' '.join(words)

    text = re.sub('\.', '', text)
    text = re.sub('\/', '', text)
    text = re.sub('^ll ', '', text)
    text = re.sub('-', ' ', text)
    text = re.sub(',', '', text)
    text = re.sub('^t ', 'the ', text)
    text = re.sub(';', 'l', text)

    words = [word if not word in Corrections2.keys() else Corrections2[word] for word in text.split(' ')]

    text = ' '.join(words)

    return text.strip()


def Prepare_Data(Annotations, All=False):
    tuple_keys = ('sentence', 'structured_rep', 'label')
    get_keys = itemgetter(*tuple_keys)
    if All == True:
        Annotations = [dict(zip(tuple_keys, get_keys(Annotation))) for Annotation in Annotations]
    else:
        Annotations = [dict(zip(tuple_keys, get_keys(Annotation))) for Annotation in Annotations if
                       Annotation['label'] == 'true']
    return Annotations


with open('nlvr-dataset/nlvr/train/train.json', 'r') as json_file:
    Annot_Train = [json.loads(line) for line in json_file]
with open('nlvr-dataset/nlvr/dev/dev.json', 'r') as json_file:
    Annot_Val = [json.loads(line) for line in json_file]
with open('nlvr-dataset/nlvr/test/test.json', 'r') as json_file:
    Annot_Test = [json.loads(line) for line in json_file]

Annot_Train = Prepare_Data(Annot_Train) + Prepare_Data(Annot_Val)
Annot_Test = Prepare_Data(Annot_Test)

Train_T = [Annot['sentence'] for Annot in Annot_Train]
Test_T = [Annot['sentence'] for Annot in Annot_Test]
Val_T = [Annot['sentence'] for Annot in Annot_Val]

Train_T = list(map(Preprocessing, Train_T))
Val_T = list(map(Preprocessing, Train_T))
Test_T = list(map(Preprocessing, Test_T))

Train_T = ['startseq ' + text + ' endseq' for text in (Train_T)]
Test_T = ['startseq ' + text + ' endseq' for text in Test_T]
Val_T = ['startseq ' + text + ' endseq' for text in Val_T]

print(Train_T[0], len(Train_T))
print(Test_T[0], len(Test_T))

with open('preprocessed-dataset/preprocessed_train_v3.json', 'r') as json_file:
    pre_Train = [json.loads(line) for line in json_file]
print(type(pre_Train))

corpus_tokenizer = tokenization(Train_T)
vocab_size = len(corpus_tokenizer.word_index) + 1

Colors = {'#0099ff': corpus_tokenizer.word_index['blue'], 'Yellow': corpus_tokenizer.word_index['yellow'],
          'Black': corpus_tokenizer.word_index['black']}
Shapes = {'triangle': corpus_tokenizer.word_index['triangle'], 'circle': corpus_tokenizer.word_index['circle'],
          'square': corpus_tokenizer.word_index['square']}

Train_SR = Structured_representations_Features(Annot_Train+Annot_Val )
Test_SR = Structured_representations_Features(Annot_Test)

print(f'voc_size: {vocab_size}')
print(Train_SR.shape, Test_SR.shape)

lens = [len(T.split(' ')) for T in Train_T + Test_T ]
Voc = list(set(sum([T.split() for T in Train_T], [])))
max_length = np.array(lens).max()

train_SR, train_x, train_y = Rebuild_Dataset(Train_T, Train_SR, corpus_tokenizer, max_length, vocab_size)
print(train_x.shape, train_y.shape, train_SR.shape)

test_SR, test_x, test_y = Rebuild_Dataset(Test_T, Test_SR, corpus_tokenizer, max_length, vocab_size)
print(test_x.shape, test_y.shape, test_SR.shape)


def my_model(vocab_size, max_length):
    inputs1 = Input(shape=(120,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))

    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256, input_shape=(120, max_length), return_sequences=True)(se1)
    # se3 = Dropout(0.5)(se2)
    se4 = LSTM(256, return_sequences=True)(se2)
    # se5 = Dropout(0.5)(se4)
    se6 = LSTM(256)(se4)
    decoder1 = add([fe2, se6])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = (Dense(vocab_size, activation='softmax')(decoder2))
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



model = my_model(vocab_size, max_length)  # define_model(vocab_size, max_length)
history = model.fit(x=[train_SR, train_x], y=train_y, batch_size=256, epochs=64, validation_split=0.2)
model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, SR, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)[0]

        # predict next word
        yhat = model.predict([SR.reshape(1, -1), sequence.reshape(1, -1)], steps=1, verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text



print(test_x.shape, test_y.shape, test_SR.shape)
print(len(Test_T), len(Test_SR))


def Generate_Captions(model, SRs, tokenizer, max_length):
    predicted = list()
    for SR in SRs:
        yhat = generate_desc(model, tokenizer, SR, max_length)
        yhat = re.sub('startseq', '', yhat)
        yhat = re.sub('endseq', '', yhat)
        predicted.append(yhat.strip())

    return predicted

print("generate captions...")
Captions = Generate_Captions(reconstructed_model, Test_SR, corpus_tokenizer, max_length)
print("done!")


with open('Generated_Train_Captions64.pkl', 'wb') as output_file:
    pickle.dump(Captions, output_file)
    print(Captions)

my_seq= encode_sequences(corpus_tokenizer,max_length , Captions)


def plot_result(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

plot_result(history)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions




# the transformer model

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs1 = Input(shape=(120,))

inputs = layers.Input(shape=(max_length,))
embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=[inputs1, inputs], outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x=[train_SR, train_x], y=np.asarray(my_seq), batch_size=32, epochs=2)
