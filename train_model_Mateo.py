import json
import pickle
import re
from operator import itemgetter
import tensorflow as tf
from keras import Sequential, Input, Model
from keras.layers import LSTM, Dense, Dropout, Embedding, add, Flatten
from keras_nlp.layers import TokenAndPositionEmbedding
from nltk.translate.bleu_score import corpus_bleu
from tensorflow import keras
import keras_nlp
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import re
import string
import random
import numpy
import numpy as np
import torch
from keras.utils import pad_sequences
from keras_preprocessing.text import Tokenizer
from torch import device

import preprocessing_utils_v1

def get_structured_representations(data):
    formatted_data = []
    for image in data:
        frames = []
        for List in image['structured_rep']:
            frame = [[0] * 5] * 8
            for i, shape in enumerate(List):
                formatted_shape = [shapes[shape['type']], colors[shape['color']], shape['x_loc'], shape['y_loc'],
                                   shape['size']]
                frame[i] = formatted_shape
            frames += frame
        frames = np.array(frames).flatten()
        formatted_data.append(frames)
    return np.array(formatted_data)


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


def index_to_words(Seq, corpus_tokenizer):
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


def rebuild_Dataset(descriptions, SRs, tokenizer, max_length, vocab_size):
    X1, X2, y = [], [], []
    for i, desc in enumerate(descriptions):
        SR, in_seq, out_seq = create_sequences(tokenizer, max_length, desc, SRs[i], vocab_size)

        X1.extend(SR)
        X2.extend(in_seq)
        y.extend(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# def get_vocab(list):
#     res = []
#     for l in list:
#         [res.append(x) for x in l if x not in res]
#     return  len(res)+1, res

with open('nlvr-dataset/nlvr/train/train.json', 'r') as json_file:
    data_train = [json.loads(line) for line in json_file]
with open('nlvr-dataset/nlvr/dev/dev.json', 'r') as json_file:
    data_dev = [json.loads(line) for line in json_file]
with open('nlvr-dataset/nlvr/test/test.json', 'r') as json_file:
    data_test = [json.loads(line) for line in json_file]

data_train = preprocessing_utils_v1.prepare_data(data_train) + preprocessing_utils_v1.prepare_data(data_dev)
data_test = preprocessing_utils_v1.prepare_data(data_test)

Train_T = [Annot['sentence'] for Annot in data_train]
Test_T = [Annot['sentence'] for Annot in data_test]

Train_T = list(map(preprocessing_utils_v1.preprocess, Train_T))
Test_T = list(map(preprocessing_utils_v1.preprocess, Test_T))

Train_T = ['startseq ' + text + ' endseq' for text in Train_T]
Test_T = ['startseq ' + text + ' endseq' for text in Test_T]

print(Train_T[0], len(Train_T))
print(Test_T[0], len(Test_T))

with open('preprocessed-dataset/preprocessed_train_v3.json', 'r') as json_file:
    pre_Train = [json.loads(line) for line in json_file]
print(type(pre_Train))

# f = open('preprocessed-dataset/preprocessed_train_v3.json', 'r')
# data = json.load(f)
# word_list=[]
# for i in  data:
#     word_list.append(i["sentence"])
# voc_size, voc=get_vocab(word_list)

corpus_tokenizer = tokenization(Train_T)
vocab_size = len(corpus_tokenizer.word_index) + 1

colors = {'#0099ff': corpus_tokenizer.word_index['blue'], 'Yellow': corpus_tokenizer.word_index['yellow'],
          'Black': corpus_tokenizer.word_index['black']}
shapes = {'triangle': corpus_tokenizer.word_index['triangle'], 'circle': corpus_tokenizer.word_index['circle'],
          'square': corpus_tokenizer.word_index['square']}

Train_SR = get_structured_representations(data_train + data_dev)
Test_SR = get_structured_representations(data_test)

print(f'voc_size: {vocab_size}')
print(Train_SR.shape, Test_SR.shape)

lens = [len(T.split(' ')) for T in Train_T + Test_T]
Voc = list(set(sum([T.split() for T in Train_T], [])))
max_length = np.array(lens).max()

train_SR, train_x, train_y = rebuild_Dataset(Train_T, Train_SR, corpus_tokenizer, max_length, vocab_size)
print(train_x.shape, train_y.shape, train_SR.shape)

test_SR, test_x, test_y = rebuild_Dataset(Test_T, Test_SR, corpus_tokenizer, max_length, vocab_size)
print(test_x.shape, test_y.shape, test_SR.shape)


def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(120,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    se4 = Dropout(0.5)(se3)

    # decoder model
    decoder1 = add([fe2, se4])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_2(vocab_size, max_length):
    inputs1 = Input(shape=(120,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))

    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256, input_shape=(120, max_length), return_sequences=True)(se1)
   # se3 = Dropout(0.5)(se2)
    se4 = LSTM(256, return_sequences=True)(se2)
    #se5 = Dropout(0.5)(se4)
    se6 = LSTM(256)(se4)
    decoder1 = add([fe2, se6])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = (Dense(vocab_size, activation='softmax')(decoder2))
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# input_1 = Input(shape=(24,5))
# input_2 = Input(shape=(max_length,))
#
# LSTM_1 = LSTM(units=128, input_shape=(24, 5), activation="tanh", recurrent_activation="relu", return_sequences=True)(
#     input_1)
# # RV = RepeatVector(64)(LSTM_1)
# LSTM_2 = LSTM(units=128, activation="tanh", recurrent_activation="relu")(LSTM_1)
# Out1 = Dense(units=max_length, activation='relu')(LSTM_2)
#
# embedding_2 = Embedding(vocab_size, 128, mask_zero=True)(input_2)
# word2 = Dropout(0.1)(embedding_2)
# context2 = LSTM(128)(word2)
# Flat2 = Flatten()(context2)
# Out2 = Dense(units=max_length)(Flat2)


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

model = model_2(vocab_size, max_length)  # define_model(vocab_size, max_length)

checkpoint = ModelCheckpoint('Captions/Meta-{epoch: model.fit03d}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=1,
                             save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min')

CallBacks = [checkpoint, reduce_lr_loss]  #

history = model.fit(x=[train_SR, train_x], y=train_y, batch_size=256, epochs=64)

model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")


# # aggregation = concatenate([Out2, Out3], axis = -1)
# aggregation = add([Out1, Out2])
# decoder = Dense(256, activation='relu')(aggregation)
# output = Dense(vocab_size, activation='softmax')(decoder)
#
# model = Model(inputs=[input_1, input_2], outputs=output)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


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


def evaluate_model(model, descriptions, SRs, tokenizer, max_length):
    actual, predicted = list(), list()
    for i, desc in enumerate(descriptions):
        yhat = generate_desc(model, tokenizer, SRs[i], max_length)
        actual.append(desc)
        predicted.append(yhat.split())

    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


print(test_x.shape, test_y.shape, test_SR.shape)
print(len(Test_T), len(Test_SR), corpus_tokenizer)


# evaluate_model(reconstructed_model, Test_T, Test_SR, corpus_tokenizer, max_length)

def Generate_Captions(model, descriptions, SRs, tokenizer, max_length):
    predicted = list()
    for SR in SRs:
        yhat = generate_desc(model, tokenizer, SR, max_length)
        yhat = re.sub('startseq', '', yhat)
        yhat = re.sub('endseq', '', yhat)
        predicted.append(yhat.strip())
    return predicted


Captions = Generate_Captions(reconstructed_model, Test_T, Test_SR, corpus_tokenizer, max_length)

with open('Generated_Train_Captions64.pkl', 'wb') as output_file:
    pickle.dump(Captions, output_file)

print(Captions)
